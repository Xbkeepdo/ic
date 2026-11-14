#!/usr/bin/env python3
"""First-risk-word-gated residual intervention on 100 seed-43 COCO images."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from models.dgst_capture import (
    _layer_attention_module,
    _layer_mlp_module,
    resolve_decoder_layers,
)
from scripts.generation_time_curve_residual_seed43 import (
    BASELINE,
    CONFIG,
    PROMPT,
    SEED,
    SPLITS,
    _json,
    _rows,
    _scale_prediction_row,
    curves_and_coefficients,
)
from scripts.generation_time_sar_ems_seed43 import RiskScorer, _chair_module, _decode
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/generation_time_gated_residual_seed43/llava_1_5_7b/test_100"
MODEL = "llava_1_5_7b"
LIMIT = 100
METHODS = ("gated_attention", "gated_ffn", "gated_joint")


def strong_coefficients() -> dict:
    curves = curves_and_coefficients(MODEL)
    middle = curves["middle_layers"]
    sar = np.asarray(curves["sar"]["distance"], dtype=np.float64)
    ems = np.asarray(curves["ems"]["distance"], dtype=np.float64)
    attention = np.ones(curves["n_layers"], dtype=np.float64)
    ffn = np.ones(curves["n_layers"], dtype=np.float64)
    attention[middle] += 0.7 * sar[middle] / (sar[middle].max() + 1e-12)
    ffn[middle] -= 0.8 * ems[middle] / (ems[middle].max() + 1e-12)
    curves["attention_scale"] = attention.tolist()
    curves["ffn_scale"] = ffn.tolist()
    curves["coefficient_rule"] = (
        "alpha=1+0.7*d_sar/max_middle(d_sar); "
        "beta=1-0.8*d_ems/max_middle(d_ems)"
    )
    return curves


@contextmanager
def gated_residuals(model, attention_scale: list[float],
                    ffn_scale: list[float], start_index: int):
    """Apply branch scales beginning at response prediction index start_index."""
    layers = resolve_decoder_layers(model)
    if len(layers) != len(attention_scale) or len(layers) != len(ffn_scale):
        raise ValueError("Coefficient count differs from decoder layer count")
    handles = []
    counters = {"attention": [0] * len(layers), "ffn": [0] * len(layers)}
    applied = {"attention": 0, "ffn": 0}
    try:
        for index, layer in enumerate(layers):
            a = float(attention_scale[index])
            f = float(ffn_scale[index])
            if a != 1.0:
                def attention_hook(_module, _args, output, idx=index, factor=a):
                    position = counters["attention"][idx]
                    counters["attention"][idx] += 1
                    if position < start_index:
                        return output
                    applied["attention"] += 1
                    if isinstance(output, tuple):
                        return (_scale_prediction_row(output[0], factor), *output[1:])
                    return _scale_prediction_row(output, factor)
                handles.append(_layer_attention_module(layer).register_forward_hook(attention_hook))
            if f != 1.0:
                def ffn_hook(_module, _args, output, idx=index, factor=f):
                    position = counters["ffn"][idx]
                    counters["ffn"][idx] += 1
                    if position < start_index:
                        return output
                    applied["ffn"] += 1
                    return _scale_prediction_row(output, factor)
                handles.append(_layer_mlp_module(layer).register_forward_hook(ffn_hook))
        yield applied
    finally:
        for handle in handles:
            handle.remove()


def generate(device: str) -> None:
    torch.set_num_threads(1)
    coefficients = strong_coefficients()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coefficients.json").write_text(json.dumps(coefficients, indent=2))
    scorer = RiskScorer(MODEL)
    image_ids = sorted(map(int, _json(SPLITS)["test"]))[:LIMIT]
    baseline = _rows(BASELINE / MODEL / "test/greedy.jsonl")
    done = {method: _rows(OUT / f"{method}.jsonl")
            if (OUT / f"{method}.jsonl").exists() else {} for method in METHODS}
    config = load_config(str(CONFIG))
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    wrapper = build_model(MODEL, get_extraction_model_cfg(config, MODEL), device=device)
    wrapper.model.eval().requires_grad_(False)
    ones = [1.0] * coefficients["n_layers"]
    with Image.open(coco / f"COCO_val2014_{image_ids[0]:012d}.jpg") as opened:
        check = wrapper.generate(opened.convert("RGB"), prompt=PROMPT)
    if list(map(int, check.response_token_ids)) != baseline[image_ids[0]]["response_token_ids"]:
        raise RuntimeError("Current greedy generation differs from paired cached baseline")
    for ordinal, image_id in enumerate(image_ids, 1):
        todo = [method for method in METHODS if image_id not in done[method]]
        if not todo:
            continue
        original = baseline[image_id]
        mentions = original["mentions"]
        risky = [m for m in mentions if m["hall_score"] >= scorer.trigger_threshold]
        trigger = min(risky, key=lambda m: m["token_index"]) if risky else None
        if trigger is not None:
            with Image.open(coco / f"COCO_val2014_{image_id:012d}.jpg") as opened:
                image = opened.convert("RGB")
        for method in todo:
            if trigger is None:
                ids = list(map(int, original["response_token_ids"]))
                caption = original["generated_text"]
                elapsed = 0.0
                applied = {"attention": 0, "ffn": 0}
            else:
                attention = coefficients["attention_scale"] if method != "gated_ffn" else ones
                ffn = coefficients["ffn_scale"] if method != "gated_attention" else ones
                start = time.perf_counter()
                with gated_residuals(wrapper.model, attention, ffn,
                                     int(trigger["token_index"])) as applied:
                    generated = wrapper.generate(image, prompt=PROMPT)
                elapsed = time.perf_counter() - start
                ids = list(map(int, generated.response_token_ids))
                caption = _decode(wrapper, ids, MODEL)
                position = int(trigger["token_index"])
                if ids[:position] != original["response_token_ids"][:position]:
                    raise RuntimeError(f"{image_id}: prefix changed before intervention")
                if method != "gated_ffn" and applied["attention"] == 0:
                    raise RuntimeError("Attention hooks did not apply")
                if method != "gated_attention" and applied["ffn"] == 0:
                    raise RuntimeError("FFN hooks did not apply")
            row = {"image_id": image_id, "method": method, "seed": SEED,
                   "generated_text": caption, "response_token_ids": ids,
                   "trigger": trigger, "seconds": elapsed,
                   "applied_hook_calls": applied,
                   "changed": ids != original["response_token_ids"]}
            with (OUT / f"{method}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{ordinal}/{LIMIT}] {method}: trigger={trigger is not None} "
                  f"changed={row['changed']} seconds={elapsed:.2f}", flush=True)


def evaluate() -> None:
    image_ids = sorted(map(int, _json(SPLITS)["test"]))[:LIMIT]
    baseline = _rows(BASELINE / MODEL / "test/greedy.jsonl")
    rollback = _rows(BASELINE / MODEL / "test/risk_rollback.jsonl")
    variants = {"greedy": {i: baseline[i] for i in image_ids},
                "risk_rollback": {i: rollback[i] for i in image_ids}}
    for method in METHODS:
        source = _rows(OUT / f"{method}.jsonl")
        if any(i not in source for i in image_ids):
            raise ValueError(f"{method}: paired rows missing")
        variants[method] = {i: source[i] for i in image_ids}
    scorer = RiskScorer(MODEL)
    trigger_ids = [i for i in image_ids if any(
        m["hall_score"] >= scorer.trigger_threshold for m in baseline[i]["mentions"])]
    config = load_config(str(CONFIG))
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=BASELINE / "chair_cache.pkl",
    )
    per_image = {}
    summary = {"seed": SEED, "model": MODEL, "images": LIMIT,
               "triggered_images": len(trigger_ids), "methods": {}}
    for method, source in variants.items():
        indexed = {i: evaluator.compute_chair_token(i, source[i]["generated_text"])
                   for i in image_ids}
        per_image[method] = indexed
        rows = list(indexed.values())
        all_mentions = sum(len(r["object_mentions"]) for r in rows)
        hall_mentions = sum(len(r["mscoco_hallucinated_words"]) for r in rows)
        summary["methods"][method] = {
            "chair_s": float(np.mean([r["metrics"]["CHAIRs"] for r in rows])),
            "chair_i": hall_mentions / all_mentions if all_mentions else math.nan,
            "triggered_chair_s": float(np.mean([indexed[i]["metrics"]["CHAIRs"]
                                                for i in trigger_ids])) if trigger_ids else math.nan,
            "object_recall": float(np.mean([r["metrics"]["Recall"] for r in rows])),
            "true_mentions_per_image": float(np.mean([len(r["mscoco_real_words"]) for r in rows])),
            "object_mentions_per_image": all_mentions / LIMIT,
            "no_object_images": sum(not r["object_mentions"] for r in rows),
            "words_per_image": float(np.mean([len(r["words"]) for r in rows])),
        }
        if method in METHODS:
            summary["methods"][method].update({
                "changed_images": sum(source[i]["changed"] for i in image_ids),
                "generation_seconds_per_triggered_image": float(np.mean(
                    [source[i]["seconds"] for i in trigger_ids])) if trigger_ids else 0.0,
            })
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, LIMIT, size=(2000, LIMIT))
    base = np.asarray([per_image["greedy"][i]["metrics"]["CHAIRs"] for i in image_ids])
    base_hall = np.asarray([len(per_image["greedy"][i]["mscoco_hallucinated_words"])
                            for i in image_ids])
    base_mentions = np.asarray([len(per_image["greedy"][i]["object_mentions"])
                                for i in image_ids])
    for method, rows in per_image.items():
        if method == "greedy":
            continue
        chairs = np.asarray([rows[i]["metrics"]["CHAIRs"] for i in image_ids])
        hall = np.asarray([len(rows[i]["mscoco_hallucinated_words"]) for i in image_ids])
        mentions = np.asarray([len(rows[i]["object_mentions"]) for i in image_ids])
        delta_s = chairs[draws].mean(1) - base[draws].mean(1)
        delta_i = (hall[draws].sum(1) / np.maximum(1, mentions[draws].sum(1))
                   - base_hall[draws].sum(1) / np.maximum(1, base_mentions[draws].sum(1)))
        summary["methods"][method]["paired_delta_chair_s_ci95"] = (
            np.quantile(delta_s, [0.025, 0.975]).tolist())
        summary["methods"][method]["paired_delta_chair_i_ci95"] = (
            np.quantile(delta_i, [0.025, 0.975]).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chair_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.device)
    else:
        evaluate()


if __name__ == "__main__":
    main()
