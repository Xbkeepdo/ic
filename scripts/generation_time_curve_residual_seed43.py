#!/usr/bin/env python3
"""Seed-43, 200-image COCO pilot of fixed curve-derived residual scaling.

Coefficients use only training-image SAR/EMS class curves.  Test labels are
read only by the independent evaluate command.
"""

from __future__ import annotations

import argparse
import fcntl
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
from scripts.generation_time_sar_ems_seed43 import _chair_module, _decode
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/log_s_mechanism_ablation_811"
SPLITS = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45/splits/seed43.json"
BASELINE = ROOT / "outputs/generation_time_sar_ems_seed43_v2"
OUT = ROOT / "outputs/generation_time_curve_residual_seed43"
CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
SEED = 43
PROMPT = "Describe this image."
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
METHODS = ("attention", "ffn", "joint")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> dict[int, dict]:
    return {
        int(row["image_id"]): row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def curves_and_coefficients(model_name: str) -> dict:
    """Training-only class means and bounded, parameter-free per-layer weights."""
    train_ids = set(map(int, _json(SPLITS)["train"]))
    with np.load(SOURCE / model_name / "compact_features.npz", allow_pickle=False) as data:
        train = np.isin(data["image_id"], list(train_ids))
        labels = data["label"][train]
        if not (np.any(labels == 0) and np.any(labels == 1)):
            raise ValueError("Both training classes are required")
        n_layers = int(data["sar"].shape[1])
        if data["ems"].shape[1] != n_layers:
            raise ValueError("SAR and EMS layer counts differ")
        # Middle half of LM layers, indexed from zero.
        mid = list(range(n_layers // 4, 3 * n_layers // 4))
        result = {"model": model_name, "seed": SEED, "n_layers": n_layers,
                  "middle_layers": mid, "train_object_tokens": int(train.sum()),
                  "train_hall_tokens": int(np.sum(labels == 0)),
                  "train_real_tokens": int(np.sum(labels == 1)),
                  "source": str(SOURCE / model_name / "compact_features.npz")}
        for name in ("sar", "ems"):
            values = np.asarray(data[name][train], dtype=np.float64)
            hall = values[labels == 0].mean(axis=0)
            real = values[labels == 1].mean(axis=0)
            distance = np.abs(hall - real)
            selected = distance[mid]
            denominator = float(selected.sum() + selected.max() + 1e-12)
            weight = np.zeros(n_layers, dtype=np.float64)
            weight[mid] = selected / denominator
            result[name] = {"hall_mean": hall.tolist(), "real_mean": real.tolist(),
                            "distance": distance.tolist(), "weight": weight.tolist()}
    result["attention_scale"] = (1.0 + np.asarray(result["sar"]["weight"])).tolist()
    result["ffn_scale"] = (1.0 - np.asarray(result["ems"]["weight"])).tolist()
    return result


def _scale_prediction_row(value: torch.Tensor, factor: float) -> torch.Tensor:
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"Unexpected branch output shape: {tuple(value.shape)}")
    if value.shape[1] == 1:
        return value * factor
    # In prefill, only the last prompt position predicts the first answer token.
    result = value.clone()
    result[:, -1, :] *= factor
    return result


@contextmanager
def scaled_residuals(model, attention_scale: list[float], ffn_scale: list[float]):
    layers = resolve_decoder_layers(model)
    if len(layers) != len(attention_scale) or len(layers) != len(ffn_scale):
        raise ValueError("Coefficient count differs from decoder layer count")
    handles = []
    calls = {"attention": 0, "ffn": 0}
    try:
        for index, layer in enumerate(layers):
            a = float(attention_scale[index])
            f = float(ffn_scale[index])
            if a != 1.0:
                def attention_hook(_module, _args, output, factor=a):
                    calls["attention"] += 1
                    if isinstance(output, tuple):
                        return (_scale_prediction_row(output[0], factor), *output[1:])
                    return _scale_prediction_row(output, factor)
                handles.append(_layer_attention_module(layer).register_forward_hook(attention_hook))
            if f != 1.0:
                def ffn_hook(_module, _args, output, factor=f):
                    calls["ffn"] += 1
                    return _scale_prediction_row(output, factor)
                handles.append(_layer_mlp_module(layer).register_forward_hook(ffn_hook))
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def generate(model_name: str, limit: int, methods: list[str], device: str,
             start_offset: int = 0, end_offset: int | None = None) -> None:
    torch.set_num_threads(1)
    coefficients = curves_and_coefficients(model_name)
    out = OUT / model_name / f"test_{limit}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "coefficients.json").write_text(json.dumps(coefficients, indent=2))
    all_image_ids = sorted(map(int, _json(SPLITS)["test"]))[:limit]
    if start_offset < 0 or (end_offset is not None and end_offset < start_offset):
        raise ValueError("Invalid image offset range")
    image_ids = all_image_ids[start_offset:end_offset]
    if not image_ids:
        return
    baseline = _rows(BASELINE / model_name / "test/greedy.jsonl")
    if not all(image_id in baseline for image_id in image_ids):
        raise ValueError("Paired greedy baseline is incomplete")
    config = load_config(str(CONFIG))
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    done = {method: _rows(out / f"{method}.jsonl") if (out / f"{method}.jsonl").exists() else {}
            for method in methods}
    wrapper = build_model(model_name, get_extraction_model_cfg(config, model_name), device=device)
    wrapper.model.eval().requires_grad_(False)
    first_id = image_ids[0]
    with Image.open(coco / f"COCO_val2014_{first_id:012d}.jpg") as opened:
        check_image = opened.convert("RGB")
    check = wrapper.generate(check_image, prompt=PROMPT)
    if list(map(int, check.response_token_ids)) != baseline[first_id]["response_token_ids"]:
        raise RuntimeError("Current greedy generation differs from cached paired baseline")
    base_a = coefficients["attention_scale"]
    base_f = coefficients["ffn_scale"]
    ones = [1.0] * coefficients["n_layers"]
    for ordinal, image_id in enumerate(image_ids, 1):
        todo = [method for method in methods if image_id not in done[method]]
        if not todo:
            continue
        with Image.open(coco / f"COCO_val2014_{image_id:012d}.jpg") as opened:
            image = opened.convert("RGB")
        for method in todo:
            a = base_a if method in ("attention", "joint") else ones
            f = base_f if method in ("ffn", "joint") else ones
            start = time.perf_counter()
            with scaled_residuals(wrapper.model, a, f) as calls:
                generated = wrapper.generate(image, prompt=PROMPT)
            elapsed = time.perf_counter() - start
            if (method in ("attention", "joint") and calls["attention"] == 0 or
                    method in ("ffn", "joint") and calls["ffn"] == 0):
                raise RuntimeError(f"{method}: residual hooks did not fire")
            ids = list(map(int, generated.response_token_ids))
            text = _decode(wrapper, ids, model_name)
            row = {"image_id": image_id, "method": method, "seed": SEED,
                   "generated_text": text, "response_token_ids": ids,
                   "seconds": elapsed, "hook_calls": calls,
                   "changed": ids != baseline[image_id]["response_token_ids"]}
            with (out / f"{method}.jsonl").open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            print(f"[{model_name} {ordinal}/{limit}] {method}: "
                  f"changed={row['changed']} seconds={elapsed:.2f}", flush=True)


def evaluate(model_name: str, limit: int) -> None:
    image_ids = sorted(map(int, _json(SPLITS)["test"]))[:limit]
    out = OUT / model_name / f"test_{limit}"
    baseline = _rows(BASELINE / model_name / "test/greedy.jsonl")
    variants = {"greedy": {image_id: baseline[image_id] for image_id in image_ids}}
    for method in METHODS:
        path = out / f"{method}.jsonl"
        if path.exists():
            source = _rows(path)
            if any(image_id not in source for image_id in image_ids):
                raise ValueError(f"{method}: paired rows missing")
            variants[method] = {image_id: source[image_id] for image_id in image_ids}
    config = load_config(str(CONFIG))
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=BASELINE / "chair_cache.pkl",
    )
    per_image = {}
    summary = {"seed": SEED, "model": model_name, "images": limit, "methods": {}}
    for method, source in variants.items():
        rows = [evaluator.compute_chair_token(image_id, source[image_id]["generated_text"])
                for image_id in image_ids]
        per_image[method] = rows
        total = sum(len(row["object_mentions"]) for row in rows)
        hall = sum(len(row["mscoco_hallucinated_words"]) for row in rows)
        summary["methods"][method] = {
            "chair_s": float(np.mean([row["metrics"]["CHAIRs"] for row in rows])),
            "chair_i": hall / total if total else math.nan,
            "object_recall": float(np.mean([row["metrics"]["Recall"] for row in rows])),
            "true_mentions_per_image": float(np.mean([len(row["mscoco_real_words"]) for row in rows])),
            "object_mentions_per_image": total / limit,
            "no_object_images": sum(not row["object_mentions"] for row in rows),
            "words_per_image": float(np.mean([len(row["words"]) for row in rows])),
        }
        if method != "greedy":
            summary["methods"][method].update({
                "changed_images": sum(source[i]["changed"] for i in image_ids),
                "generation_seconds_per_image": float(np.mean([source[i]["seconds"] for i in image_ids])),
            })
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, limit, size=(2000, limit))
    base = np.asarray([row["metrics"]["CHAIRs"] for row in per_image["greedy"]])
    base_hall = np.asarray([len(row["mscoco_hallucinated_words"]) for row in per_image["greedy"]])
    base_mentions = np.asarray([len(row["object_mentions"]) for row in per_image["greedy"]])
    base_recall = np.asarray([row["metrics"]["Recall"] for row in per_image["greedy"]])
    for method, rows in per_image.items():
        if method == "greedy":
            continue
        values = np.asarray([row["metrics"]["CHAIRs"] for row in rows])
        delta = values[draws].mean(1) - base[draws].mean(1)
        summary["methods"][method]["paired_delta_chair_s_ci95"] = (
            np.quantile(delta, [0.025, 0.975]).tolist()
        )
        hall = np.asarray([len(row["mscoco_hallucinated_words"]) for row in rows])
        mentions = np.asarray([len(row["object_mentions"]) for row in rows])
        delta_i = (hall[draws].sum(1) / np.maximum(1, mentions[draws].sum(1))
                   - base_hall[draws].sum(1) / np.maximum(1, base_mentions[draws].sum(1)))
        summary["methods"][method]["paired_delta_chair_i_ci95"] = (
            np.quantile(delta_i, [0.025, 0.975]).tolist()
        )
        recall = np.asarray([row["metrics"]["Recall"] for row in rows])
        delta_recall = recall[draws].mean(1) - base_recall[draws].mean(1)
        summary["methods"][method]["paired_delta_recall_ci95"] = (
            np.quantile(delta_recall, [0.025, 0.975]).tolist()
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "chair_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--end-offset", type=int)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 400:
        raise ValueError("--limit must be in 1..400")
    if args.command == "generate":
        generate(args.model, args.limit, args.methods, args.device,
                 args.start_offset, args.end_offset)
    else:
        evaluate(args.model, args.limit)


if __name__ == "__main__":
    main()
