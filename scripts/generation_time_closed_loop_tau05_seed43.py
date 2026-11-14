#!/usr/bin/env python3
"""Repeated detector-guided rollback on the same seed-43 LLaVA COCO pilot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model
from scripts.generation_time_sar_ems_seed43 import (
    CFG_DGST, OUT as BASE_OUT, PROMPT, PROBES, REQUIREMENTS, ROOT,
    RiskScorer, _chair_module, _decode, _generation_kwargs, _inputs,
    _mention_matcher, _mentions, _read_json,
)
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


THRESHOLD = 0.5
LIMIT = 100
MODEL = "llava_1_5_7b"
OUT = ROOT / "outputs/generation_time_closed_loop_tau05_seed43/llava_1_5_7b/test_100"
BASELINE = BASE_OUT / "llava_1_5_7b/test/greedy.jsonl"
CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"


class PrefixBans(LogitsProcessor):
    def __init__(self, prompt_length: int, bans: dict[tuple[int, ...], set[int]]):
        self.prompt_length = prompt_length
        self.bans = bans
        self.applied = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        prefix = tuple(int(x) for x in input_ids[0, self.prompt_length:].tolist())
        banned = self.bans.get(prefix)
        if banned:
            scores = scores.clone()
            scores[:, list(banned)] = -torch.inf
            if not torch.isfinite(scores).any():
                raise RuntimeError("All logits banned at a prefix")
            self.applied += 1
        return scores


@torch.inference_mode()
def score_all_mentions(wrapper, scorer, matcher, image: Image.Image, ids: list[int]):
    caption = _decode(wrapper, ids, MODEL)
    mentions = _mentions(matcher, wrapper.tokenizer, ids, caption)
    if mentions:
        features = wrapper.extract_token_features_batch(
            image=image, response_token_ids=ids,
            response_token_indices=[m["token_index"] for m in mentions],
            target_token_ids=[m["token_id"] for m in mentions],
            cfg_dgst_t=CFG_DGST, prompt=PROMPT, requirements=REQUIREMENTS,
        )
        scores = scorer.score_features(features)
        for mention, score in zip(mentions, scores):
            mention["hall_score"] = float(score)
    return caption, mentions


@torch.inference_mode()
def generate_image(wrapper, scorer, matcher, image: Image.Image, image_id: int,
                   baseline: dict, max_rollbacks: int) -> dict:
    ids = [int(x) for x in baseline["response_token_ids"]]
    bans: dict[tuple[int, ...], set[int]] = {}
    steps = []
    started = time.perf_counter()
    for _ in range(max_rollbacks + 1):
        caption, mentions = score_all_mentions(wrapper, scorer, matcher, image, ids)
        trigger = next((m for m in mentions if m["hall_score"] >= THRESHOLD), None)
        if trigger is None:
            reason = "no_detected_hallucination"
            break
        if len(steps) >= max_rollbacks:
            reason = "rollback_limit"
            break
        index = int(trigger["token_index"])
        prefix = tuple(ids[:index])
        forbidden = bans.setdefault(prefix, set())
        token = int(ids[index])
        if token in forbidden:
            raise AssertionError("A banned token was generated under the same prefix")
        forbidden.add(token)
        step = {"word": trigger["canonical"], "surface": trigger["surface"],
                "score": trigger["hall_score"], "token_index": index,
                "banned_first_id": token, "bans_at_prefix": len(forbidden)}
        inputs = _inputs(wrapper, image, MODEL)
        prompt_length = int(inputs["input_ids"].shape[1])
        processor = PrefixBans(prompt_length, bans)
        generated = wrapper.model.generate(
            **inputs, **_generation_kwargs(wrapper, MODEL),
            logits_processor=LogitsProcessorList([processor]),
        )
        new_ids = [int(x) for x in generated[0, prompt_length:].tolist()]
        if new_ids[:index] != ids[:index] or processor.applied < 1:
            raise AssertionError("Rollback did not preserve the prefix and apply a ban")
        if len(new_ids) <= index or new_ids[index] == token:
            raise AssertionError("Rejected first token was regenerated")
        step["replacement_first_id"] = new_ids[index]
        step["discarded_tokens"] = len(ids) - index
        steps.append(step)
        ids = new_ids
    return {"image_id": image_id, "generated_text": caption,
            "response_token_ids": ids, "final_mentions": mentions,
            "rollbacks": len(steps), "actions": steps,
            "stopping_reason": reason, "seconds": time.perf_counter() - started}


def _load_results() -> dict[int, dict]:
    rows = {}
    legacy = OUT / "closed_loop.jsonl"
    if legacy.exists():
        rows.update((int(row["image_id"]), row) for row in
                    (json.loads(line) for line in legacy.read_text().splitlines() if line.strip()))
    for path in (OUT / "rows").glob("*.json"):
        row = json.loads(path.read_text())
        rows[int(row["image_id"])] = row
    return rows


def generate(device: str, image_id: int | None, max_rollbacks: int,
             start_index: int, end_index: int) -> None:
    torch.set_num_threads(1)
    config = load_config(str(CONFIG))
    image_ids = sorted(_read_json(PROBES / "splits/seed43.json")["test"])[:LIMIT]
    image_ids = image_ids[start_index:end_index]
    if image_id is not None:
        if image_id not in image_ids:
            raise ValueError(f"Image {image_id} is outside the fixed 100-image sample")
        image_ids = [image_id]
    baseline = {int(row["image_id"]): row for row in
                (json.loads(line) for line in BASELINE.read_text().splitlines() if line.strip())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows").mkdir(exist_ok=True)
    done = set(_load_results())
    if all(i in done for i in image_ids):
        print("All requested images already generated", flush=True)
        return
    scorer = RiskScorer(MODEL)
    matcher = _mention_matcher()
    wrapper = build_model(MODEL, get_extraction_model_cfg(config, MODEL), device=device)
    wrapper.model.eval().requires_grad_(False)
    coco = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
    for ordinal, iid in enumerate(image_ids, 1):
        if iid in done:
            continue
        with Image.open(coco / f"COCO_val2014_{iid:012d}.jpg") as opened:
            image = opened.convert("RGB")
        row = generate_image(wrapper, scorer, matcher, image, iid, baseline[iid], max_rollbacks)
        path = OUT / "rows" / f"{iid}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        print(f"[{ordinal}/{len(image_ids)}] image={iid} rollbacks={row['rollbacks']} "
              f"reason={row['stopping_reason']} seconds={row['seconds']:.1f}", flush=True)


def evaluate() -> None:
    config = load_config(str(CONFIG))
    image_ids = sorted(_read_json(PROBES / "splits/seed43.json")["test"])[:LIMIT]
    baseline = {int(row["image_id"]): row for row in
                (json.loads(line) for line in BASELINE.read_text().splitlines() if line.strip())}
    rows = _load_results()
    if set(image_ids) - rows.keys():
        raise ValueError("Closed-loop generation has missing images")
    evaluator = _chair_module().CocoChairEvaluator.from_cache(
        instances_file=get_dataset_cfg(config)["annotation_file"],
        captions_file=get_dataset_cfg(config).get("captions_file"),
        cache_path=ROOT / "outputs/generation_time_paper_transfer_seed43/chair_cache.pkl",
    )
    original = [evaluator.compute_chair_token(i, baseline[i]["generated_text"]) for i in image_ids]
    revised = [evaluator.compute_chair_token(i, rows[i]["generated_text"]) for i in image_ids]
    def metrics(items):
        return {"CHAIRs": np.mean([r["metrics"]["CHAIRs"] for r in items]).item(),
                "CHAIRi": sum(len(r["mscoco_hallucinated_words"]) for r in items)
                          / sum(len(r["object_mentions"]) for r in items),
                "object_recall": np.mean([r["metrics"]["Recall"] for r in items]).item(),
                "hallucinated_mentions": sum(len(r["mscoco_hallucinated_words"]) for r in items),
                "all_mentions": sum(len(r["object_mentions"]) for r in items),
                "words_per_image": np.mean([len(r["words"]) for r in items]).item()}
    base_s = np.array([r["metrics"]["CHAIRs"] for r in original])
    new_s = np.array([r["metrics"]["CHAIRs"] for r in revised])
    base_h = np.array([len(r["mscoco_hallucinated_words"]) for r in original])
    new_h = np.array([len(r["mscoco_hallucinated_words"]) for r in revised])
    base_m = np.array([len(r["object_mentions"]) for r in original])
    new_m = np.array([len(r["object_mentions"]) for r in revised])
    base_r = np.array([r["metrics"]["Recall"] for r in original])
    new_r = np.array([r["metrics"]["Recall"] for r in revised])
    draws = np.random.default_rng(43).integers(0, LIMIT, size=(5000, LIMIT))
    delta_s = (new_s - base_s)[draws].mean(axis=1)
    delta_i = (new_h[draws].sum(axis=1) / np.maximum(1, new_m[draws].sum(axis=1))
               - base_h[draws].sum(axis=1) / np.maximum(1, base_m[draws].sum(axis=1)))
    delta_r = (new_r - base_r)[draws].mean(axis=1)
    action_labels = []
    for i, reference in zip(image_ids, original):
        gt = set(reference["mscoco_gt_words"])
        action_labels.extend(a["word"] not in gt for a in rows[i]["actions"])
    result = {"seed": 43, "images": LIMIT, "threshold": THRESHOLD,
              "baseline": metrics(original), "closed_loop": metrics(revised),
              "delta_chairs_ci95": np.quantile(delta_s, [.025, .975]).tolist(),
              "delta_chairi_ci95": np.quantile(delta_i, [.025, .975]).tolist(),
              "delta_object_recall_ci95": np.quantile(delta_r, [.025, .975]).tolist(),
              "fixed_images": int(((base_s == 1) & (new_s == 0)).sum()),
              "introduced_images": int(((base_s == 0) & (new_s == 1)).sum()),
              "triggered_images": sum(rows[i]["rollbacks"] > 0 for i in image_ids),
              "total_rollbacks": sum(rows[i]["rollbacks"] for i in image_ids),
              "actions_on_chair_hallucinated_objects": int(sum(action_labels)),
              "actions_on_chair_grounded_objects": int(len(action_labels)-sum(action_labels)),
              "max_rollbacks": max(rows[i]["rollbacks"] for i in image_ids),
              "unresolved_images": [i for i in image_ids if rows[i]["stopping_reason"] != "no_detected_hallucination"],
              "detector_clear_but_chair_hallucinated_images": int(new_s.sum()),
              "changed_images": sum(rows[i]["generated_text"] != baseline[i]["generated_text"] for i in image_ids),
              "generation_seconds": sum(rows[i]["seconds"] for i in image_ids)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    global_threshold = THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--max-rollbacks", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=LIMIT)
    parser.add_argument("--threshold", type=float, default=global_threshold)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be strictly between 0 and 1")
    THRESHOLD = args.threshold
    if THRESHOLD != global_threshold:
        tag = f"tau{THRESHOLD:.7f}".replace(".", "p")
        OUT = ROOT / f"outputs/generation_time_closed_loop_{tag}_seed43/llava_1_5_7b/test_100"
    if args.command == "generate":
        generate(args.device, args.image_id, args.max_rollbacks,
                 args.start_index, args.end_index)
    else:
        evaluate()
