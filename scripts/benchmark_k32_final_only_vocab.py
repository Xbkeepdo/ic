#!/usr/bin/env python3
"""Compare full K=32 SAR+EMS extraction with dual and final-only vocab readout.

This is an isolated benchmark. Both methods use all visual tokens in one
vocabulary projection batch; the final-only path skips raw-state logits.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts import benchmark_k32_feature_complexity as original
from scripts import benchmark_k32_feature_optimization_branch as branch
from models.dgst_capture import apply_decoder_final_norm


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_k032_50_final_only_vocab"
METHODS = ("raw_plus_final", "final_only")
ALL_CHUNK = 4096
GIB = 1024**3


@torch.no_grad()
def target_probabilities_final_only(
    visual_states: torch.Tensor,
    target_token_ids: list[int],
    output_layer: Any,
    final_norm: Any,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the original final-norm math without projecting the raw states.

    The first returned tensor is an alias because both benchmark callers
    discard it; the production probability function remains untouched.
    """
    weight = output_layer.weight
    targets = torch.tensor(
        [int(value) for value in target_token_ids],
        dtype=torch.long, device=weight.device,
    )
    norm_parts = []
    for start in range(0, len(visual_states), max(1, int(chunk_size))):
        raw = visual_states[start : start + int(chunk_size)].to(
            device=weight.device, dtype=weight.dtype,
        )
        normed = apply_decoder_final_norm(final_norm, raw)
        logits = F.linear(normed, weight, bias=None)
        log_partition = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        selected = logits.index_select(-1, targets).float()
        probability = torch.exp(selected - log_partition)
        norm_parts.append(probability.transpose(0, 1))
    norm_probability = torch.cat(norm_parts, dim=1)
    return norm_probability, norm_probability


def run_model(model: str, device_name: str, num_images: int, check_images: int, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    image_ids, warmup_id = original.benchmark_image_ids(num_images)
    source = original._source_dir(model)
    labels = {int(k): v for k, v in original._json(source / "labeling.json").items()}
    generations = {int(k): v for k, v in original._json(source / "generations.json").items()}
    config = copy.deepcopy(original.load_config(str(original.OURS_CONFIG)))
    config["semantic_attention_topk"]["vocab_chunk_size"] = ALL_CHUNK
    baseline = original.load_config(str(original.BASELINE_CONFIG))
    wrapper = original.build_model(
        model, original.get_extraction_model_cfg(baseline, model), device=device_name,
    )
    wrapper.model.eval().requires_grad_(False)
    production_function = original.target_probabilities
    branch_function = branch.target_probabilities

    def select_method(method: str) -> None:
        fn = production_function if method == "raw_plus_final" else target_probabilities_final_only
        original.target_probabilities = fn
        branch.target_probabilities = fn

    def run_one(method: str, image_id: int) -> dict[str, Any]:
        select_method(method)
        return original.extract_ours(
            wrapper=wrapper, model=model, config=config, image_id=image_id,
            label=labels[image_id], generation=generations[image_id],
        )

    try:
        warmup = {method: run_one(method, warmup_id) for method in METHODS}
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        rows = []
        for offset, image_id in enumerate(image_ids):
            order = METHODS[offset % len(METHODS):] + METHODS[:offset % len(METHODS)]
            per_method = {}
            for method in order:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                meta = run_one(method, image_id)
                torch.cuda.synchronize(device)
                per_method[method] = {
                    "seconds": time.perf_counter() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
                    "object_tokens": int(meta["object_count"]),
                    "feature_dim": int(meta["feature_dim"]),
                }
                if meta["feature_dim"] != warmup[method]["feature_dim"]:
                    raise ValueError(f"{model}/{image_id}/{method}: feature width changed")
            if per_method[METHODS[0]]["object_tokens"] != per_method[METHODS[1]]["object_tokens"]:
                raise ValueError(f"{model}/{image_id}: object count differs")
            rows.append({"image_id": image_id, "methods": per_method})
            print(f"[{model}] {offset + 1}/{len(image_ids)}", flush=True)

        summaries = {}
        for method in METHODS:
            values = [row["methods"][method] for row in rows]
            summaries[method] = {
                "seconds_per_image": statistics.mean(row["seconds"] for row in values),
                "median_seconds_per_image": statistics.median(row["seconds"] for row in values),
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in values),
                "object_tokens": sum(row["object_tokens"] for row in values),
                "feature_dim": warmup[method]["feature_dim"],
            }

        numeric_rows = []
        for image_id in image_ids[:check_images]:
            features = {}
            for method in METHODS:
                select_method(method)
                features[method] = branch.extract_features(
                    wrapper=wrapper, model=model, config=config, image_id=image_id,
                    label=labels[image_id], generation=generations[image_id],
                    variant="reference",
                )
            a, b = features[METHODS[0]], features[METHODS[1]]
            if a["keys"] != b["keys"]:
                raise ValueError(f"{model}/{image_id}: target alignment changed")
            numeric_rows.append({
                "image_id": image_id, "objects": len(a["keys"]),
                "differences": {
                    name: np.abs(a[name].astype(np.float64) - b[name].astype(np.float64)).reshape(-1).tolist()
                    for name in ("risk", "sar")
                },
            })
        numeric_summary = {}
        for name in ("risk", "sar"):
            values = np.asarray([
                delta for row in numeric_rows for delta in row["differences"][name]
            ], dtype=np.float64)
            numeric_summary[name] = {
                "elements": int(values.size), "max_abs": float(values.max()),
                "mean_abs": float(values.mean()),
                "p99_abs": float(np.quantile(values, 0.99)),
            }
        payload = {
            "schema": "k32-final-only-vocab-vs-dual-v1", "model": model,
            "device": device_name, "image_ids": image_ids,
            "warmup_image_id": warmup_id, "vocab_chunk_size": ALL_CHUNK,
            "top_k": original.TOP_K, "source_tau": original.TAU,
            "timing_scope": "image loading + teacher-forced forward + complete feature extraction; excludes model load, caption generation, serialization, numeric check",
            "summary": summaries,
            "numeric_check": {
                "images": len(numeric_rows),
                "objects": sum(row["objects"] for row in numeric_rows),
                "summary": numeric_summary,
            },
            "per_image": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{model}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {destination}", flush=True)
    finally:
        original.target_probabilities = production_function
        branch.target_probabilities = branch_function
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=original.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--check-images", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.check_images < 1 or args.check_images > args.num_images:
        parser.error("--check-images must be between 1 and --num-images")
    run_model(args.model, args.device, args.num_images, args.check_images, args.output_dir)


if __name__ == "__main__":
    main()
