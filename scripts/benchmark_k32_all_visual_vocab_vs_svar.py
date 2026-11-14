#!/usr/bin/env python3
"""Time original K=32 SAR+EMS with all-visual vocab readout against SVAR.

The production extractor is called unchanged except for a per-run in-memory
semantic vocabulary chunk size. Each image runs all methods in rotating order.
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

from scripts import benchmark_k32_feature_complexity as original
from scripts.benchmark_k32_feature_optimization_branch import extract_features


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_k032_50_all_visual_vocab_vs_svar"
METHODS = ("ours_chunk32", "ours_all_visual", "svar")
ALL_CHUNK = 4096
GIB = 1024**3


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
    config32 = original.load_config(str(original.OURS_CONFIG))
    configall = copy.deepcopy(config32)
    configall["semantic_attention_topk"]["vocab_chunk_size"] = ALL_CHUNK
    baseline_full = original.load_config(str(original.BASELINE_CONFIG))
    baseline_cfg = original.baseline_config(baseline_full)
    wrapper = original.build_model(
        model, original.get_extraction_model_cfg(baseline_full, model), device=device_name,
    )
    wrapper.model.eval().requires_grad_(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = output_dir / "_runtime_scratch" / model / "svar"
    scratch.mkdir(parents=True, exist_ok=True)
    runtime = original.BaselineRuntime(
        wrapper=wrapper, methods=("svar",), baseline_dir=scratch,
        config=baseline_cfg, device=device_name, resume=False, inline_dhcp=True,
    )

    def run_one(method: str, image_id: int) -> dict[str, Any]:
        common = {
            "wrapper": wrapper, "model": model, "image_id": image_id,
            "label": labels[image_id], "generation": generations[image_id],
        }
        if method == "ours_chunk32":
            return original.extract_ours(config=config32, **common)
        if method == "ours_all_visual":
            return original.extract_ours(config=configall, **common)
        return original.extract_baseline(
            runtime=runtime, config=baseline_full, **common,
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
            counts = {entry["object_tokens"] for entry in per_method.values()}
            if len(counts) != 1:
                raise ValueError(f"{model}/{image_id}: object count differs by method")
            rows.append({"image_id": image_id, "methods": per_method})
            print(f"[{model}] {offset + 1}/{len(image_ids)}", flush=True)

        summaries = {}
        for method in METHODS:
            seconds = [row["methods"][method]["seconds"] for row in rows]
            summaries[method] = {
                "seconds_per_image": statistics.mean(seconds),
                "median_seconds_per_image": statistics.median(seconds),
                "total_seconds": sum(seconds),
                "peak_allocated_gib": max(row["methods"][method]["peak_allocated_gib"] for row in rows),
                "feature_dim": warmup[method]["feature_dim"],
                "object_tokens": sum(row["methods"][method]["object_tokens"] for row in rows),
            }

        numeric_rows = []
        for image_id in image_ids[:check_images]:
            common = {
                "wrapper": wrapper, "model": model, "image_id": image_id,
                "label": labels[image_id], "generation": generations[image_id],
                "variant": "reference",
            }
            reference = extract_features(config=config32, **common)
            all_visual = extract_features(config=configall, **common)
            if reference["keys"] != all_visual["keys"]:
                raise ValueError(f"{model}/{image_id}: feature target keys changed")
            numeric_rows.append({
                "image_id": image_id,
                "objects": len(reference["keys"]),
                "differences": {
                    name: np.abs(reference[name].astype(np.float64) - all_visual[name].astype(np.float64)).reshape(-1).tolist()
                    for name in ("risk", "sar")
                },
            })
        numeric_summary = {}
        for name in ("risk", "sar"):
            values = np.asarray([
                delta for row in numeric_rows for delta in row["differences"][name]
            ], dtype=np.float64)
            numeric_summary[name] = {
                "elements": int(values.size),
                "max_abs": float(values.max()),
                "mean_abs": float(values.mean()),
                "p99_abs": float(np.quantile(values, 0.99)),
            }
        payload = {
            "schema": "k32-all-visual-vocab-vs-svar-v1",
            "model": model, "device": device_name,
            "image_ids": image_ids, "warmup_image_id": warmup_id,
            "baseline_chunk": 32, "all_visual_chunk": ALL_CHUNK,
            "top_k": original.TOP_K, "source_tau": original.TAU,
            "timing_scope": "image loading + teacher-forced forward + complete feature extraction; excludes model load, caption generation, serialization, numeric check",
            "memory_scope": "PyTorch CUDA max_memory_allocated per method call, including resident model weights",
            "summary": summaries,
            "numeric_check": {
                "images": len(numeric_rows),
                "objects": sum(row["objects"] for row in numeric_rows),
                "summary": numeric_summary,
            },
            "per_image": rows,
        }
        destination = output_dir / f"{model}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {destination}", flush=True)
    finally:
        runtime.close()
        del runtime, wrapper
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
