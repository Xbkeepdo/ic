#!/usr/bin/env python3
"""Extract LLaVA K=32/tau=.03 SAR+EMS using final-only all-visual readout.

The original 8:2 compact cache supplies the exact image/target cohort and
ordering. Two GPU workers may process disjoint image shards. Checkpoints are
100-image NPZ files, so interrupted runs can resume without losing all work.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts import benchmark_k32_feature_complexity as original
from scripts import benchmark_k32_feature_optimization_branch as branch
from scripts.benchmark_k32_final_only_vocab import target_probabilities_final_only


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/llava_final_only_all_visual_4000"
OLD = ROOT / "outputs/log_s_hpre_k_tau_ablation/llava_1_5_7b/compact_features.npz"
MODEL = "llava_1_5_7b"
GROUP_SIZE = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-images", type=int, default=5)
    args = parser.parse_args()
    if args.num_shards < 1 or args.shard not in range(args.num_shards):
        parser.error("Invalid image shard")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(OLD, allow_pickle=False) as saved:
        old = {key: np.asarray(saved[key]) for key in (
            "image_id", "response_token_idx", "target_token_id",
            "log_s_hpre_k032_tau003", "f_log",
        )}
    image_ids = np.unique(old["image_id"])
    selected = image_ids[args.shard::args.num_shards]
    positions_by_image = {
        int(image_id): np.flatnonzero(old["image_id"] == image_id)
        for image_id in selected
    }
    source = original._source_dir(MODEL)
    labels = {int(k): v for k, v in original._json(source / "labeling.json").items()}
    generations = {int(k): v for k, v in original._json(source / "generations.json").items()}
    if not set(map(int, selected)).issubset(labels.keys() & generations.keys()):
        raise ValueError("Old feature images are missing from ENDAC generations")
    config = copy.deepcopy(original.load_config(str(original.OURS_CONFIG)))
    config["semantic_attention_topk"]["vocab_chunk_size"] = 4096
    baseline = original.load_config(str(original.BASELINE_CONFIG))
    wrapper = original.build_model(
        MODEL, original.get_extraction_model_cfg(baseline, MODEL), device=args.device,
    )
    wrapper.model.eval().requires_grad_(False)
    original_fn = branch.target_probabilities
    shard_dir = OUT / f"shard{args.shard}_of_{args.num_shards}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    check_rows = []
    started = time.monotonic()
    completed = 0
    try:
        for group_start in range(0, len(selected), GROUP_SIZE):
            group = selected[group_start:group_start + GROUP_SIZE]
            destination = shard_dir / f"part_{group_start // GROUP_SIZE:03d}.npz"
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    if not np.array_equal(saved["image_ids"], group):
                        raise ValueError(f"Checkpoint image mismatch: {destination}")
                    completed += len(group)
                    continue
            result_indices, result_risk, result_sar = [], [], []
            for image_id in group:
                image_id = int(image_id)
                rows = positions_by_image[image_id]
                common = {
                    "wrapper": wrapper, "model": MODEL, "config": config,
                    "image_id": image_id, "label": labels[image_id],
                    "generation": generations[image_id], "variant": "reference",
                }
                if len(check_rows) < args.check_images:
                    branch.target_probabilities = original_fn
                    reference = branch.extract_features(**common)
                else:
                    reference = None
                branch.target_probabilities = target_probabilities_final_only
                optimized = branch.extract_features(**common)
                expected = [
                    (int(old["response_token_idx"][i]), int(old["target_token_id"][i]))
                    for i in rows
                ]
                found = optimized["keys"]
                if len(set(found)) != len(found) or set(found) != set(expected):
                    raise ValueError(f"{image_id}: target set differs from old 8:2 cache")
                order = [found.index(key) for key in expected]
                optimized_risk = optimized["risk"][order]
                optimized_sar = optimized["sar"][order]
                if reference is not None:
                    if reference["keys"] != found:
                        raise ValueError(f"{image_id}: reference target set/order differs")
                    check_rows.append({
                        "image_id": image_id, "objects": len(rows),
                        "reference_vs_old": {
                            name: float(np.max(np.abs(reference[name][order] - old[key][rows])))
                            for name, key in (("risk", "log_s_hpre_k032_tau003"), ("sar", "f_log"))
                        },
                        "optimized_vs_old": {
                            name: float(np.max(np.abs(optimized[name][order] - old[key][rows])))
                            for name, key in (("risk", "log_s_hpre_k032_tau003"), ("sar", "f_log"))
                        },
                    })
                result_indices.extend(rows.tolist())
                result_risk.append(optimized_risk)
                result_sar.append(optimized_sar)
                completed += 1
                if completed % 50 == 0:
                    print(f"shard {args.shard}: {completed}/{len(selected)} images", flush=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez(
                temporary, image_ids=group,
                row_indices=np.asarray(result_indices, dtype=np.int64),
                risk=np.concatenate(result_risk), sar=np.concatenate(result_sar),
            )
            os.replace(temporary, destination)
            (shard_dir / "progress.json").write_text(json.dumps({
                "completed_images": completed,
                "total_images": len(selected),
                "elapsed_seconds": time.monotonic() - started,
                "reference_check": check_rows,
            }, indent=2) + "\n")
        print(f"shard {args.shard} complete: {completed}/{len(selected)} images", flush=True)
    finally:
        branch.target_probabilities = original_fn


if __name__ == "__main__":
    main()
