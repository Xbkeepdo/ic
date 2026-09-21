#!/usr/bin/env python3
"""Train matched COCO4000-512 risk probes for the weighted-cost/source grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import _probe_config
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe
from utils.config_utils import load_config
from utils.io_utils import load_pkl

WEIGHTS = (2, 4, 6, 8)
SEEDS = (43, 44, 45)


def train(run_dir: Path, device: str) -> None:
    torch.set_num_threads(1)
    rows = load_pkl(str(run_dir / "features.pkl"))
    if not rows:
        raise ValueError("No extracted features")
    if len({(int(r["image_id"]), int(r["response_token_idx"]), int(r["target_token_id"])) for r in rows}) != len(rows):
        raise ValueError("Duplicate target key")
    splits = json.loads((run_dir / "image_splits.json").read_text())
    train_ids = {int(x) for x in splits["train"]}
    test_ids = {int(x) for x in splits["test"]}
    if train_ids & test_ids or len(train_ids) != 3200 or len(test_ids) != 800:
        raise ValueError("Expected strict 3200/800 image split")
    images = np.array([int(r["image_id"]) for r in rows])
    train_mask = np.isin(images, list(train_ids))
    test_mask = np.isin(images, list(test_ids))
    if not np.all(train_mask | test_mask):
        raise ValueError("Extracted target outside strict image split")
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    out = run_dir / "weighted_cost_probe"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    source_modes = tuple(rows[0].get("dgst_t_source_modes") or [rows[0]["dgst_t_source_distribution_mode"]])
    methods = tuple(rows[0]["dgst_t_four_gate_methods"])
    for source_mode in source_modes:
      for method in methods:
        for digit in WEIGHTS:
            name = f"{method}_w0{digit}"
            state = "hmid" if method.startswith("hmid_") else "hpre"
            key = f"dgst_t_{method}_risk_linear_matched_state_target_w0{digit}_{state}"
            key += f"_source_{source_mode}_per_layer" if len(source_modes) > 1 else "_per_layer"
            X = np.stack([np.asarray(r[key], dtype=np.float32) for r in rows])
            if not np.isfinite(X).all():
                raise ValueError(f"Nonfinite risk: {name}")
            cases = []
            for seed in SEEDS:
                case_dir = out / source_mode / name / f"seed{seed}"
                metrics_path = case_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text())
                else:
                    cfg = replace(
                        _probe_config(config, seed),
                        hidden_sizes=(128,), batch_norm=False,
                        standardize_input=True,
                    )
                    metrics = train_and_evaluate_probe(
                        X_train=X[train_mask], y_train=labels[train_mask],
                        X_val=np.empty((0, X.shape[1]), dtype=np.float32),
                        y_val=np.empty((0,), dtype=np.int32),
                        X_test=X[test_mask], y_test=labels[test_mask],
                        config=cfg, device=torch.device(device),
                        output_dir=str(case_dir),
                    )
                    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
                cases.append(metrics)
                print("DONE", name, seed, flush=True)
            for threshold in ("fixed_0.5", "train_f1"):
                reports = [case["threshold_reports"][threshold]["test_metrics"] for case in cases]
                summary.append({
                    "source": source_mode, "risk": name, "threshold": threshold,
                    "test_hall_aupr_mean": float(np.mean([r["hallucination_positive"]["aupr"] for r in reports])),
                    "test_hall_aupr_std": float(np.std([r["hallucination_positive"]["aupr"] for r in reports])),
                    "test_real_f1_mean": float(np.mean([r["real_positive"]["f1"] for r in reports])),
                    "test_real_f1_std": float(np.std([r["real_positive"]["f1"] for r in reports])),
                })
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "summary.md").write_text(
        "# Weighted cost and attention-write source comparison\n\n"
        "Same COCO4000-512 generations; strict image split 3200/800. "
        "Each risk uses three seeds (43–45), one hidden layer of width 128, "
        "training-only input z-score, and no BatchNorm.\n\n"
        "| Source | Risk | Threshold | Hall AUPR mean±SD | Real F1 mean±SD |\n"
        "|---|---|---|---:|---:|\n"
        + "\n".join(
            f"| {r['source']} | {r['risk']} | {r['threshold']} | {r['test_hall_aupr_mean']:.4f}±{r['test_hall_aupr_std']:.4f} | "
            f"{r['test_real_f1_mean']:.4f}±{r['test_real_f1_std']:.4f} |"
            for r in summary
        ) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train(args.run_dir.resolve(), args.device)
