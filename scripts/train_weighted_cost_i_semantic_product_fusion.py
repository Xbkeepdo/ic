#!/usr/bin/env python3
"""Fuse weighted-cost risks with layerwise I × semantic ENDAC features."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import MODELS, _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from scripts.train_weighted_cost_attention_i_fusion import (  # noqa: E402
    RUNS,
    SEEDS,
    _baseline_path,
    _load_model,
    _risk_specs,
)
from utils.config_utils import load_config  # noqa: E402


OUT = ROOT / "outputs/weighted_cost_i_semantic_product_fusion"
PRODUCTS = {
    "I_x_raw_semantic_only": "raw_semantic_only",
    "I_x_norm_semantic_only": "norm_semantic_only",
    "I_x_raw_semantic_attention": "raw_semantic_attention",
    "I_x_norm_semantic_attention": "norm_semantic_attention",
}
EXPECTED_CASES = 36 * len(PRODUCTS) * len(SEEDS)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2) + "\n")
    os.replace(temporary, path)


def report(metrics: dict, threshold: str) -> dict:
    return metrics["threshold_reports"][threshold]["test_metrics"]


def summarize(model: str, model_out: Path, audit: dict) -> None:
    rows = []
    for spec in _risk_specs():
        baselines = [json.loads(_baseline_path(model, spec, seed).read_text()) for seed in SEEDS]
        for product in PRODUCTS:
            cases = [
                json.loads((model_out / spec["name"] / product / f"seed{seed}" / "metrics.json").read_text())
                for seed in SEEDS
            ]
            for threshold in ("fixed_0.5", "train_f1"):
                fused_reports = [report(case, threshold) for case in cases]
                base_reports = [report(case, threshold) for case in baselines]
                row = {
                    "model": model,
                    "threshold": threshold,
                    "run": spec["run"],
                    "source": spec["source"],
                    "method": spec["method"],
                    "weight": spec["weight"],
                    "risk": spec["name"],
                    "added_feature": product,
                }
                for metric_name, (side, key) in {
                    "real_f1": ("real_positive", "f1"),
                    "real_auc": ("real_positive", "auc"),
                    "hall_aupr": ("hallucination_positive", "aupr"),
                }.items():
                    fused = np.asarray([value[side][key] for value in fused_reports], dtype=np.float64)
                    baseline = np.asarray([value[side][key] for value in base_reports], dtype=np.float64)
                    row[f"{metric_name}_mean"] = float(fused.mean())
                    row[f"{metric_name}_std"] = float(fused.std())
                    row[f"baseline_{metric_name}_mean"] = float(baseline.mean())
                    row[f"delta_{metric_name}_mean"] = float((fused - baseline).mean())
                rows.append(row)

    with (model_out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: weighted-cost risk + I × semantic feature",
        "",
        f"Exact aligned mentions: {audit['matched_rows']} "
        f"(train {audit['matched_train_rows']}, test {audit['matched_test_rows']}).",
        "Input is [risk[L], I[L] × semantic_feature[L]]. Products are computed before "
        "training-only column z-score. Probe: one hidden layer width 128, ReLU, "
        "dropout 0.3, no BatchNorm, seeds 43–45.",
        "",
        "| Threshold | Run | Source | Method | w | Added feature | Real F1 | Real AUC | Δ AUC | Hall AUPR | Δ Hall AUPR |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['run']} | {row['source']} | {row['method']} | "
            f"{row['weight']} | {row['added_feature']} | "
            f"{row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | "
            f"{row['delta_real_auc_mean']:+.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{row['delta_hall_aupr_mean']:+.4f} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")


def train(model: str, device: str, output_root: Path, dry_run: bool) -> None:
    torch.set_num_threads(1)
    data, audit = _load_model(model)
    products = {
        name: (data["extras"]["I"] * data["extras"][semantic]).astype(np.float32)
        for name, semantic in PRODUCTS.items()
    }
    for name, matrix in products.items():
        if matrix.shape != data["extras"]["I"].shape or not np.isfinite(matrix).all():
            raise ValueError(f"{model}: invalid product {name}")
    audit = {
        **audit,
        "schema": "weighted-cost-i-semantic-product-fusion-v1",
        "product_features": PRODUCTS,
        "fusion_count": len(data["risks"]) * len(products),
        "training_case_count": len(data["risks"]) * len(products) * len(SEEDS),
    }
    if audit["training_case_count"] != EXPECTED_CASES:
        raise AssertionError(f"{model}: unexpected case count")
    print(json.dumps(audit, indent=2), flush=True)
    if dry_run:
        return

    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    atomic_json(model_out / "protocol.json", {
        **audit,
        "architecture": "one hidden layer, width 128",
        "standardize_input": True,
        "batch_norm": False,
        "dropout": 0.3,
        "seeds": list(SEEDS),
    })
    completed = sum(1 for _ in model_out.glob("*/*/seed*/metrics.json"))
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "training", "completed": completed, "total": EXPECTED_CASES,
    })
    config_root = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    labels = data["label"]
    for spec in _risk_specs():
        risk = data["risks"][spec["name"]]
        for product_name, product in products.items():
            features = np.concatenate((risk, product), axis=1).astype(np.float32, copy=False)
            for seed in SEEDS:
                case = model_out / spec["name"] / product_name / f"seed{seed}"
                metrics_path = case / "metrics.json"
                if metrics_path.exists():
                    print("SKIP", model, spec["name"], product_name, seed, flush=True)
                    continue
                config = replace(
                    _probe_config(config_root, seed),
                    hidden_sizes=(128,), batch_norm=False, standardize_input=True,
                )
                metrics = train_and_evaluate_probe(
                    X_train=features[data["train_mask"]],
                    y_train=labels[data["train_mask"]],
                    X_val=np.empty((0, features.shape[1]), dtype=np.float32),
                    y_val=np.empty((0,), dtype=np.int64),
                    X_test=features[data["test_mask"]],
                    y_test=labels[data["test_mask"]],
                    config=config,
                    device=torch.device(device),
                    output_dir=str(case),
                )
                metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
                completed += 1
                atomic_json(model_out / "progress.json", {
                    "model": model, "stage": "training", "completed": completed,
                    "total": EXPECTED_CASES, "risk": spec["name"],
                    "added_feature": product_name, "seed": seed,
                })
                print("DONE", model, spec["name"], product_name, seed, flush=True)
    summarize(model, model_out, audit)
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "complete", "completed": EXPECTED_CASES,
        "total": EXPECTED_CASES,
    })
    print("SUMMARY", model_out / "summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.model, args.device, args.output_root.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
