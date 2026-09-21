#!/usr/bin/env python3
"""Train fixed weighted-cost risk with raw semantic and I combinations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
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
    SEEDS,
    _baseline_path,
    _load_model,
    _risk_specs,
)
from utils.config_utils import load_config  # noqa: E402


OUT = ROOT / "outputs/fixed_wcost_raw_semantic_i_combinations"
RISK_NAME = "wcost_all__softmax__hpre_raw_logit_gauss__w02"
FEATURE_GROUPS = {
    "risk_raw_semantic_attention_I": ("risk", "raw_semantic_attention", "I"),
    "risk_raw_semantic_only_I": ("risk", "raw_semantic_only", "I"),
    "raw_semantic_attention": ("raw_semantic_attention",),
    "raw_semantic_only": ("raw_semantic_only",),
    "I": ("I",),
    "raw_semantic_attention_I": ("raw_semantic_attention", "I"),
    "raw_semantic_only_I": ("raw_semantic_only", "I"),
}
EXPECTED_CASES = len(FEATURE_GROUPS) * len(SEEDS)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2) + "\n")
    os.replace(temporary, path)


def report(metrics: dict, threshold: str) -> dict:
    return metrics["threshold_reports"][threshold]["test_metrics"]


def fixed_risk_spec() -> dict:
    matches = [spec for spec in _risk_specs() if spec["name"] == RISK_NAME]
    if len(matches) != 1:
        raise AssertionError(f"Expected one fixed risk spec, found {len(matches)}")
    return matches[0]


def summarize(model: str, model_out: Path, audit: dict) -> None:
    spec = fixed_risk_spec()
    metric_sources = {
        "risk": [_baseline_path(model, spec, seed) for seed in SEEDS],
        **{
            name: [model_out / name / f"seed{seed}" / "metrics.json" for seed in SEEDS]
            for name in FEATURE_GROUPS
        },
    }
    rows = []
    for feature_group, paths in metric_sources.items():
        cases = [json.loads(path.read_text()) for path in paths]
        for threshold in ("fixed_0.5", "train_f1"):
            reports = [report(case, threshold) for case in cases]
            row = {
                "model": model,
                "threshold": threshold,
                "feature_group": feature_group,
                "run": spec["run"],
                "source": spec["source"],
                "method": spec["method"],
                "weight": spec["weight"],
            }
            for metric_name, (side, key) in {
                "real_f1": ("real_positive", "f1"),
                "real_auc": ("real_positive", "auc"),
                "hall_aupr": ("hallucination_positive", "aupr"),
            }.items():
                values = np.asarray([value[side][key] for value in reports], dtype=np.float64)
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(values.std())
            rows.append(row)

    risk_by_threshold = {
        threshold: next(
            row for row in rows
            if row["feature_group"] == "risk" and row["threshold"] == threshold
        )
        for threshold in ("fixed_0.5", "train_f1")
    }
    for row in rows:
        baseline = risk_by_threshold[row["threshold"]]
        row["delta_real_auc_vs_risk"] = row["real_auc_mean"] - baseline["real_auc_mean"]
        row["delta_hall_aupr_vs_risk"] = row["hall_aupr_mean"] - baseline["hall_aupr_mean"]

    with (model_out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {model}: fixed risk with raw semantic / I combinations",
        "",
        f"Aligned mentions: {audit['matched_rows']} "
        f"(train {audit['matched_train_rows']}, test {audit['matched_test_rows']}).",
        "Fixed risk: WCOST-ALL / softmax / hpre_raw_logit_gauss / w=0.2. "
        "The risk-only row reuses its completed baseline metrics.",
        "All newly trained groups use one hidden layer of width 128, ReLU, dropout 0.3, "
        "training-only column z-score, no BatchNorm, seeds 43–45.",
        "",
        "| Threshold | Feature group | Real F1 | Real AUC | Δ AUC vs risk | Hall AUPR | Δ Hall AUPR vs risk |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['feature_group']} | "
            f"{row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | "
            f"{row['delta_real_auc_vs_risk']:+.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{row['delta_hall_aupr_vs_risk']:+.4f} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")


def train(model: str, device: str, output_root: Path, dry_run: bool) -> None:
    torch.set_num_threads(1)
    data, source_audit = _load_model(model)
    spec = fixed_risk_spec()
    risk = data["risks"][RISK_NAME]
    matrices = {}
    for name, parts in FEATURE_GROUPS.items():
        components = [risk if part == "risk" else data["extras"][part] for part in parts]
        matrix = np.concatenate(components, axis=1) if len(components) > 1 else components[0]
        matrix = matrix.astype(np.float32, copy=False)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{model}: nonfinite feature group {name}")
        matrices[name] = matrix

    audit = {
        **source_audit,
        "schema": "fixed-wcost-raw-semantic-i-combinations-v1",
        "fixed_risk": spec,
        "feature_groups": FEATURE_GROUPS,
        "trained_group_count": len(FEATURE_GROUPS),
        "training_case_count": EXPECTED_CASES,
        "input_dimensions": {name: int(value.shape[1]) for name, value in matrices.items()},
    }
    print(json.dumps(audit, indent=2), flush=True)
    if dry_run:
        return

    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    atomic_json(model_out / "alignment.json", audit)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    example = replace(
        _probe_config(root_config, SEEDS[0]),
        hidden_sizes=(128,), batch_norm=False, standardize_input=True,
    )
    atomic_json(model_out / "protocol.json", {
        **audit,
        "probe": _json_ready(asdict(example)),
        "seeds": list(SEEDS),
        "risk_only": "reused existing weighted_cost_probe metrics",
    })
    completed = sum(1 for _ in model_out.glob("*/seed*/metrics.json"))
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "training", "completed": completed, "total": EXPECTED_CASES,
    })
    labels = data["label"]
    for name, features in matrices.items():
        for seed in SEEDS:
            case = model_out / name / f"seed{seed}"
            metrics_path = case / "metrics.json"
            if metrics_path.exists():
                print("SKIP", model, name, seed, flush=True)
                continue
            config = replace(
                _probe_config(root_config, seed),
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
                "total": EXPECTED_CASES, "feature_group": name, "seed": seed,
            })
            print("DONE", model, name, seed, flush=True)
    summarize(model, model_out, audit)
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "complete", "completed": EXPECTED_CASES, "total": EXPECTED_CASES,
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
