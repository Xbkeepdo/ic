#!/usr/bin/env python3
"""Audit and summarize softmax-probability weighted-cost risk plus norm semantic Fs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fixed_wcost_softmax_prob_norm_semantic_transform_fusion"
OLD = ROOT / "outputs/fixed_wcost_norm_semantic_transform_fusion"
MODELS = ["llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b", "qwen3_vl_8b"]
F_NAMES = ["sqrt_residual_fusion", "log_residual_fusion", "mass_preserving_fusion", "asinh_sqrt_fusion"]
SEEDS = [43, 44, 45]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    assert np.isfinite(array).all()
    return float(array.mean()), float(array.std(ddof=0))


def assert_finite_numbers(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_finite_numbers(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_numbers(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def metrics_at(paths: list[Path]) -> dict:
    metrics = [json.loads(path.read_text()) for path in paths]
    for item in metrics:
        assert_finite_numbers(item)
    return {
        "auc": mean_std([item["auc"] for item in metrics]),
        "f1": mean_std([item["f1"] for item in metrics]),
        "hall_aupr": mean_std([item["hallucination_positive"]["aupr"] for item in metrics]),
    }


def main() -> None:
    rows = []
    baselines = []
    audit = {"status": "passed", "models": {}}
    for model in MODELS:
        model_out = OUT / model
        progress = json.loads((model_out / "progress.json").read_text())
        alignment = json.loads((model_out / "alignment.json").read_text())
        protocol = json.loads((model_out / "protocol.json").read_text())
        assert progress == {"model": model, "stage": "complete", "completed": 12, "total": 12}
        assert alignment["targets"] == alignment["matched_targets"] == protocol["targets"]
        assert protocol["risk"]["field"] == (
            "dgst_t_hpre_softmax_prob_gauss_risk_linear_matched_state_target_"
            "w02_hpre_source_softmax_per_layer"
        )
        risk_paths = [
            ROOT / "outputs" / model / protocol["risk"]["run_name"] / "weighted_cost_probe"
            / protocol["risk"]["source"]
            / f"{protocol['risk']['method']}_{protocol['risk']['weight']}"
            / f"seed{seed}" / "metrics.json"
            for seed in SEEDS
        ]
        soft_risk = metrics_at(risk_paths)
        old_risk_protocol = json.loads((OLD / model / "protocol.json").read_text())
        old_risk_paths = [
            ROOT / "outputs" / model / old_risk_protocol["risk"]["run_name"] / "weighted_cost_probe"
            / old_risk_protocol["risk"]["source"]
            / f"{old_risk_protocol['risk']['method']}_{old_risk_protocol['risk']['weight']}"
            / f"seed{seed}" / "metrics.json"
            for seed in SEEDS
        ]
        old_risk = metrics_at(old_risk_paths)
        baselines.append({
            "model": model,
            "real_f1": soft_risk["f1"], "real_auc": soft_risk["auc"],
            "hall_aupr": soft_risk["hall_aupr"], "raw_logit_auc": old_risk["auc"],
        })
        stats = {"risk": soft_risk}
        for name in F_NAMES:
            standalone = metrics_at([OLD / model / name / f"seed{seed}" / "metrics.json" for seed in SEEDS])
            group = f"risk_{name}"
            fused_paths = [model_out / group / f"seed{seed}" / "metrics.json" for seed in SEEDS]
            fused = metrics_at(fused_paths)
            old_fused = metrics_at([OLD / model / group / f"seed{seed}" / "metrics.json" for seed in SEEDS])
            stats[name] = standalone
            stats[group] = fused
            for seed, path in zip(SEEDS, fused_paths):
                case = path.parent
                cfg = json.loads((case / "config.json").read_text())
                assert cfg["hidden_sizes"] == [128]
                assert cfg["batch_norm"] is False and cfg["standardize_input"] is True
                scaler = np.load(case / "input_scaler.npz")
                assert np.isfinite(scaler["mean"]).all()
                assert np.isfinite(scaler["scale"]).all() and (scaler["scale"] > 0).all()
            assert protocol["feature_groups"][group][1] == 2 * protocol["layers"]
            rows.append({
                "model": model,
                "feature": group,
                "real_f1_mean": fused["f1"][0], "real_f1_std": fused["f1"][1],
                "real_auc_mean": fused["auc"][0], "real_auc_std": fused["auc"][1],
                "delta_auc_vs_softmax_prob_risk": fused["auc"][0] - soft_risk["auc"][0],
                "delta_auc_vs_raw_logit_fusion": fused["auc"][0] - old_fused["auc"][0],
                "hall_aupr_mean": fused["hall_aupr"][0], "hall_aupr_std": fused["hall_aupr"][1],
                "delta_hall_aupr_vs_softmax_prob_risk": fused["hall_aupr"][0] - soft_risk["hall_aupr"][0],
                "delta_hall_aupr_vs_raw_logit_fusion": fused["hall_aupr"][0] - old_fused["hall_aupr"][0],
            })
        audit["models"][model] = {
            "targets": alignment["targets"], "train_targets": alignment["train_targets"],
            "test_targets": alignment["test_targets"], "matched_targets": alignment["matched_targets"],
            "trained_metrics": 12, "scalers": 12,
            "softmax_prob_risk_auc": soft_risk["auc"][0], "raw_logit_risk_auc": old_risk["auc"][0],
        }

    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Softmax-probability weighted-cost risk + norm semantic F",
        "",
        "Risk: `dgst_t_hpre_softmax_prob_gauss_risk_linear_matched_state_target_w02_hpre_source_softmax_per_layer`.",
        "The four F features are reused unchanged from the final-RMSNorm semantic transform experiment. Probe: one hidden layer 128, training-only column z-score, no BatchNorm, dropout 0.3, seeds 43–45.",
        "Metrics use the train-F1 threshold and report mean±population SD.",
        "",
        "## Risk-only baseline",
        "",
        "| Model | Real F1 | Real AUC | Hall AUPR | Raw-logit risk AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in baselines:
        lines.append(
            f"| {row['model']} | {row['real_f1'][0]:.4f}±{row['real_f1'][1]:.4f} | "
            f"{row['real_auc'][0]:.4f}±{row['real_auc'][1]:.4f} | "
            f"{row['hall_aupr'][0]:.4f}±{row['hall_aupr'][1]:.4f} | "
            f"{row['raw_logit_auc'][0]:.4f}±{row['raw_logit_auc'][1]:.4f} |"
        )
    lines += [
        "",
        "## Risk + F",
        "",
        "| Model | Risk + F | Real F1 | Real AUC | Δ AUC vs this risk | Δ AUC vs raw-logit risk+F | Hall AUPR | Δ Hall vs this risk | Δ Hall vs raw-logit risk+F |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['feature']} | {row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | {row['delta_auc_vs_softmax_prob_risk']:+.4f} | "
            f"{row['delta_auc_vs_raw_logit_fusion']:+.4f} | {row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{row['delta_hall_aupr_vs_softmax_prob_risk']:+.4f} | {row['delta_hall_aupr_vs_raw_logit_fusion']:+.4f} |"
        )
    lines += [
        "", "## Audit", "",
        "All target rows align exactly on `(image_id, response_token_idx, target_token_id)`. All 48 new metrics and training-only scalers are finite; every scaler has positive scales. All runs use `[128]`, input standardization, and no BatchNorm.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
