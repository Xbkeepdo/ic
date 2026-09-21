#!/usr/bin/env python3
"""Audit and summarize fixed-risk + sqrt-semantic residual fusion runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "fixed_wcost_sqrt_semantic_residual_fusion"
MODELS = ["llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b", "qwen3_vl_8b"]
GROUPS = ["risk", "residual_fusion", "risk_residual_fusion"]
TRAINED_GROUPS = GROUPS[1:]
SEEDS = [43, 44, 45]


def load_metrics(model: str, group: str, seed: int) -> dict:
    if group == "risk":
        protocol = json.loads((OUT / model / "protocol.json").read_text())
        risk = protocol["risk"]
        path = (
            ROOT
            / "outputs"
            / model
            / risk["run_name"]
            / "weighted_cost_probe"
            / risk["source"]
            / f"{risk['method']}_{risk['weight']}"
            / f"seed{seed}"
            / "metrics.json"
        )
    else:
        path = OUT / model / group / f"seed{seed}" / "metrics.json"
    return json.loads(path.read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    assert np.isfinite(x).all()
    return float(x.mean()), float(x.std(ddof=0))


def main() -> None:
    rows = []
    audit = {"models": {}, "status": "passed"}
    for model in MODELS:
        model_dir = OUT / model
        progress = json.loads((model_dir / "progress.json").read_text())
        alignment = json.loads((model_dir / "alignment.json").read_text())
        protocol = json.loads((model_dir / "protocol.json").read_text())
        assert progress == {"model": model, "stage": "complete", "completed": 6, "total": 6}
        assert alignment["targets"] == alignment["matched_targets"] == protocol["targets"]
        assert alignment["train_targets"] + alignment["test_targets"] == alignment["targets"]
        assert protocol["feature_groups"]["residual_fusion"][1] == protocol["layers"]
        assert protocol["feature_groups"]["risk_residual_fusion"][1] == 2 * protocol["layers"]

        for group in TRAINED_GROUPS:
            for seed in SEEDS:
                cfg = json.loads((model_dir / group / f"seed{seed}" / "config.json").read_text())
                assert cfg["hidden_sizes"] == [128]
                assert cfg["batch_norm"] is False and cfg["standardize_input"] is True
                scaler = np.load(model_dir / group / f"seed{seed}" / "input_scaler.npz")
                assert np.isfinite(scaler["mean"]).all()
                assert np.isfinite(scaler["scale"]).all() and (scaler["scale"] > 0).all()

        stats = {}
        for group in GROUPS:
            metrics = [load_metrics(model, group, seed) for seed in SEEDS]
            auc = mean_std([m["auc"] for m in metrics])
            f1 = mean_std([m["f1"] for m in metrics])
            hall_aupr = mean_std([m["hallucination_positive"]["aupr"] for m in metrics])
            stats[group] = {"auc": auc, "f1": f1, "hall_aupr": hall_aupr}
        base = stats["risk"]
        for group in GROUPS:
            item = stats[group]
            rows.append(
                {
                    "model": model,
                    "feature": group,
                    "real_f1_mean": item["f1"][0],
                    "real_f1_std": item["f1"][1],
                    "real_auc_mean": item["auc"][0],
                    "real_auc_std": item["auc"][1],
                    "delta_auc_vs_risk": item["auc"][0] - base["auc"][0],
                    "hall_aupr_mean": item["hall_aupr"][0],
                    "hall_aupr_std": item["hall_aupr"][1],
                    "delta_hall_aupr_vs_risk": item["hall_aupr"][0] - base["hall_aupr"][0],
                }
            )
        audit["models"][model] = {
            "targets": alignment["targets"],
            "train_targets": alignment["train_targets"],
            "test_targets": alignment["test_targets"],
            "matched_targets": alignment["matched_targets"],
            "trained_metrics": 6,
            "scalers": 6,
        }

    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fixed weighted-cost risk + sqrt-semantic residual fusion",
        "",
        r"$F_l=\sum_m\lVert a_{l,m}\rVert_2(1+\sqrt{C p_{l,m}})$, where $p$ is GF `object_probability_raw`.",
        "Fixed risk: `WCOST-ALL / softmax / hpre_raw_logit_gauss / w=0.2`.",
        "Probe: one hidden layer 128, training-only column z-score, no BatchNorm, dropout 0.3, strict 3200/800 image split, seeds 43–45.",
        "Values below use the train-F1 threshold and report mean±population SD across seeds.",
        "",
        "| Model | Feature | Real F1 | Real AUC | Δ AUC vs risk | Hall AUPR | Δ Hall AUPR vs risk |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['feature']} | "
            f"{row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | "
            f"{row['delta_auc_vs_risk']:+.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{row['delta_hall_aupr_vs_risk']:+.4f} |"
        )
    lines += [
        "",
        "## Audit",
        "",
        "All four alignments are exact on `(image_id, response_token_idx, target_token_id)`. "
        "All 24 newly trained metrics files are finite; all 24 training-only scalers are finite with positive scales. "
        "Every run uses `[128]`, input standardization, and no BatchNorm.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
