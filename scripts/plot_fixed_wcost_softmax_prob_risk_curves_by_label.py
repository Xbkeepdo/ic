#!/usr/bin/env python3
"""Plot the fixed softmax-probability weighted-cost risk by CHAIR label."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/fixed_wcost_softmax_prob_norm_semantic_transform_fusion"
OUT = ROOT / "outputs/fixed_wcost_softmax_prob_risk_curves_by_label"
MODELS = ["llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b", "qwen3_vl_8b"]
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}
FIELD = (
    "dgst_t_hpre_softmax_prob_gauss_risk_linear_matched_state_target_"
    "w02_hpre_source_softmax_per_layer"
)


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    sem = values.std(axis=0, ddof=1, dtype=np.float64) / np.sqrt(values.shape[0])
    return mean, sem


def load_stats(model: str) -> dict:
    with np.load(SOURCE / model / "aligned_features.npz", allow_pickle=False) as saved:
        risk = np.asarray(saved["risk"], dtype=np.float64)
        labels = np.asarray(saved["label"], dtype=np.int8)
        test = np.asarray(saved["test_mask"], dtype=bool)
    if risk.ndim != 2 or len(labels) != len(risk) or len(test) != len(risk):
        raise ValueError(f"{model}: incompatible aligned arrays")
    if not np.isfinite(risk).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model}: invalid risk or labels")
    hall_values = risk[test & (labels == 0)]
    real_values = risk[test & (labels == 1)]
    if len(hall_values) < 2 or len(real_values) < 2:
        raise ValueError(f"{model}: insufficient test examples")
    hall_mean, hall_sem = mean_sem(hall_values)
    real_mean, real_sem = mean_sem(real_values)
    return {
        "hall_count": len(hall_values), "real_count": len(real_values),
        "hall_mean": hall_mean, "hall_sem": hall_sem,
        "real_mean": real_mean, "real_sem": real_sem,
    }


def draw(axis, model: str, stats: dict) -> None:
    layers = np.arange(1, len(stats["hall_mean"]) + 1)
    for label, color, prefix in (
        ("Hallucination", "#d55e00", "hall"),
        ("Non-hallucination", "#0072b2", "real"),
    ):
        mean = stats[f"{prefix}_mean"]
        sem = stats[f"{prefix}_sem"]
        axis.plot(layers, mean, color=color, lw=2.1, label=label)
        axis.fill_between(layers, mean - sem, mean + sem, color=color, alpha=.18, linewidth=0)
    axis.set_title(f"{DISPLAY[model]} · HALL {stats['hall_count']:,} / REAL {stats['real_count']:,}")
    axis.set_xlabel("Decoder layer (1-based)")
    axis.set_ylabel("Mean transport risk")
    axis.set_xlim(1, int(layers[-1]))
    axis.grid(alpha=.23)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats_by_model = {model: load_stats(model) for model in MODELS}
    rows = []
    summaries = []
    for model, stats in stats_by_model.items():
        gap = stats["hall_mean"] - stats["real_mean"]
        peak = int(np.argmax(np.abs(gap)))
        summaries.append({
            "model": model,
            "hall_count": stats["hall_count"], "real_count": stats["real_count"],
            "layers": len(gap), "hall_layer_mean": float(stats["hall_mean"].mean()),
            "real_layer_mean": float(stats["real_mean"].mean()),
            "hall_minus_real_layer_mean": float(gap.mean()),
            "peak_abs_gap_layer": peak + 1, "peak_hall_minus_real": float(gap[peak]),
        })
        for layer in range(len(gap)):
            rows.append({
                "model": model, "layer": layer + 1,
                "hall_count": stats["hall_count"], "real_count": stats["real_count"],
                "hall_mean": stats["hall_mean"][layer], "hall_sem": stats["hall_sem"][layer],
                "real_mean": stats["real_mean"][layer], "real_sem": stats["real_sem"][layer],
                "hall_minus_real": gap[layer],
            })
        folder = OUT / model
        folder.mkdir(exist_ok=True)
        fig, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
        draw(axis, model, stats)
        axis.legend(frameon=False)
        fig.savefig(folder / "test.png", dpi=220)
        fig.savefig(folder / "test.pdf")
        plt.close(fig)

    with (OUT / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader(); writer.writerows(summaries)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        draw(axis, model, stats_by_model[model])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.015))
    fig.suptitle("Fixed softmax-probability weighted-cost risk · test split · mean ± SEM",
                 y=1.065)
    fig.savefig(OUT / "test_all_models.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "test_all_models.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Fixed softmax-probability weighted-cost risk curves by label", "",
        f"Field: `{FIELD}`.",
        "Only the strict 800-image test split is used. Curves are target-level means; shaded regions are SEM. Labels are 0=HALL and 1=REAL. Values are raw transport risks without cross-model normalization.", "",
        "[Combined PNG](test_all_models.png) · [Combined PDF](test_all_models.pdf) · [Layer CSV](curves.csv)", "",
        "| Model | HALL / REAL | Layer mean HALL | Layer mean REAL | HALL−REAL | Largest absolute gap layer | Plot |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY[row['model']]} | {row['hall_count']} / {row['real_count']} | "
            f"{row['hall_layer_mean']:.6f} | {row['real_layer_mean']:.6f} | "
            f"{row['hall_minus_real_layer_mean']:+.6f} | {row['peak_abs_gap_layer']} "
            f"({row['peak_hall_minus_real']:+.6f}) | "
            f"[PNG]({row['model']}/test.png) · [PDF]({row['model']}/test.pdf) |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
