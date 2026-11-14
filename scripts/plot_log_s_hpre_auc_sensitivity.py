#!/usr/bin/env python3
"""Plot the Risk+F_log K/tau AUROC sensitivity for four LVLMs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/log_s_hpre_k_tau_ablation"
MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL"),
    ("internvl_2_5_8b", "InternVL2.5"),
    ("qwen3_vl_8b", "Qwen3-VL"),
)
KS = (16, 32, 64, 128)
TAUS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
SELECTED_K = 128
SELECTED_TAU = 0.03
REFERENCE_K = 64
REFERENCE_TAU = 0.07
SHADES = ("#9b9b9b", "#737373", "#464646", "#111111")
MARKERS = ("o", "s", "^", "D")


def read_grid(model: str, split: str) -> dict[tuple[int, float], float]:
    name = "validation_summary.csv" if split == "validation" else "test_grid_summary.csv"
    with (EXPERIMENT / model / name).open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["group"] == "risk_f_log"]
    values = {(int(r["top_k"]), round(float(r["tau"]), 2)): 100 * float(r["real_auc_mean"]) for r in rows}
    expected = {(k, tau) for k in KS for tau in TAUS}
    if set(values) != expected or len(rows) != len(expected):
        raise ValueError(f"{model}/{split}: incomplete or duplicate K/tau grid")
    return values


def make_figure(split: str) -> list[Path]:
    grids = {model: read_grid(model, split) for model, _ in MODELS}
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.7), sharey="col")
    for col, (model, label) in enumerate(MODELS):
        grid = grids[model]
        upper, lower = axes[0, col], axes[1, col]
        ymin, ymax = min(grid.values()), max(grid.values())
        margin = max(0.055, 0.13 * (ymax - ymin))
        for axis in (upper, lower):
            axis.set_ylim(ymin - margin, ymax + margin)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(axis="y", color="#dedede", linewidth=0.55, alpha=0.9)
            axis.tick_params(direction="out", length=2.5, width=0.7)
            axis.axhline(grid[(REFERENCE_K, REFERENCE_TAU)], color="#a5a5a5", ls="--", lw=0.9, zorder=1)

        for k, color, marker in zip(KS, SHADES, MARKERS):
            upper.plot(
                TAUS, [grid[(k, tau)] for tau in TAUS],
                color=color, lw=1.25 if k != SELECTED_K else 1.6,
                marker=marker, ms=3.3, markerfacecolor="white",
                markeredgewidth=0.85, zorder=3,
            )
        upper.plot(
            SELECTED_TAU, grid[(SELECTED_K, SELECTED_TAU)],
            marker="*", color="#111111", markersize=9, linestyle="None", zorder=5,
        )
        upper.set_title(label, fontweight="bold", pad=8)
        upper.set_xlim(0.015, 0.085)
        upper.set_xticks(TAUS)
        upper.set_xticklabels([f"{tau:.2f}" for tau in TAUS], rotation=45)
        upper.set_xlabel(r"Source temperature $\tau$")

        lower.plot(
            KS, [grid[(k, SELECTED_TAU)] for k in KS],
            color="#111111", lw=1.45, marker="o", ms=4,
            markerfacecolor="white", markeredgewidth=0.9, zorder=3,
        )
        lower.plot(
            SELECTED_K, grid[(SELECTED_K, SELECTED_TAU)],
            marker="*", color="#111111", markersize=9, linestyle="None", zorder=5,
        )
        lower.set_xticks(KS)
        lower.set_xlim(9, 135)
        lower.set_xlabel(r"Transport support $K$ ($\tau=0.03$)")

    axes[0, 0].set_ylabel("AUROC (%)")
    axes[1, 0].set_ylabel("AUROC (%)")
    handles = [
        Line2D([0], [0], color=color, marker=marker, markerfacecolor="white", lw=1.4, ms=4, label=f"K={k}")
        for k, color, marker in zip(KS, SHADES, MARKERS)
    ]
    handles += [
        Line2D([0], [0], color="#a5a5a5", ls="--", lw=0.9, label=r"Reference $K=64,\ \tau=0.07$"),
        Line2D([0], [0], color="#111111", marker="*", lw=0, ms=8, label=r"Selected $K=128,\ \tau=0.03$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=6, frameon=False, fontsize=8)
    fig.suptitle(
        "Risk + $F_{\\mathrm{log}}$ AUROC sensitivity "
        + ("(internal validation)" if split == "validation" else "(held-out test)"),
        y=0.995, fontsize=12, fontweight="bold",
    )
    fig.subplots_adjust(left=0.057, right=0.99, top=0.89, bottom=0.17, wspace=0.24, hspace=0.46)
    output = EXPERIMENT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"risk_f_log_auc_sensitivity_four_models_{split}"
    paths = []
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "both"), default="both")
    args = parser.parse_args()
    for split in (("validation", "test") if args.split == "both" else (args.split,)):
        for path in make_figure(split):
            print(path)


if __name__ == "__main__":
    main()
