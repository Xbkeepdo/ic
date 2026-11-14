#!/usr/bin/env python3
"""Render the SAR/EMS/SAR+EMS layer ablation as TeX and a vector table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_layer_component_ablation_k064_811/llava_1_5_7b"
TOP_K = 64
SPLIT_LABEL = "8:1:1"
LAYERS = ("1-8", "9-18", "19-27", "28-32", "1-32")
GROUPS = (("sar", "SAR"), ("ems", "EMS"), ("sar_ems", "SAR+EMS"))
METRICS = (("auroc", "AUC"), ("real_aupr", "AUPR"), ("real_f1", "F1"))
STEM = "layer_component_k064_real_positive"


def load_means() -> dict[tuple[str, str, str], float]:
    rows = list(csv.DictReader((OUT / "per_seed_results.csv").open()))
    buckets = defaultdict(list)
    metric_alias = {"auroc": "auc", "real_aupr": "aupr", "real_f1": "f1"}
    for row in rows:
        for metric, _ in METRICS:
            key = metric if metric in row else metric_alias[metric]
            buckets[(row["group"], row["layer_set"], metric)].append(float(row[key]))
    expected = {(group, layer, metric) for group, _ in GROUPS
                for layer in LAYERS for metric, _ in METRICS}
    if set(buckets) != expected or any(len(v) != 3 for v in buckets.values()):
        raise ValueError("Expected three seed scores for each of the 45 table cells")
    return {key: 100 * float(np.mean(values)) for key, values in buckets.items()}


def maxima(data: dict) -> set[tuple[str, str, str]]:
    winners = set()
    for group, _ in GROUPS:
        for metric, _ in METRICS:
            best = max(data[(group, layer, metric)] for layer in LAYERS)
            winners.update((group, layer, metric) for layer in LAYERS
                           if np.isclose(data[(group, layer, metric)], best))
    return winners


def write_tex(data: dict, winners: set) -> None:
    lines = [
        "% Requires \\usepackage{booktabs,multirow}.",
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccccc@{}}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Layer Set}}"
        " & \\multicolumn{3}{c}{\\textbf{SAR}}"
        " & \\multicolumn{3}{c}{\\textbf{EMS}}"
        " & \\multicolumn{3}{c}{\\textbf{SAR+EMS}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(l){8-10}",
        " & " + " & ".join(label for _group, _ in GROUPS for _metric, label in METRICS)
        + " \\\\",
        "\\midrule",
    ]
    for layer in LAYERS:
        label = "All layers" if layer == "1-32" else f"Layers {layer.replace('-', '--')}"
        if layer == "1-32":
            label = "\\textbf{All layers}"
        cells = []
        for group, _ in GROUPS:
            for metric, _ in METRICS:
                value = f"{data[(group, layer, metric)]:.2f}"
                if (group, layer, metric) in winners:
                    value = f"\\textbf{{{value}}}"
                cells.append(value)
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule", "\\end{tabular*}",
        f"\\caption{{Layer-set ablation on LLaVA-1.5-7B with $K={TOP_K}$ and $\\tau=0.03$."
        " AUC denotes AUROC; AUPR and F1 treat real object mentions as positive."
        " Values are test-set percentages averaged over seeds 43, 44, and 45."
        f" F1 uses a Real-F1-maximizing threshold selected on the {'training' if SPLIT_LABEL == '8:2' else 'validation'} split."
        f" The full-layer rows use the same original {SPLIT_LABEL} evaluation."
        " Bold marks the best layer set within each feature group and metric.}",
        f"\\label{{tab:llava-logs-k{TOP_K}-{SPLIT_LABEL.replace(':', '')}-layer-component}}",
        "\\end{table*}", "",
    ])
    (OUT / f"{STEM}.tex").write_text("\n".join(lines))


def draw(data: dict, winners: set) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11.4, 4.1))
    ax.set_xlim(0, 10.45)
    ax.set_ylim(0, 4.12)
    ax.axis("off")
    label_width, col_width, group_gap = 1.58, 0.92, 0.28
    centers = [label_width + group * (3 * col_width + group_gap)
               + (metric + 0.5) * col_width for group in range(3)
               for metric in range(3)]
    ax.plot([0, 10.42], [3.92, 3.92], color="black", lw=1.35)
    ax.text(0.08, 3.50, "Layer Set", weight="bold", va="center", fontsize=11.5)
    for index, (_group, label) in enumerate(GROUPS):
        subset = centers[3 * index:3 * index + 3]
        ax.text(np.mean(subset), 3.73, label, weight="bold", ha="center",
                va="center", fontsize=13)
        ax.plot([subset[0] - 0.43, subset[-1] + 0.43], [3.49, 3.49],
                color="black", lw=0.65)
        for offset, (_metric, metric_label) in enumerate(METRICS):
            ax.text(subset[offset], 3.22, metric_label, ha="center",
                    va="center", fontsize=10.5, weight="bold")
    ax.plot([0, 10.42], [2.96, 2.96], color="black", lw=0.9)
    for row_index, layer in enumerate(LAYERS):
        y = 2.72 - row_index * 0.48
        text = "All layers" if layer == "1-32" else f"Layers {layer.replace('-', '–')}"
        weight = "bold" if layer == "1-32" else "normal"
        ax.text(0.08, y, text, ha="left", va="center", fontsize=11, weight=weight)
        for group_index, (group, _label) in enumerate(GROUPS):
            for metric_index, (metric, _label) in enumerate(METRICS):
                ax.text(centers[3 * group_index + metric_index], y,
                        f"{data[(group, layer, metric)]:.2f}", ha="center",
                        va="center", fontsize=10.7,
                        weight="bold" if (group, layer, metric) in winners else "normal")
    ax.plot([0, 10.42], [0.55, 0.55], color="black", lw=1.35)
    ax.text(0.03, 0.20,
            f"LLaVA-1.5-7B  ·  K = {TOP_K}  ·  τ = 0.03  ·  {SPLIT_LABEL} split  ·  Real-positive test scores (%)",
            fontsize=9, color="0.35", va="center")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.99, bottom=0.015)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{STEM}.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, choices=(32, 64), default=64)
    parser.add_argument("--split-protocol", choices=("811", "82"), default="811")
    args = parser.parse_args()
    TOP_K = args.top_k
    SPLIT_LABEL = "8:2" if args.split_protocol == "82" else "8:1:1"
    OUT = ROOT / "outputs" / f"log_s_layer_component_ablation_k{TOP_K:03d}_{args.split_protocol}" / "llava_1_5_7b"
    STEM = f"layer_component_k{TOP_K:03d}_real_positive"
    values = load_means()
    bold = maxima(values)
    write_tex(values, bold)
    draw(values, bold)
    print(OUT / f"{STEM}.pdf")
