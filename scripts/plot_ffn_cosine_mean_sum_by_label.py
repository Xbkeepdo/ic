#!/usr/bin/env python3
"""Plot full COCO Real/Hall layer curves for FFN cosine mean and sum."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MEAN_SOURCE = ROOT / "outputs/ffn_cosine_i_s_811"
SUM_SOURCE = ROOT / "outputs/ffn_cosine_sum_i_s_811"
OUT = ROOT / "outputs/ffn_cosine_mean_sum_curves_by_label"
MODELS = (("llava_1_5_7b", "LLaVA-1.5-7B"),
          ("qwen3_vl_8b", "Qwen3-VL-8B"))
FEATURES = (("mean", r"$F_{\mathrm{mean}}$"),
            ("sum", r"$F_{\mathrm{sum}}$"))
GROUPS = ((1, "Real", "#0072B2"), (0, "Hallucination", "#D55E00"))


def clustered_mean_ci(values: np.ndarray, image_ids: np.ndarray):
    """Token-weighted mean and image-clustered 95% normal-approximation CI."""
    n, layers = values.shape
    unique, inverse = np.unique(image_ids, return_inverse=True)
    clusters = len(unique)
    if n < 2 or clusters < 2:
        raise ValueError("at least two tokens and two images are required")
    mean = values.mean(axis=0)
    cluster_residual = np.zeros((clusters, layers), dtype=np.float64)
    np.add.at(cluster_residual, inverse, values - mean)
    se = np.sqrt((clusters / (clusters - 1)) *
                 np.square(cluster_residual).sum(axis=0)) / n
    return mean, se, mean - 1.96 * se, mean + 1.96 * se, clusters


def load_model(model: str):
    with np.load(MEAN_SOURCE / model / "compact_features.npz", allow_pickle=False) as saved:
        labels = np.asarray(saved["label"], dtype=np.int8)
        images = np.asarray(saved["image_id"], dtype=np.int64)
        responses = np.asarray(saved["response_token_idx"], dtype=np.int32)
        tokens = np.asarray(saved["target_token_id"], dtype=np.int32)
        matrix = np.asarray(saved["i_s_ffn_cos_mean"], dtype=np.float64)
    layers = matrix.shape[1] // 3
    if matrix.shape != (len(labels), 3 * layers) or layers < 1:
        raise ValueError(f"{model}: invalid mean matrix")
    mean = matrix[:, 2 * layers:]
    with np.load(SUM_SOURCE / model / "compact_features.npz", allow_pickle=False) as saved:
        for name, expected in (("label", labels), ("image_id", images),
                               ("response_token_idx", responses),
                               ("target_token_id", tokens)):
            if not np.array_equal(saved[name], expected):
                raise ValueError(f"{model}: mean/sum {name} alignment differs")
        summation = np.asarray(saved["ffn_cosine_sum"], dtype=np.float64)
        counts = np.asarray(saved["visual_count"], dtype=np.int32)
    if (summation.shape != mean.shape or counts.shape != labels.shape or
            not np.isfinite(mean).all() or not np.isfinite(summation).all() or
            set(np.unique(labels).tolist()) != {0, 1} or (counts <= 0).any()):
        raise ValueError(f"{model}: invalid feature or label array")
    max_error = float(np.abs(summation - mean * counts[:, None]).max())
    if max_error > 1e-4:
        raise ValueError(f"{model}: mean/sum identity differs by {max_error}")
    return {"mean": mean, "sum": summation}, labels, images, layers, max_error


def collect():
    rows, model_stats = [], []
    for model, display in MODELS:
        features, labels, images, layers, error = load_model(model)
        model_stats.append((model, display, len(labels), len(set(images.tolist())),
                            int((labels == 1).sum()), int((labels == 0).sum()), error))
        for feature_name, values in features.items():
            for label, label_name, _color in GROUPS:
                mask = labels == label
                mean, se, lower, upper, clusters = clustered_mean_ci(
                    values[mask], images[mask]
                )
                for index in range(layers):
                    rows.append({
                        "model": model, "feature": feature_name,
                        "label": label_name, "layer": index + 1,
                        "token_count": int(mask.sum()), "image_count": clusters,
                        "mean": float(mean[index]), "se_image_cluster": float(se[index]),
                        "ci95_lower": float(lower[index]), "ci95_upper": float(upper[index]),
                    })
    return rows, model_stats


def plot(rows):
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.9))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.85,
                        hspace=0.30, wspace=0.24)
    handles = []
    for column, (model, display) in enumerate(MODELS):
        axes[0, column].set_title(display, weight="semibold", pad=10)
        for row_index, (feature, label_text) in enumerate(FEATURES):
            ax = axes[row_index, column]
            selected = [r for r in rows if r["model"] == model and r["feature"] == feature]
            n_layers = max(r["layer"] for r in selected)
            for _label_value, label, color in GROUPS:
                group = sorted((r for r in selected if r["label"] == label),
                               key=lambda r: r["layer"])
                x = np.asarray([r["layer"] for r in group])
                mean = np.asarray([r["mean"] for r in group])
                lower = np.asarray([r["ci95_lower"] for r in group])
                upper = np.asarray([r["ci95_upper"] for r in group])
                line, = ax.plot(x, mean, color=color, lw=2.0, label=label)
                ax.fill_between(x, lower, upper, color=color, alpha=0.17, lw=0)
                if row_index == 0 and column == 0:
                    handles.append(line)
            ax.axhline(0, color="#777777", lw=0.7, alpha=0.5)
            ax.set_xlim(1, n_layers)
            ax.set_xticks([1, *(tick for tick in range(5, n_layers, 5)
                                  if n_layers - tick >= 3), n_layers])
            ax.grid(axis="y", color="#d0d0d0", lw=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylabel(f"{label_text} (cosine)")
            if row_index == 1:
                ax.set_xlabel("Decoder layer")
    fig.legend(handles, ["Real", "Hallucination"], ncol=2,
               loc="upper center", bbox_to_anchor=(0.52, 0.98), frameon=False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"ffn_cosine_mean_sum_real_hall_all4000.{suffix}",
                    dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, stats = collect()
    with (OUT / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    lines = [
        "# FFN 视觉对齐特征的 Real/Hall 逐层曲线", "",
        "全量 COCO4000 ENDAC object token；F_mean 为视觉 token 的原始 FFN-hpre cosine 均值，F_sum 为其直接求和。曲线为 object token 加权均值；阴影为按图片聚类的近似 95% CI。每个模型和特征使用自身纵轴尺度，图中未进行检测器训练。", "",
        "[PNG](ffn_cosine_mean_sum_real_hall_all4000.png) · [PDF](ffn_cosine_mean_sum_real_hall_all4000.pdf) · [SVG](ffn_cosine_mean_sum_real_hall_all4000.svg) · [逐层数值](curves.csv)", "",
        "| 模型 | Object token | 有目标图片 | Real token | Hall token | F_sum−M×F_mean 最大绝对差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, display, n, images, real, hall, error in stats:
        lines.append(f"| {display} | {n} | {images} | {real} | {hall} | {error:.3g} |")
    lines += ["", "| 模型 | 特征 | 最大 Real−Hall 均值差及层 | 最后一层 Real−Hall 均值差 |",
              "|---|---|---:|---:|"]
    for model, display in MODELS:
        for feature, feature_display in FEATURES:
            selected = [row for row in rows if row["model"] == model and row["feature"] == feature]
            means = {(row["label"], row["layer"]): row["mean"] for row in selected}
            layers = max(row["layer"] for row in selected)
            gaps = {layer: means[("Real", layer)] - means[("Hallucination", layer)]
                    for layer in range(1, layers + 1)}
            peak = max(gaps, key=gaps.get)
            lines.append(f"| {display} | {feature_display} | {gaps[peak]:+.4f} (layer {peak}) | {gaps[layers]:+.4f} |")
    lines += ["", "两种聚合的相应检测结果见 [FFN mean 实验](../ffn_cosine_i_s_811/complete_results.md) 和 [FFN sum 实验](../ffn_cosine_sum_i_s_811/complete_results.md)。", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(OUT / "ffn_cosine_mean_sum_real_hall_all4000.png", flush=True)


if __name__ == "__main__":
    main()
