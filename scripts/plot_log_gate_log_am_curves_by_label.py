#!/usr/bin/env python3
"""Plot F_l=sum_m log1p(gate_lm)log1p(||a_lm||) by CHAIR label."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BY_MODE = {
    "log": ROOT / "outputs/fixed_wcost_log_gate_log_am",
    "raw": ROOT / "outputs/fixed_wcost_raw_gate_am",
}
MODELS = ("llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b", "qwen3_vl_8b")
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def load_rows(source: Path) -> tuple[list[dict], list[dict]]:
    curves, summaries = [], []
    for model in MODELS:
        saved = np.load(source / model / "compact_features.npz", allow_pickle=False)
        feature = np.asarray(saved["feature"], dtype=np.float64)
        labels = np.asarray(saved["label"], dtype=np.int8)
        if feature.ndim != 2 or not np.isfinite(feature).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{model}: invalid cached feature")
        for split, selected in (
            ("all", np.ones(len(labels), dtype=bool)),
            ("test", np.asarray(saved["test_mask"], dtype=bool)),
        ):
            selected_labels = labels[selected]
            selected_feature = feature[selected]
            hall = selected_feature[selected_labels == 0]
            real = selected_feature[selected_labels == 1]
            hall_mean, hall_sem = stats(hall)
            real_mean, real_sem = stats(real)
            gap = hall_mean - real_mean
            peak = int(np.argmax(np.abs(gap)))
            summaries.append({
                "model": model,
                "split": split,
                "hall_count": len(hall),
                "real_count": len(real),
                "layers": feature.shape[1],
                "hall_layer_average": float(hall_mean.mean()),
                "real_layer_average": float(real_mean.mean()),
                "layer_average_gap": float(gap.mean()),
                "peak_abs_gap_layer": peak + 1,
                "peak_gap": float(gap[peak]),
            })
            for index in range(feature.shape[1]):
                curves.append({
                    "model": model,
                    "split": split,
                    "layer": index + 1,
                    "hall_count": len(hall),
                    "real_count": len(real),
                    "hall_mean": float(hall_mean[index]),
                    "hall_sem": float(hall_sem[index]),
                    "real_mean": float(real_mean[index]),
                    "real_sem": float(real_sem[index]),
                    "hall_minus_real": float(gap[index]),
                })
    return curves, summaries


def plot(split: str, curves: list[dict], output_dir: Path, feature_mode: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        selected = [row for row in curves if row["model"] == model and row["split"] == split]
        layers = np.asarray([row["layer"] for row in selected])
        for prefix, label, color in (
            ("hall", "Hallucination", "#d55e00"),
            ("real", "Non-hallucination", "#0072b2"),
        ):
            mean = np.asarray([row[f"{prefix}_mean"] for row in selected])
            sem = np.asarray([row[f"{prefix}_sem"] for row in selected])
            axis.plot(layers, mean, color=color, linewidth=2.1, label=label)
            axis.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.18)
        axis.set_title(DISPLAY[model])
        axis.set_xlabel("Decoder layer (1-based)")
        axis.set_ylabel(r"Mean $F_l$")
        axis.set_xlim(1, int(layers[-1]))
        axis.grid(alpha=0.23)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    title_split = "all aligned targets" if split == "all" else "held-out 800-image test split"
    formula = (
        r"$F_l=\sum_m \log(1+G_{l,m})\log(1+\Vert a_{l,m}\Vert_2)$"
        if feature_mode == "log"
        else r"$F_l=\sum_m G_{l,m}\Vert a_{l,m}\Vert_2$"
    )
    fig.suptitle(
        formula + f"\n{title_split}; shaded band = ±1 SEM",
        y=1.085,
    )
    fig.savefig(output_dir / f"{split}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{split}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", choices=tuple(SOURCE_BY_MODE), default="log")
    args = parser.parse_args()
    source = SOURCE_BY_MODE[args.feature_mode]
    output_dir = source / "curves_by_label"
    output_dir.mkdir(parents=True, exist_ok=True)
    curves, summaries = load_rows(source)
    with (output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for split in ("all", "test"):
        plot(split, curves, output_dir, args.feature_mode)
    definition = (
        "`F_l = sum_m log1p(G_lm) log1p(||a_lm||_2)`"
        if args.feature_mode == "log"
        else "`F_l = sum_m G_lm ||a_lm||_2`"
    )
    lines = [
        "# F_l 的幻觉与非幻觉逐层曲线", "",
        f"{definition}；曲线为目标 token 均值，阴影为 ±1 SEM。标签 0=幻觉，1=非幻觉。", "",
        "| 模型 | 样本 | HALL / REAL | HALL 层平均 | REAL 层平均 | HALL−REAL | 最大绝对差层 | 该层差值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY[row['model']]} | {row['split']} | {row['hall_count']} / {row['real_count']} | "
            f"{row['hall_layer_average']:.5f} | {row['real_layer_average']:.5f} | "
            f"{row['layer_average_gap']:+.5f} | {row['peak_abs_gap_layer']} | {row['peak_gap']:+.5f} |"
        )
    lines += [
        "", "## 图和数据", "",
        "- 全部目标：[PNG](all.png) · [PDF](all.pdf)",
        "- 测试集目标：[PNG](test.png) · [PDF](test.pdf)",
        "- [逐层 CSV](curves.csv) · [汇总 CSV](summary.csv)",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
