#!/usr/bin/env python3
"""Plot sums of normal-CDF gates over MAD-standardized semantic probabilities."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import MODELS  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


RUN = "COCO4000-512-WCOST-ALL"
FIELD = "dgst_t_hpre_softmax_prob_gauss_gate_per_layer"
OUT = ROOT / "outputs/softmax_gauss_mad_cdf_sum_curves_by_label"
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def sigmoid_gate_to_normal_cdf(gate: np.ndarray) -> np.ndarray:
    """Convert saved sigmoid(z) to Phi(z), preserving saturated limits."""
    gate = np.asarray(gate, dtype=np.float64)
    if not np.isfinite(gate).all() or (gate < 0).any() or (gate > 1).any():
        raise ValueError("Saved Gaussian MAD gate must be finite and in [0, 1]")
    tensor = torch.from_numpy(gate)
    # logit(0/1) gives -/+inf, and ndtr maps those limits exactly to 0/1.
    return torch.special.ndtr(torch.logit(tensor)).numpy()


def load_model(model: str) -> dict[str, np.ndarray]:
    run = ROOT / "outputs" / model / RUN
    rows = load_pkl(run / "features.pkl")
    if not rows or FIELD not in rows[0]:
        raise KeyError(f"{model}: missing {FIELD}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    first = np.asarray(rows[0][FIELD])
    gate_sum = np.empty((len(rows), first.shape[0]), dtype=np.float64)
    indices_by_shape: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        shape = tuple(np.asarray(row[FIELD]).shape)
        if len(shape) != 2 or shape[0] != first.shape[0]:
            raise ValueError(f"{model}: invalid gate shape at row {index}: {shape}")
        indices_by_shape.setdefault(shape, []).append(index)
    for indices in indices_by_shape.values():
        for start in range(0, len(indices), 64):
            chunk = indices[start:start + 64]
            saved_gate = np.stack([
                np.asarray(rows[index][FIELD], dtype=np.float32)
                for index in chunk
            ])
            gate_sum[chunk] = sigmoid_gate_to_normal_cdf(saved_gate).sum(axis=2)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model}: expected labels 0 and 1")
    if gate_sum.ndim != 2 or not np.isfinite(gate_sum).all() or (gate_sum < 0).any():
        raise ValueError(f"{model}: invalid gate sum matrix {gate_sum.shape}")
    split = json.loads((run / "image_splits.json").read_text())
    test_ids = np.asarray(split["test"], dtype=np.int64)
    data = {
        "label": labels,
        "gate_sum": gate_sum,
        "all": np.ones(len(rows), dtype=bool),
        "test": np.isin(image_ids, test_ids),
    }
    del rows
    gc.collect()
    return data


def class_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, sem


def summarize(model: str, data: dict[str, np.ndarray], split: str) -> tuple[list[dict], dict]:
    selected = data[split]
    labels = data["label"][selected]
    values = data["gate_sum"][selected]
    hall_values = values[labels == 0]
    real_values = values[labels == 1]
    if len(hall_values) < 2 or len(real_values) < 2:
        raise ValueError(f"{model}/{split}: each class requires at least two rows")
    hall_mean, hall_sem = class_stats(hall_values)
    real_mean, real_sem = class_stats(real_values)
    rows = []
    for index in range(values.shape[1]):
        rows.append({
            "model": model,
            "split": split,
            "layer": index + 1,
            "hall_count": len(hall_values),
            "real_count": len(real_values),
            "hall_mean": float(hall_mean[index]),
            "hall_sem": float(hall_sem[index]),
            "real_mean": float(real_mean[index]),
            "real_sem": float(real_sem[index]),
            "hall_minus_real": float(hall_mean[index] - real_mean[index]),
        })
    gap = hall_mean - real_mean
    peak = int(np.argmax(np.abs(gap)))
    summary = {
        "model": model,
        "split": split,
        "hall_count": len(hall_values),
        "real_count": len(real_values),
        "layers": values.shape[1],
        "hall_layer_average": float(hall_mean.mean()),
        "real_layer_average": float(real_mean.mean()),
        "layer_average_gap": float(gap.mean()),
        "peak_abs_gap_layer": peak + 1,
        "peak_gap": float(gap[peak]),
    }
    return rows, summary


def plot_split(split: str, rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        selected = [row for row in rows if row["model"] == model and row["split"] == split]
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
        axis.set_ylabel(r"Mean $\sum_m G_{l,m}$")
        axis.set_xlim(1, int(layers[-1]))
        axis.grid(alpha=0.23)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.035))
    title_split = "all aligned targets" if split == "all" else "held-out 800-image test split"
    fig.suptitle(
        "Layerwise sum of Gaussian-MAD normal-CDF gate on semantic softmax probabilities\n"
        f"{title_split}; shaded band = ±1 SEM",
        y=1.075,
    )
    fig.savefig(output_dir / f"{split}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{split}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows, summaries = [], []
    for model in MODELS:
        data = load_model(model)
        for split in ("all", "test"):
            rows, summary = summarize(model, data, split)
            curve_rows.extend(rows)
            summaries.append(summary)
        del data
        gc.collect()
        print(f"DONE {model}", flush=True)

    with (args.output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for split in ("all", "test"):
        plot_split(split, curve_rows, args.output_dir)

    lines = [
        "# Semantic softmax probability 的 Gauss-MAD CDF gate 逐层和", "",
        "对 semantic softmax probability 沿视觉 token 维做 median/MAD 标准化："
        "`z=(p-median)/(1.4826*MAD+epsilon)`，再计算标准正态 CDF `G=Phi(z)`，最后得到"
        " `S_l=sum_m G_lm`。", "",
        f"输入字段：`{FIELD}`。该字段保存 `sigmoid(z)`；脚本通过 `z=logit(gate)` 恢复 z，"
        "再计算 `Phi(z)`。浮点饱和值按 CDF 极限处理。阴影为目标 token 间均值的 ±1 SEM；"
        "标签 0=幻觉，1=非幻觉。", "",
        "| 模型 | 样本 | HALL / REAL | HALL 层平均 | REAL 层平均 | HALL−REAL | 最大绝对差层 | 该层差值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY[row['model']]} | {row['split']} | "
            f"{row['hall_count']} / {row['real_count']} | "
            f"{row['hall_layer_average']:.4f} | {row['real_layer_average']:.4f} | "
            f"{row['layer_average_gap']:+.4f} | {row['peak_abs_gap_layer']} | "
            f"{row['peak_gap']:+.4f} |"
        )
    lines += [
        "", "## 图和数据", "",
        "- 全部目标：[PNG](all.png) · [PDF](all.pdf)",
        "- 测试集目标：[PNG](test.png) · [PDF](test.pdf)",
        "- [逐层 CSV](curves.csv) · [汇总 CSV](summary.csv)",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "summary.md", flush=True)


if __name__ == "__main__":
    main()
