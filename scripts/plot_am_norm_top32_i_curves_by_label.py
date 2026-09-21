#!/usr/bin/env python3
"""Plot layerwise sums of the 32 largest visual-token a_m norms by CHAIR label."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io_utils import load_pkl  # noqa: E402


GF_ROOT = ROOT.parent / "gfchair"
SOURCE = GF_ROOT / "outputs/coco4000_512_endac_811"
OUT = ROOT / "outputs/am_norm_top32_i_curves_by_label"
RUN = "COCO4000-512-WCOST-ALL"
TOP_K = 32
MODELS = ("llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b", "qwen3_vl_8b")
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def class_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def topk_sum(values: np.ndarray, k: int) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] < k:
        raise ValueError(f"Top-{k} is invalid for shape {values.shape}")
    return np.partition(values, values.shape[1] - k, axis=1)[:, -k:].sum(axis=1)


def load_rows() -> tuple[list[dict], list[dict]]:
    curves, summaries = [], []
    for model in MODELS:
        source_path = SOURCE / model / "am_norms/features.pkl"
        rows = load_pkl(source_path)
        if not rows:
            raise ValueError(f"{model}: empty a_m norm file")
        first = np.asarray(rows[0]["am_norm"], dtype=np.float32)
        feature = np.empty((len(rows), first.shape[0]), dtype=np.float32)
        labels = np.empty(len(rows), dtype=np.int8)
        image_ids = np.empty(len(rows), dtype=np.int64)
        for index, row in enumerate(rows):
            norms = np.asarray(row["am_norm"], dtype=np.float32)
            if norms.shape[0] != feature.shape[1] or not np.isfinite(norms).all() or (norms < 0).any():
                raise ValueError(f"{model}/{row['sample_id']}: invalid a_m norm matrix")
            feature[index] = topk_sum(norms, TOP_K)
            labels[index] = int(row["label"])
            image_ids[index] = int(row["image_id"])
        del rows
        gc.collect()
        split = json.loads(
            (ROOT / "outputs" / model / RUN / "image_splits.json").read_text()
        )
        test_mask = np.isin(image_ids, np.asarray(split["test"], dtype=np.int64))
        if set(np.unique(labels)) != {0, 1} or not np.isfinite(feature).all():
            raise ValueError(f"{model}: invalid Top-{TOP_K} feature")
        for split_name, selected in (
            ("all", np.ones(len(labels), dtype=bool)),
            ("test", test_mask),
        ):
            split_labels = labels[selected]
            split_feature = feature[selected].astype(np.float64, copy=False)
            hall = split_feature[split_labels == 0]
            real = split_feature[split_labels == 1]
            hall_mean, hall_sem = class_stats(hall)
            real_mean, real_sem = class_stats(real)
            gap = hall_mean - real_mean
            peak = int(np.argmax(np.abs(gap)))
            summaries.append({
                "model": model,
                "split": split_name,
                "hall_count": len(hall),
                "real_count": len(real),
                "layers": feature.shape[1],
                "hall_layer_average": float(hall_mean.mean()),
                "real_layer_average": float(real_mean.mean()),
                "layer_average_gap": float(gap.mean()),
                "peak_abs_gap_layer": peak + 1,
                "peak_gap": float(gap[peak]),
            })
            for layer in range(feature.shape[1]):
                curves.append({
                    "model": model,
                    "split": split_name,
                    "layer": layer + 1,
                    "hall_count": len(hall),
                    "real_count": len(real),
                    "hall_mean": float(hall_mean[layer]),
                    "hall_sem": float(hall_sem[layer]),
                    "real_mean": float(real_mean[layer]),
                    "real_sem": float(real_sem[layer]),
                    "hall_minus_real": float(gap[layer]),
                })
        print(f"DONE {model}: {len(labels)} targets", flush=True)
    return curves, summaries


def plot(split: str, curves: list[dict]) -> None:
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
        axis.set_ylabel(r"Mean $I_l^{\mathrm{top32}}$")
        axis.set_xlim(1, int(layers[-1]))
        axis.grid(alpha=0.23)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    title_split = "all aligned targets" if split == "all" else "held-out 800-image test split"
    fig.suptitle(
        r"$I_l^{\mathrm{top32}}=\sum_{m\in\mathrm{Top32}(\Vert a_{l,m}\Vert_2)}\Vert a_{l,m}\Vert_2$"
        f"\n{title_split}; shaded band = ±1 SEM",
        y=1.085,
    )
    fig.savefig(OUT / f"{split}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{split}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves, summaries = load_rows()
    with (OUT / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for split in ("all", "test"):
        plot(split, curves)
    lines = [
        "# a_m 范数 Top-32 区域的 I 逐层曲线", "",
        "每个目标、每层独立选取 `||a_lm||_2` 最大的 32 个视觉 token，再将其范数求和。曲线为目标 token 均值，阴影为 ±1 SEM；标签 0=幻觉，1=非幻觉。", "",
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
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
