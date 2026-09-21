#!/usr/bin/env python3
"""Plot the four COCO4000-512 ICLR risk curves by CHAIR label."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import MODELS, RISK_KEYS  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402

OUT = ROOT / "outputs/iclr_512_risk_curves_by_label"
TITLES = {
    "raw_semantic": "Raw logits · semantic cost",
    "raw_target_penalty": "Raw logits · + target penalty",
    "softmax_semantic": "Softmax probability · semantic cost",
    "softmax_target_penalty": "Softmax probability · + target penalty",
}
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def load_model(model: str) -> tuple[dict[str, np.ndarray], dict]:
    audit_path = ROOT / "outputs/iclr_512_endac_attention_i_fusion" / model / "alignment.json"
    audit = json.loads(audit_path.read_text())
    if (audit["matched_rows"] != audit["iclr_rows"]
            or audit["unmatched_gf_rows"] != 0
            or audit["rejected_prefix_rows"] != 0
            or audit["identical_full_generation_images"] != 4000):
        raise ValueError(f"{model}: alignment is not complete")
    rows = load_pkl(audit["iclr_source"])
    if len(rows) != audit["matched_rows"]:
        raise ValueError(f"{model}: feature row count differs from alignment audit")
    data = {
        "label": np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        "image_id": np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64),
        "position": np.asarray([int(row["response_token_idx"]) for row in rows], dtype=np.int32),
    }
    for name, key in RISK_KEYS.items():
        data[name] = np.stack([np.asarray(row[key], dtype=np.float32) for row in rows])
    del rows
    gc.collect()
    if set(np.unique(data["label"])) != {0, 1}:
        raise ValueError(f"{model}: expected both CHAIR labels")
    layers = audit["layers"]
    if any(data[name].shape != (len(data["label"]), layers) or
           not np.isfinite(data[name]).all() for name in RISK_KEYS):
        raise ValueError(f"{model}: risk shape or finite check failed")
    run = ROOT / "outputs" / model / audit["risk_run_name"]
    split = json.loads((run / "image_splits.json").read_text())
    test_ids = np.asarray(split["test"], dtype=np.int64)
    data["test"] = np.isin(data["image_id"], test_ids)
    if int(data["test"].sum()) != audit["matched_test_rows"]:
        raise ValueError(f"{model}: test row count differs from alignment audit")
    return data, audit


def summarize(data: dict[str, np.ndarray], selection: np.ndarray, bin_width: int) -> dict:
    labels = data["label"][selection]
    bins = data["position"][selection] // bin_width
    hall = labels == 0
    real = labels == 1
    if not hall.any() or not real.any():
        raise ValueError("Both labels are required")
    common = []
    weights = []
    for bin_id in np.unique(bins):
        h = int(np.count_nonzero(hall & (bins == bin_id)))
        r = int(np.count_nonzero(real & (bins == bin_id)))
        if h >= 5 and r >= 5:
            common.append(int(bin_id))
            weights.append(min(h, r))
    if not common:
        raise ValueError("No shared generation-position bins")
    result = {
        "hall_count": int(hall.sum()),
        "real_count": int(real.sum()),
        "common_bins": len(common),
        "common_weight": int(sum(weights)),
        "curves": {},
    }
    for name in RISK_KEYS:
        values = data[name][selection].astype(np.float64, copy=False)
        raw_hall = values[hall].mean(axis=0)
        raw_real = values[real].mean(axis=0)
        balanced = []
        for class_mask in (hall, real):
            by_bin = np.stack([
                values[class_mask & (bins == bin_id)].mean(axis=0)
                for bin_id in common
            ])
            balanced.append(np.average(by_bin, axis=0, weights=weights))
        result["curves"][name] = {
            "hall": raw_hall,
            "real": raw_real,
            "hall_balanced": balanced[0],
            "real_balanced": balanced[1],
        }
    return result


def plot(model: str, split: str, stats: dict, folder: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    for axis, name in zip(axes.flat, RISK_KEYS):
        curves = stats["curves"][name]
        layers = np.arange(1, len(curves["hall"]) + 1)
        for label, color, field in (("Hallucination", "#d55e00", "hall"),
                                    ("Non-hallucination", "#0072b2", "real")):
            axis.plot(layers, curves[field], color=color, lw=2, label=label)
            axis.plot(layers, curves[f"{field}_balanced"], color=color,
                      lw=1.7, ls="--", alpha=.85)
        axis.set_title(TITLES[name])
        axis.set_xlabel("Decoder layer (1-based)")
        axis.set_ylabel("Mean transport risk")
        axis.grid(alpha=.23)
        axis.set_xlim(1, int(layers[-1]))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.04))
    fig.suptitle(f"{DISPLAY[model]} · {split} · HALL {stats['hall_count']:,} / REAL {stats['real_count']:,}\n"
                 "Solid: mention mean    Dashed: generation-position-balanced mean", y=1.13)
    fig.savefig(folder / f"{split}.png", dpi=220, bbox_inches="tight")
    fig.savefig(folder / f"{split}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_curves(model: str, split: str, stats: dict, folder: Path) -> list[dict]:
    output = []
    with (folder / f"{split}.csv").open("w", newline="") as handle:
        fields = ("risk", "layer", "hall_mean", "real_mean", "hall_minus_real",
                  "hall_position_balanced", "real_position_balanced", "balanced_gap")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, curves in stats["curves"].items():
            gap = curves["hall"] - curves["real"]
            balanced_gap = curves["hall_balanced"] - curves["real_balanced"]
            peak = int(np.argmax(np.abs(gap)))
            output.append({
                "model": model, "split": split, "risk": name,
                "hall_count": stats["hall_count"], "real_count": stats["real_count"],
                "common_bins": stats["common_bins"], "common_weight": stats["common_weight"],
                "hall_mean": float(curves["hall"].mean()),
                "real_mean": float(curves["real"].mean()),
                "hall_minus_real": float(gap.mean()),
                "balanced_gap": float(balanced_gap.mean()),
                "peak_gap_layer": peak + 1,
                "peak_gap": float(gap[peak]),
            })
            for i in range(len(gap)):
                writer.writerow({
                    "risk": name, "layer": i + 1,
                    "hall_mean": curves["hall"][i], "real_mean": curves["real"][i],
                    "hall_minus_real": gap[i],
                    "hall_position_balanced": curves["hall_balanced"][i],
                    "real_position_balanced": curves["real_balanced"][i],
                    "balanced_gap": balanced_gap[i],
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--position-bin-width", type=int, default=32)
    args = parser.parse_args()
    if args.position_bin_width < 1:
        parser.error("--position-bin-width must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in args.models:
        data, audit = load_model(model)
        folder = args.output_dir / model
        folder.mkdir(parents=True, exist_ok=True)
        for split, mask in (("all", np.ones(len(data["label"]), dtype=bool)),
                            ("test", data["test"])):
            stats = summarize(data, mask, args.position_bin_width)
            plot(model, split, stats, folder)
            summaries.extend(save_curves(model, split, stats, folder))
        del data
        gc.collect()
        print(f"DONE {model}: {audit['matched_rows']} matched rows", flush=True)

    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# COCO4000-512 ICLR risk：幻觉与非幻觉逐层曲线", "",
        "四模型、四种 risk：raw/softmax-prob × 纯语义 cost/加目标惩罚 cost。",
        "标签：0=幻觉（HALL），1=非幻觉（REAL）。实线是逐目标均值；虚线是生成位置均衡后的均值：每32个 response token 分箱，只保留两类各至少5个目标的箱，各箱权重取两类较小样本数。",
        "`all` 包含全部完全匹配目标；`test` 只含图像级 800 张测试图像中的目标。风险值为原始传输代价，未作跨模型或跨分支标准化；不同图的纵轴不宜直接比较。虚线仅控制粗粒度生成位置，属于描述性统计。", "",
        "| 模型 | 样本 | HALL / REAL | 共同位置箱 | 共同箱权重 | 曲线 | 数值 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for model in args.models:
        for split in ("all", "test"):
            one = next(row for row in summaries if row["model"] == model and row["split"] == split)
            lines.append(f"| {DISPLAY[model]} | {split} | {one['hall_count']} / {one['real_count']} | "
                         f"{one['common_bins']} | {one['common_weight']} | "
                         f"[PNG]({model}/{split}.png) · [PDF]({model}/{split}.pdf) | "
                         f"[CSV]({model}/{split}.csv) |")
    lines += ["", "## 各层平均差异", "",
              "差值均为 HALL − REAL；正值表示幻觉目标的平均 risk 更高。", "",
              "| 模型 | 样本 | Risk | HALL | REAL | 原始差值 | 位置均衡差值 | 差异最大层 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {DISPLAY[row['model']]} | {row['split']} | {row['risk']} | "
                     f"{row['hall_mean']:.5f} | {row['real_mean']:.5f} | "
                     f"{row['hall_minus_real']:+.5f} | {row['balanced_gap']:+.5f} | "
                     f"{row['peak_gap_layer']} |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(args.output_dir / "summary.md", flush=True)


if __name__ == "__main__":
    main()
