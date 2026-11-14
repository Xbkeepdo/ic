#!/usr/bin/env python3
"""Train LLaVA Log-S SAR+EMS probes on selected decoder-layer ranges."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as original


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/log_s_mechanism_ablation_811/llava_1_5_7b"
RANGE_SETS = {
    "original": ((1, 8), (9, 16), (16, 24), (24, 32)),
    "nonoverlap": ((1, 8), (9, 18), (19, 27), (28, 32)),
}
SEEDS = (43, 44, 45)
METRICS = (
    ("AUROC", ("test", "auroc")),
    ("Real F1", ("test", "real", "f1")),
    ("Real Recall", ("test", "real", "recall")),
    ("Real AUPR", ("test", "real", "aupr")),
    ("Hall F1", ("test", "hall", "f1")),
    ("Hall Recall", ("test", "hall", "recall")),
    ("Hall AUPR", ("test", "hall", "aupr")),
)


def metric(result: dict, keys: tuple[str, ...]) -> float:
    value = result
    for key in keys:
        value = value[key]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--range-set", choices=RANGE_SETS, default="original")
    args = parser.parse_args()
    ranges = RANGE_SETS[args.range_set]
    out_name = ("log_s_layer_range_ablation_nonoverlap_811"
                if args.range_set == "nonoverlap" else "log_s_layer_range_ablation_811")
    out = ROOT / "outputs" / out_name / "llava_1_5_7b"

    with np.load(SOURCE / "compact_features.npz", allow_pickle=False) as saved:
        sar = np.asarray(saved["sar"], dtype=np.float32)
        ems = np.asarray(saved["ems"], dtype=np.float32)
        full = np.asarray(saved["sar_ems"], dtype=np.float32)
        labels = np.asarray(saved["label"], dtype=np.int64)
        image_ids = np.asarray(saved["image_id"], dtype=np.int64)
    if sar.shape != ems.shape or sar.ndim != 2 or sar.shape[1] != 32:
        raise ValueError(f"Expected matching 32-layer SAR/EMS, got {sar.shape}/{ems.shape}")
    if not np.array_equal(full, np.concatenate((sar, ems), axis=1)):
        raise ValueError("Full reference is not SAR concatenated with EMS")
    if not (np.isfinite(sar).all() and np.isfinite(ems).all()):
        raise ValueError("Feature cache contains non-finite values")

    split = original.load_split()
    device = original.resolve_device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    original.atomic_json(out / "experiment.json", {
        "model": "llava_1_5_7b",
        "features": ["SAR", "EMS"],
        "source": str(SOURCE / "compact_features.npz"),
        "layer_numbering": "1-based, inclusive",
        "ranges": [list(bounds) for bounds in ranges],
        "overlapping_layers": [] if args.range_set == "nonoverlap" else [16, 24],
        "top_k": 32,
        "tau": 0.03,
        "split": str(original.SPLIT_PATH),
        "image_counts": {name: len(ids) for name, ids in split.items()},
        "seeds": list(SEEDS),
        "reference_results": str(SOURCE / "sar_ems"),
        "sample_count": len(labels),
    })

    rows = []
    for start, end in ranges:
        case = f"layers_{start:02d}_{end:02d}"
        matrix = np.concatenate((sar[:, start - 1:end], ems[:, start - 1:end]), axis=1)
        original.DISPLAY[case] = f"SAR + EMS, layers {start}–{end}"
        for seed in SEEDS:
            result_path = out / case / f"seed{seed}" / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                print(f"Training {case} seed {seed} on {device}", flush=True)
                result = original.train_case(
                    model="llava_1_5_7b", case=case, matrix=matrix,
                    labels=labels, image_ids=image_ids, split=split,
                    seed=seed, device=device, output_dir=result_path.parent,
                )
            row = {"case": case, "layers": f"{start}–{end}", "seed": seed,
                   "input_dim": matrix.shape[1]}
            row.update({name: metric(result, keys) for name, keys in METRICS})
            rows.append(row)
        print(f"Completed {case}", flush=True)

    for seed in SEEDS:
        result = json.loads((SOURCE / "sar_ems" / f"seed{seed}" / "result.json").read_text())
        row = {"case": "full_reference", "layers": "1–32", "seed": seed,
               "input_dim": 64}
        row.update({name: metric(result, keys) for name, keys in METRICS})
        rows.append(row)

    with (out / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# LLaVA Log-S 层段消融（8:1:1）", "",
        "EMS 和 SAR 均截取相同层段，然后拼接；K=32，τ=0.03。层号从 1 开始且包含端点。" + (
            "四段互不重合，共覆盖全部 32 层。" if args.range_set == "nonoverlap"
            else "第 16、24 层在相邻层段重复。"),
        "使用同一批 3200/400/400 图片及 object token；单隐藏层 128、输入按训练集标准化、无 BatchNorm、dropout 0.3、batch 256、Adam LR 1e-3、weight decay 1e-5、验证损失早停 patience 10。",
        "Real/Hall 阈值分别仅从验证集选取；以下为固定测试集三随机种 43/44/45 的均值 ± 总体标准差，单位为百分比。全层行复用原实验结果。", "",
        "| 层段 | 输入维度 | " + " | ".join(name for name, _ in METRICS) + " |",
        "|---|---:|" + "|".join("---:" for _ in METRICS) + "|",
    ]
    for case in [*(f"layers_{start:02d}_{end:02d}" for start, end in ranges), "full_reference"]:
        group = [row for row in rows if row["case"] == case]
        vals = []
        for name, _ in METRICS:
            arr = np.asarray([row[name] for row in group], dtype=np.float64)
            vals.append(f"{100*arr.mean():.2f} ± {100*arr.std():.2f}")
        lines.append(f"| {group[0]['layers']} | {group[0]['input_dim']} | " + " | ".join(vals) + " |")
    lines.extend(["", "逐随机种数值见 `per_seed_results.csv`。", ""])
    (out / "complete_results.md").write_text("\n".join(lines))
    print(out / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
