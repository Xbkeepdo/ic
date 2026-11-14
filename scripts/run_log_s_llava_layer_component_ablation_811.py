#!/usr/bin/env python3
"""LLaVA layer ablation for SAR, EMS, and their concatenation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as training


ROOT = Path(__file__).resolve().parents[1]
RANGES = ((1, 8), (9, 18), (19, 27), (28, 32))
GROUPS = ("sar", "ems", "sar_ems")
SEEDS = (43, 44, 45)
METRICS = {
    "auroc": ("test", "auroc"),
    "real_aupr": ("test", "real", "aupr"),
    "real_f1": ("test", "real", "f1"),
}


def get(result: dict, keys: tuple[str, ...]) -> float:
    value = result
    for key in keys:
        value = value[key]
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, choices=(32, 64), default=64)
    args = parser.parse_args()
    source_name = ("log_s_mechanism_ablation_811" if args.top_k == 32
                   else "log_s_mechanism_ablation_k064_811")
    source = ROOT / "outputs" / source_name / "llava_1_5_7b"
    output = ROOT / "outputs" / f"log_s_layer_component_ablation_k{args.top_k:03d}_811" / "llava_1_5_7b"
    with np.load(source / "compact_features.npz", allow_pickle=False) as cache:
        sar = np.asarray(cache["sar"], dtype=np.float32)
        ems = np.asarray(cache["ems"], dtype=np.float32)
        full = np.asarray(cache["sar_ems"], dtype=np.float32)
        labels = np.asarray(cache["label"], dtype=np.int64)
        image_ids = np.asarray(cache["image_id"], dtype=np.int64)
    if sar.shape != ems.shape or sar.ndim != 2 or sar.shape[1] != 32:
        raise ValueError(f"Expected 32 aligned layers, got SAR={sar.shape}, EMS={ems.shape}")
    if not np.array_equal(full, np.concatenate((sar, ems), axis=1)):
        raise ValueError("Full reference differs from SAR concatenated with EMS")
    if not np.isfinite(full).all():
        raise ValueError("Features contain non-finite values")
    if sum(end - start + 1 for start, end in RANGES) != 32 or any(
        RANGES[index][1] + 1 != RANGES[index + 1][0] for index in range(3)
    ):
        raise ValueError("Layer sets must partition all 32 decoder layers")

    split = training.load_split()
    device = training.resolve_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    training.atomic_json(output / "experiment.json", {
        "model": "llava_1_5_7b", "top_k": args.top_k, "tau": 0.03,
        "ranges_inclusive": [list(pair) for pair in RANGES],
        "groups": list(GROUPS), "seeds": list(SEEDS),
        "source": str(source / "compact_features.npz"),
        "split": str(training.SPLIT_PATH),
        "image_counts": {name: len(ids) for name, ids in split.items()},
        "sample_count": len(labels),
        "positive_class": "real",
        "threshold_selection": "maximize validation Real F1",
        "full_layer_reference": str(source),
    })

    results = []
    for group in GROUPS:
        for start, end in (*RANGES, (1, 32)):
            case = f"{group}_layers_{start:02d}_{end:02d}"
            if group == "sar":
                matrix = sar[:, start - 1:end]
            elif group == "ems":
                matrix = ems[:, start - 1:end]
            else:
                matrix = np.concatenate((sar[:, start - 1:end], ems[:, start - 1:end]), axis=1)
            training.DISPLAY[case] = f"{group.upper()}, layers {start}-{end}"
            for seed in SEEDS:
                if (start, end) == (1, 32):
                    path = source / group / f"seed{seed}" / "result.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    result = json.loads(path.read_text())
                else:
                    path = output / case / f"seed{seed}" / "result.json"
                    if path.exists():
                        result = json.loads(path.read_text())
                    else:
                        print(f"Training {case} seed {seed} on {device}", flush=True)
                        result = training.train_case(
                            model="llava_1_5_7b", case=case, matrix=matrix,
                            labels=labels, image_ids=image_ids, split=split,
                            seed=seed, device=device, output_dir=path.parent,
                        )
                row = {
                    "group": group, "layer_set": f"{start}-{end}",
                    "seed": seed, "input_dim": matrix.shape[1],
                    "source_result": str(path),
                }
                row.update({name: get(result, keys) for name, keys in METRICS.items()})
                results.append(row)
            print(f"Completed {case}", flush=True)

    with (output / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for group in GROUPS:
        for start, end in (*RANGES, (1, 32)):
            layer_set = f"{start}-{end}"
            subset = [r for r in results if r["group"] == group and r["layer_set"] == layer_set]
            summary[(group, layer_set)] = {
                name: (float(np.mean([r[name] for r in subset])),
                       float(np.std([r[name] for r in subset])))
                for name in METRICS
            }
    lines = [
        f"# LLaVA Log-S 层段与组成特征消融（K={args.top_k}）", "",
        "四个不重合层段为 1–8、9–18、19–27、28–32；SAR、EMS 各自截取这些层，拼接组把同一层段的两种特征连接。全层行复用同一缓存与协议下的原始消融结果。",
        "COCO ENDAC 3200/400/400 图片级划分；τ=0.03；seeds 43/44/45；单隐藏层 128、训练集 z-score、无 BatchNorm、dropout 0.3、batch 256、Adam LR 1e-3、weight decay 1e-5、验证损失早停 patience 10。",
        "Real 为正类。F1 阈值仅在验证集以 Real F1 最大化确定；AUC 即 AUROC，AUC/AUPR 不依赖阈值。下表是测试集均值 ± 总体标准差，单位为百分比。", "",
        "| 特征 | 层段 | 输入维度 | AUC | Real AUPR | Real F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for start, end in (*RANGES, (1, 32)):
            layer_set = f"{start}-{end}"
            stats = summary[(group, layer_set)]
            dimension = (end - start + 1) * (2 if group == "sar_ems" else 1)
            cells = [f"{100*stats[name][0]:.2f} ± {100*stats[name][1]:.2f}" for name in METRICS]
            display = "SAR+EMS" if group == "sar_ems" else group.upper()
            lines.append(f"| {display} | {layer_set} | {dimension} | " + " | ".join(cells) + " |")
    lines.extend(["", "逐随机种结果见 `per_seed_results.csv`；提取源、层段和划分见 `experiment.json`。", ""])
    (output / "complete_results.md").write_text("\n".join(lines))
    print(output / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
