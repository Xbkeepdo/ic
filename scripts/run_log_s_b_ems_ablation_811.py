#!/usr/bin/env python3
"""Train B+EMS and compare it with SAR+EMS on seeded COCO 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts.run_log_s_source_correspondence_ablation_811 import METRICS, MODELS, REFERENCE, SEEDS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_b_ems_ablation_811"
CASE = "b_ems"


def prepare(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mechanism_path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    semantic_path = ROOT / "outputs/norm_semantic_transform_curves_by_label" / model / "compact_features.npz"
    with np.load(mechanism_path, allow_pickle=False) as source, np.load(semantic_path, allow_pickle=False) as semantic:
        sar = np.asarray(source["sar"], dtype=np.float32)
        ems = np.asarray(source["ems"], dtype=np.float32)
        i_sum = np.asarray(source["i_ems"][:, :sar.shape[1]], dtype=np.float32)
        labels = np.asarray(source["label"], dtype=np.int8)
        image_ids = np.asarray(source["image_id"], dtype=np.int64)
        if not (np.array_equal(labels, semantic["label"]) and
                np.array_equal(image_ids, semantic["image_id"]) and
                np.array_equal(sar, semantic["log_residual_fusion"]) and
                np.array_equal(source["sar_ems"], np.concatenate((sar, ems), axis=1))):
            raise ValueError(f"{model}: original SAR/EMS and normalized-semantic cache differ")
        b = (sar.astype(np.float64) - i_sum.astype(np.float64)).astype(np.float32)
        if b.shape != ems.shape or not np.isfinite(b).all() or np.any(b < -1e-5):
            raise ValueError(f"{model}: invalid B = SAR - I")
        matrix = np.concatenate((b, ems), axis=1)
        audit = {
            "model": model,
            "samples": len(labels),
            "layers": sar.shape[1],
            "formula": "B_l = sum_m I_lm*log(1+C*p_lm) = SAR_l-I_l",
            "input_order": ["B", "EMS"],
            "reference_order": ["SAR", "EMS"],
            "sar_norm_semantic_anchor": "exact array equality",
            "ems_reference_anchor": "exact array equality",
            "b_min": float(b.min()),
            "b_max": float(b.max()),
            "mechanism_source": str(mechanism_path),
            "semantic_source": str(semantic_path),
        }
    (OUT / model).mkdir(parents=True, exist_ok=True)
    protocol.atomic_json(OUT / model / "feature_audit.json", audit)
    return matrix, labels, image_ids


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    protocol.DISPLAY[CASE] = "B + EMS"
    matrix, labels, image_ids = prepare(model)
    for seed in SEEDS:
        reference = json.loads((REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json").read_text())
        split_list = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
        if reference["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
            raise ValueError(f"{model}/seed{seed}: reference image counts differ")
        if any(split[a] & split[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"{model}/seed{seed}: split overlaps")
        token_counts = {part: int(np.isin(image_ids, list(ids)).sum()) for part, ids in split.items()}
        if token_counts != reference["token_counts"] or sum(token_counts.values()) != len(labels):
            raise ValueError(f"{model}/seed{seed}: reference object-token cohort differs")
        folder = OUT / model / f"seed{seed}" / CASE
        if (folder / "result.json").exists():
            saved = json.loads((folder / "result.json").read_text())
            if saved["token_counts"] != token_counts:
                raise ValueError(f"{model}/seed{seed}: saved cohort differs")
            print(f"[{model}] reuse seed{seed}", flush=True)
            continue
        protocol.train_case(
            model=model, case=CASE, matrix=matrix, labels=labels,
            image_ids=image_ids, split=split, seed=seed, device=device,
            output_dir=folder, split_protocol="seeded_image_level_811",
        )
        print(f"[{model}] completed seed{seed}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO 8:1:1：B + EMS 消融", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B，4000 张图像；种子 43/44/45 同时控制 3200/400/400 图片划分与检测器初始化。",
        "原版 SAR=I+B，B_l=∑ₘ I_{l,m}log(1+C p_{l,m})。仅将 [SAR,EMS] 中的 SAR 替换为 B；EMS 使用同一 K=32、τ=0.03 特征。训练集标准化、单隐藏层宽度 128、dropout 0.3、batch 256、Adam 学习率 0.001、weight decay 1e-5、验证损失选 checkpoint。Real/Hall F1 阈值分别由验证集选取。指标是测试集百分数，± 是三个种子的总体标准差。", "",
        "| 模型 | 特征 | AUROC | Real AUPR | Hall AUPR | Real F1 | Real Recall | Hall F1 | Hall Recall | Accuracy (Real threshold) | Accuracy (Hall threshold) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        grouped = {}
        for variant, display in (("sar_ems", "SAR + EMS"), (CASE, "B + EMS")):
            values = []
            for seed in SEEDS:
                path = (REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json" if variant == "sar_ems" else OUT / model / f"seed{seed}" / CASE / "result.json")
                saved = json.loads(path.read_text())
                row = protocol.flatten(saved)
                values.append(row)
                rows.append({"model": model, "variant": variant, "seed": seed, **row})
            grouped[variant] = values
            formatted = " | ".join(f"{100*np.mean([v[k] for v in values]):.2f} ± {100*np.std([v[k] for v in values]):.2f}" for k in METRICS)
            lines.append(f"| {model} | {display} | {formatted} |")
        delta = " | ".join(f"{100*np.mean([a[k]-b[k] for a,b in zip(grouped['sar_ems'],grouped[CASE])]):+.2f}" for k in METRICS)
        lines.append(f"| {model} | Δ SAR+EMS − B+EMS (pp) | {delta} |")
    lines += ["", "## 逐种子测试指标", "", "| 模型 | 特征 | 种子 | " + " | ".join(METRICS) + " |", "|---|---|---:|" + "---:|"*len(METRICS)]
    for row in rows:
        lines.append(f"| {row['model']} | {row['variant']} | {row['seed']} | " + " | ".join(f"{100*row[k]:.3f}" for k in METRICS) + " |")
    lines += ["", "原始逐种子结果与测试分数见各模型目录的 `result.json`、`scores.npz`；特征核验见 `feature_audit.json`。", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model required unless --summarize")


if __name__ == "__main__":
    main()
