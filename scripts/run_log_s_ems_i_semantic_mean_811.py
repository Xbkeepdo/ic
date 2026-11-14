#!/usr/bin/env python3
"""Compare summed and averaged semantic probability on the shared COCO 8:1:1 split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts.run_log_s_ems_i_semantic_mean_seeded_82 import load_features


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_ems_i_semantic_mean_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
CASES = ("ems_i_e_sum", "ems_i_e_mean")
NAMES = {
    "sar_ems": "EMS+SAR",
    "ems_i_e_sum": "EMS+I+E_sum",
    "ems_i_e_mean": "EMS+I+E_mean",
}
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def matrices(model: str, audit_dir: Path = OUT) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mean_matrix, labels, image_ids = load_features(model, audit_dir=audit_dir)
    layers = mean_matrix.shape[1] // 3
    semantic_path = ROOT / "outputs/norm_semantic_transform_curves_by_label" / model / "compact_features.npz"
    mechanism_path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    with np.load(semantic_path, allow_pickle=False) as source:
        semantic_sum = np.asarray(source["semantic_mass"], dtype=np.float32)
    with np.load(mechanism_path, allow_pickle=False) as source:
        if not (np.array_equal(labels, source["label"]) and
                np.array_equal(image_ids, source["image_id"]) and
                np.array_equal(mean_matrix[:, :layers], source["ems"])):
            raise ValueError(f"{model}: 8:1:1 mechanism cache differs from mean feature source")
        baseline_rows = len(source["sar_ems"])
    if len(labels) != baseline_rows or semantic_sum.shape != (len(labels), layers):
        raise ValueError(f"{model}: baseline/sum shape mismatch")
    sum_matrix = mean_matrix.copy()
    sum_matrix[:, -layers:] = semantic_sum
    return {"ems_i_e_sum": sum_matrix, "ems_i_e_mean": mean_matrix}, labels, image_ids


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    protocol.DISPLAY.update({case: NAMES[case] for case in CASES})
    matrix_by_case, labels, image_ids = matrices(model)
    split = protocol.load_split()
    if sum(len(ids) for ids in split.values()) != 4000:
        raise ValueError("expected fixed 3200/400/400 image split")
    if not set(image_ids.tolist()).issubset(set.union(*split.values())):
        raise ValueError(f"{model}: feature image IDs are outside the fixed split")
    for case in CASES:
        for seed in SEEDS:
            folder = OUT / model / case / f"seed{seed}"
            if (folder / "result.json").exists():
                print(f"[{model}] reuse {case} seed{seed}", flush=True)
                continue
            protocol.train_case(
                model=model, case=case, matrix=matrix_by_case[case], labels=labels,
                image_ids=image_ids, split=split, seed=seed, device=device,
                output_dir=folder,
            )
            print(f"[{model}] finished {case} seed{seed}", flush=True)


def result_path(model: str, case: str, seed: int) -> Path:
    if case == "sar_ems":
        return ROOT / "outputs/log_s_mechanism_ablation_811" / model / case / f"seed{seed}" / "result.json"
    return OUT / model / case / f"seed{seed}" / "result.json"


def summarize() -> None:
    records = []
    lines = [
        "# COCO 8:1:1：E 求和与平均的训练结果", "",
        "LLaVA-1.5-7B、Qwen3-VL-8B；固定共享图片划分 3200/400/400，随机种子 43/44/45 控制 MLP 初始化和训练顺序。",
        "EMS、I、object token 对齐和训练器一致；$E_l=\\sum_m p_{l,m}$，$\\bar E_l=E_l/M_l$。所有输入维度仅用训练集统计量标准化。",
        "采用单隐藏层 128、dropout 0.3、Adam 学习率 0.001、weight decay 1e-5、验证损失选 checkpoint；Real/Hall F1 的阈值分别由验证集选择。",
        "AUROC 和 AUPR 不依赖阈值。下表为测试集百分数，± 为三个种子的总体标准差。", "",
        "LLaVA 每个样本固定 576 个视觉 token，平均与求和在标准化后理论上等价；Qwen3 每个样本的视觉 token 数可变。", "",
        "| 模型 | 特征 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        grouped = {}
        for case in ("sar_ems", *CASES):
            values = []
            for seed in SEEDS:
                saved = json.loads(result_path(model, case, seed).read_text())
                if saved["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
                    raise ValueError(f"{model}/{case}/seed{seed}: image split count mismatch")
                row = protocol.flatten(saved)
                values.append(row)
                records.append({"model": model, "case": NAMES[case], "seed": seed, **row})
            grouped[case] = values
            display = " | ".join(
                f"{100*np.mean([v[key] for v in values]):.2f} ± {100*np.std([v[key] for v in values]):.2f}"
                for key in METRICS
            )
            lines.append(f"| {model} | {NAMES[case]} | {display} |")
        delta = " | ".join(
            f"{100*np.mean([a[key]-b[key] for a,b in zip(grouped['ems_i_e_mean'], grouped['ems_i_e_sum'])]):+.3f}"
            for key in METRICS
        )
        lines.append(f"| {model} | Δ mean − sum | {delta} |")
    lines.extend(["", "## 逐种子测试指标", "",
                  "| 模型 | 特征 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in records:
        values = " | ".join(f"{100*row[key]:.2f}" for key in METRICS)
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | {values} |")
    lines.extend(["", "原始逐种子结果见各模型的 `result.json`，特征校验见 `alignment.json`。", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
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
