#!/usr/bin/env python3
"""Compare aligned and source-independent SAR+EMS for Qwen3 on seeded COCO 8:2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_log_s_source_correspondence_ablation_811 import prepare
from scripts.train_log_s_hpre_k_tau_ablation import _fit
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
OUT = ROOT / "outputs/log_s_source_correspondence_ablation_82"
MODEL = "qwen3_vl_8b"
SEEDS = (43, 44, 45)


def result_path(seed: int, variant: str) -> Path:
    if variant == "aligned":
        return REFERENCE / MODEL / f"seed{seed}" / "ours" / "metrics.json"
    return OUT / MODEL / f"seed{seed}" / "source_independent" / "metrics.json"


def metrics(path: Path) -> dict[str, float]:
    test = json.loads(path.read_text())["threshold_reports"]["train_f1"]["test_metrics"]
    return {
        "auroc": float(test["real_positive"]["auc"]),
        "real_aupr": float(test["real_positive"]["aupr"]),
        "hall_aupr": float(test["hallucination_positive"]["aupr"]),
        "real_f1": float(test["real_positive"]["f1"]),
        "hall_f1": float(test["hallucination_positive"]["f1"]),
        "accuracy": float(test["accuracy"]),
    }


def run(device: str) -> None:
    torch.set_num_threads(1)
    matrix_sar_ems, labels, images, feature_audit = prepare(MODEL)
    layers = matrix_sar_ems.shape[1] // 2
    # Match the exact original 8:2 input order: EMS, then SAR.
    matrix = np.concatenate((matrix_sar_ems[:, layers:], matrix_sar_ems[:, :layers]), axis=1)
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    audit = {
        **feature_audit,
        "split": "seeded image-level 3200/800",
        "seeds": list(SEEDS),
        "input_order": ["EMS", "I_times_one_plus_mean_log_support"],
        "reference": str(REFERENCE),
    }
    (OUT / MODEL).mkdir(parents=True, exist_ok=True)
    (OUT / MODEL / "feature_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for seed in SEEDS:
        split = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        train_images, test_images = set(split["train"]), set(split["test"])
        if len(train_images) != 3200 or len(test_images) != 800 or train_images & test_images:
            raise ValueError(f"seed{seed}: invalid image split")
        fit_mask = np.isin(images, list(train_images))
        eval_mask = np.isin(images, list(test_images))
        if not np.all(fit_mask | eval_mask) or np.any(fit_mask & eval_mask):
            raise ValueError(f"seed{seed}: incomplete object-token cohort")
        if not result_path(seed, "aligned").exists():
            raise FileNotFoundError(result_path(seed, "aligned"))
        folder = result_path(seed, "source_independent").parent
        _fit(
            matrix=matrix, labels=labels, fit_mask=fit_mask, eval_mask=eval_mask,
            seed=seed, root_config=config, device=torch.device(device), folder=folder,
        )
        print(f"[{MODEL}] seed{seed}: complete", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# Qwen3-VL-8B：COCO 8:2 的 SAR 来源对应关系消融", "",
        "4000 张图片；种子 43/44/45 同时控制图片划分和 MLP 初始化，每次 3200 train / 800 test。原版与去对应版使用相同 object token、标签、EMS（K=32，τ=0.03）及检测器配置。原版输入顺序为 [EMS, SAR]；去对应版为 [EMS, I(1+mean(s))]。",
        "SAR=∑ₘ Iₘ(1+sₘ)，sₘ=log(1+C·pₘ)，pₘ 用原版相同的 final-norm 语义概率。严格 8:2 协议没有独立验证集；检测器用最小训练损失 checkpoint，F1 阈值由训练集选取。AUROC、AUPR 与阈值无关。数值为测试集百分数，± 是三个种子的总体标准差。", "",
        "| 方法 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped = {}
    names = {"aligned": "EMS + SAR（原版）", "source_independent": "EMS + I(1+mean(s))"}
    fields = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")
    for variant in names:
        values = []
        for seed in SEEDS:
            v = metrics(result_path(seed, variant))
            values.append(v)
            rows.append({"model": MODEL, "variant": variant, "seed": seed, **v})
        grouped[variant] = values
        summary = " | ".join(f"{100*np.mean([v[k] for v in values]):.2f} ± {100*np.std([v[k] for v in values]):.2f}" for k in fields)
        lines.append(f"| {names[variant]} | {summary} |")
    delta = " | ".join(f"{100*np.mean([a[k]-b[k] for a,b in zip(grouped['aligned'],grouped['source_independent'])]):+.2f}" for k in fields)
    lines.append(f"| Δ 原版 − 去对应（百分点） | {delta} |")
    lines += ["", "## 逐种子", "", "| 方法 | 种子 | " + " | ".join(fields) + " |", "|---|---:|" + "---:|"*len(fields)]
    for row in rows:
        lines.append(f"| {names[row['variant']]} | {row['seed']} | " + " | ".join(f"{100*row[k]:.3f}" for k in fields) + " |")
    lines += ["", "原始逐种子指标见新方法各目录的 `metrics.json` 和原版对照的 `outputs/coco4000_seeded_image_split_auc_aupr/qwen3_vl_8b/seed*/ours/metrics.json`。", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.device)


if __name__ == "__main__":
    main()
