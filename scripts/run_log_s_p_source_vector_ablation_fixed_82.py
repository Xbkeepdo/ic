#!/usr/bin/env python3
"""Compare four EMS P source vectors on the original fixed COCO 8:2 split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_log_s_p_source_vector_ablation_811 import DISPLAY, MODES, load_base, load_new
from scripts.run_log_s_p_source_vector_ablation_82 import metrics
from scripts.train_log_s_hpre_k_tau_ablation import _fit
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/log_s_hpre_k_tau_ablation"
OUT = ROOT / "outputs/log_s_p_source_vector_ablation_82_fixed_split"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
FIELDS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def result_path(model: str, mode: str, seed: int) -> Path:
    return OUT / model / mode / f"seed{seed}" / "metrics.json"


def run_model(model: str, device_name: str) -> None:
    torch.set_num_threads(1)
    base = load_base(model)
    new = load_new(model, base)
    with np.load(OLD / model / "compact_features.npz", allow_pickle=False) as saved:
        old = {key: np.asarray(saved[key]) for key in (
            "image_id", "response_token_idx", "target_token_id", "label",
            "log_s_hpre_k032_tau003", "f_log", "train_mask", "strict_test_mask",
        )}
    same_metadata = all(np.array_equal(base[key], old[key]) for key in (
        "image_id", "response_token_idx", "target_token_id", "label",
    ))
    risk_error = float(np.max(np.abs(new["ffn_output"] - old["log_s_hpre_k032_tau003"])))
    sar_error = float(np.max(np.abs(base["sar"] - old["f_log"])))
    if not same_metadata or risk_error != 0.0 or sar_error != 0.0:
        raise ValueError(f"{model}: original fixed 8:2 features differ")
    fit_mask = old["train_mask"].astype(bool)
    eval_mask = old["strict_test_mask"].astype(bool)
    if np.any(fit_mask & eval_mask) or not np.all(fit_mask | eval_mask):
        raise ValueError(f"{model}: invalid fixed 8:2 masks")
    image_ids = old["image_id"]
    train_images = np.unique(image_ids[fit_mask])
    test_images = np.unique(image_ids[eval_mask])
    # The image cohort is 3200/800; some images have no labeled object token.
    if (len(train_images) > 3200 or len(test_images) > 800 or
            len(train_images) + len(test_images) != len(np.unique(image_ids)) or
            np.intersect1d(train_images, test_images).size):
        raise ValueError(f"{model}: invalid fixed image split")
    model_out = OUT / model
    model_out.mkdir(parents=True, exist_ok=True)
    audit = {
        "model": model, "samples": len(image_ids), "layers": base["sar"].shape[1],
        "image_cohort_counts": {"train": 3200, "test": 800},
        "images_with_object_tokens": {"train": len(train_images), "test": len(test_images)},
        "token_counts": {"train": int(fit_mask.sum()), "test": int(eval_mask.sum())},
        "same_labels_and_targets": same_metadata,
        "ffn_risk_max_abs_error": risk_error,
        "sar_max_abs_error": sar_error,
        "seeds": list(SEEDS),
        "input_order": ["EMS", "SAR"],
        "trainer": "original strict 8:2, no validation",
    }
    (model_out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for mode in MODES:
        matrix = np.concatenate((new[mode], base["sar"]), axis=1).astype(np.float32)
        for seed in SEEDS:
            path = result_path(model, mode, seed)
            _fit(matrix=matrix, labels=base["label"], fit_mask=fit_mask,
                 eval_mask=eval_mask, seed=seed, root_config=config,
                 device=torch.device(device_name), folder=path.parent)
            print(f"completed {model} {mode} seed{seed}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO4000 固定 8:2：EMS 源分布 P 构造向量比较", "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B；固定同一份图片级 3200/800 训练/测试划分，种子 43/44/45 沿用原版训练器，控制 MLP 初始化、样本顺序和 dropout。四组均在 CUDA 上训练，只改变 P 的构造向量；SAR、T、cost、K=32、tau=0.03、对象 token 及训练设置一致。输入顺序为 [EMS,SAR]，没有验证集；阈值在训练集按各正类 F1 选择。原版 u 的三个种子测试指标与历史 test_grid 逐项完全一致。指标单位为百分数，± 为总体标准差。", "",
        "| 模型 | P 构造向量 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy | ΔAUROC vs u (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        grouped = {}
        for mode in MODES:
            values = []
            for seed in SEEDS:
                row = {"model": model, "source_vector": mode, "seed": seed,
                       **metrics(result_path(model, mode, seed))}
                values.append(row)
                rows.append(row)
            grouped[mode] = values
            cells = " | ".join(
                f"{100*np.mean([v[field] for v in values]):.2f} ± "
                f"{100*np.std([v[field] for v in values]):.2f}"
                for field in FIELDS
            )
            delta = 100 * (np.mean([v["auroc"] for v in values]) -
                           np.mean([v["auroc"] for v in grouped["ffn_output"]]))
            lines.append(f"| {model} | {DISPLAY[mode]} | {cells} | {delta:+.2f} |")
    lines += ["", "## 逐种子结果", "",
              "| 模型 | P 构造向量 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {DISPLAY[row['source_vector']]} | {row['seed']} | " +
                     " | ".join(f"{100*row[field]:.3f}" for field in FIELDS) + " |")
    lines += ["", "各模型目录下 `audit.json` 记录特征和划分核对；逐种子 `metrics.json` 及 `config.json` 可复查。", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "complete_results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model or --summarize required")


if __name__ == "__main__":
    main()
