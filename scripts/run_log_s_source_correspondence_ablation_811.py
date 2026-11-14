#!/usr/bin/env python3
"""Compare aligned SAR with a source-independent SAR on seeded COCO 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_source_correspondence_ablation_811"
REFERENCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
CASE = "sar_no_source_correspondence_ems"
METRICS = (
    "auroc", "real_aupr", "hall_aupr", "real_f1", "real_recall",
    "hall_f1", "hall_recall", "accuracy_real_threshold", "accuracy_hall_threshold",
)


def prepare(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    semantic_path = ROOT / "outputs/norm_semantic_transform_curves_by_label" / model / "compact_features.npz"
    mechanism_path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    with np.load(semantic_path, allow_pickle=False) as semantic, np.load(mechanism_path, allow_pickle=False) as mechanism:
        image_ids = np.asarray(mechanism["image_id"], dtype=np.int64)
        labels = np.asarray(mechanism["label"], dtype=np.int8)
        if not np.array_equal(image_ids, semantic["image_id"]) or not np.array_equal(labels, semantic["label"]):
            raise ValueError(f"{model}: semantic and mechanism rows differ")
        sar = np.asarray(mechanism["sar"], dtype=np.float32)
        ems = np.asarray(mechanism["ems"], dtype=np.float32)
        original = np.asarray(mechanism["sar_ems"], dtype=np.float32)
        i_sum = np.asarray(mechanism["i_ems"][:, :sar.shape[1]], dtype=np.float32)
        semantic_sar = np.asarray(semantic["log_residual_fusion"], dtype=np.float32)
        semantic_sum = np.asarray(semantic["log_support_semantic_sum"], dtype=np.float64)
        visual_tokens = np.asarray(semantic["visual_tokens"], dtype=np.int32)
        if np.any(visual_tokens <= 0):
            raise ValueError(f"{model}: invalid visual token count")
        if not np.array_equal(sar, semantic_sar) or not np.array_equal(original, np.concatenate((sar, ems), axis=1)):
            raise ValueError(f"{model}: exact SAR+EMS source anchor failed")
        mean_s = semantic_sum / visual_tokens[:, None]
        no_correspondence = (i_sum.astype(np.float64) * (1.0 + mean_s)).astype(np.float32)
        matrix = np.concatenate((no_correspondence, ems), axis=1)
        if matrix.shape != original.shape or not np.isfinite(matrix).all():
            raise ValueError(f"{model}: invalid ablation matrix")
        delta = sar.astype(np.float64) - no_correspondence.astype(np.float64)
        audit = {
            "model": model,
            "samples": len(labels),
            "layers": sar.shape[1],
            "feature_order": ["I_times_one_plus_mean_log_support", "EMS"],
            "no_correspondence_formula": "I_l * (1 + mean_m log(1 + C*p_lm))",
            "original_formula": "sum_m I_lm * (1 + log(1 + C*p_lm))",
            "semantic_probability": "P(target token | final_norm(hpre_visual))",
            "difference_formula": "sum_m I_lm*s_lm - I_l*mean_m(s_lm) = M*Cov_m(I_lm,s_lm)",
            "original_sar_anchor": "exact array equality",
            "original_sar_ems_anchor": "exact array equality",
            "visual_token_count_min": int(visual_tokens.min()),
            "visual_token_count_max": int(visual_tokens.max()),
            "difference_mean": float(delta.mean()),
            "difference_mean_absolute": float(np.abs(delta).mean()),
            "difference_max_absolute": float(np.abs(delta).max()),
            "semantic_source": str(semantic_path),
            "mechanism_source": str(mechanism_path),
        }
    output = OUT / model
    output.mkdir(parents=True, exist_ok=True)
    protocol.atomic_json(output / "feature_audit.json", audit)
    return matrix, labels, image_ids, audit


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    protocol.DISPLAY[CASE] = "EMS + SAR without source correspondence"
    matrix, labels, image_ids, _ = prepare(model)
    for seed in SEEDS:
        original_path = REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json"
        split_path = REFERENCE / "splits" / f"seed{seed}.json"
        original = json.loads(original_path.read_text())
        split_list = json.loads(split_path.read_text())
        split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
        if original["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
            raise ValueError(f"{model}/seed{seed}: original image counts differ")
        if len(set.union(*split.values())) != 4000 or any(split[a] & split[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"seed{seed}: split is invalid")
        if not set(image_ids.tolist()).issubset(set.union(*split.values())):
            raise ValueError(f"{model}/seed{seed}: feature images outside split")
        token_counts = {name: int(np.isin(image_ids, list(ids)).sum()) for name, ids in split.items()}
        if original["token_counts"] != token_counts:
            raise ValueError(f"{model}/seed{seed}: original token cohort differs")
        folder = OUT / model / f"seed{seed}" / CASE
        path = folder / "result.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["token_counts"] != token_counts:
                raise ValueError(f"{model}/seed{seed}: saved token cohort differs")
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
        "# COCO 8:1:1：SAR 视觉来源对应关系消融", "",
        "比较 LLaVA-1.5-7B 与 Qwen3-VL-8B；4000 张图像，种子 43/44/45 同时控制 3200/400/400 图片划分与 MLP 初始化。每个种子的两组使用完全相同的 object token、标签、EMS、K=32、τ=0.03 与训练器；仅替换 SAR。",
        "原版 SAR：$F_l=I_l+B_l=\\sum_m I_{l,m}(1+s_{l,m})$；去对应版：$F_l^{\\mathrm{ind}}=I_l(1+\\bar s_l)$，其中 $s_{l,m}=\\log(1+C p_{l,m})$，$\\bar s_l=M^{-1}\\sum_m s_{l,m}$，$p_{l,m}=P(y_t\\mid\\mathrm{final\\_norm}(h_{l,m}^{\\mathrm{pre}}))$。两组都与同一 EMS 逐层拼接。差值 $F_l-F_l^{\\mathrm{ind}}=M\\operatorname{Cov}_m(I_{l,m},s_{l,m})$。",
        "输入仅用训练集统计量标准化；单隐藏层宽度 128、dropout 0.3、batch 256、Adam 学习率 0.001、weight decay $10^{-5}$，最小验证损失选 checkpoint。Real/Hall 阈值分别由验证集 F1 选择；测试集不参与选择。数值为测试集百分数，± 为三种子的总体标准差。", "",
        "| 模型 | 方法 | AUROC | Real AUPR | Hall AUPR | Real F1 | Real Recall | Hall F1 | Hall Recall | Accuracy (Real threshold) | Accuracy (Hall threshold) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        by_case = {}
        for case, name in (("sar_ems", "EMS + SAR (aligned)"), (CASE, "EMS + SAR (source independent)")):
            measures = []
            for seed in SEEDS:
                path = (REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json" if case == "sar_ems" else OUT / model / f"seed{seed}" / CASE / "result.json")
                result = json.loads(path.read_text())
                values = protocol.flatten(result)
                rows.append({"model": model, "case": case, "seed": seed, **values})
                measures.append(values)
            by_case[case] = measures
            fields = [f"{100*np.mean([item[key] for item in measures]):.2f} ± {100*np.std([item[key] for item in measures]):.2f}" for key in METRICS]
            lines.append(f"| {model} | {name} | {' | '.join(fields)} |")
        delta = [f"{100*np.mean([a[key]-b[key] for a,b in zip(by_case['sar_ems'],by_case[CASE])]):+.2f}" for key in METRICS]
        lines.append(f"| {model} | Δ aligned − independent (pp) | {' | '.join(delta)} |")
    lines += ["", "## 逐种子测试结果", "", "| 模型 | 方法 | 种子 | " + " | ".join(METRICS) + " |", "|---|---|---:|" + "---:|"*len(METRICS)]
    for row in rows:
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | " + " | ".join(f"{100*row[k]:.3f}" for k in METRICS) + " |")
    lines += ["", "原始逐种子模型、阈值与测试分数见各目录的 `result.json`、`scores.npz`；特征同源核验见各模型的 `feature_audit.json`。", ""]
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
