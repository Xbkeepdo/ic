#!/usr/bin/env python3
"""Summarize validation width search and final held-out test results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_iclr_endac_attention_i_fusion import FUSIONS, MODELS, ROOT, SEEDS
from scripts.tune_iclr_512_softmax_target_penalty_width import OUT, RISK, WIDTHS

REFERENCE = ROOT / "outputs/iclr_512_endac_attention_i_fusion_standardized_no_bn/one_hidden_128"
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def baseline(model: str, fusion: str) -> dict:
    reports = []
    for seed in SEEDS:
        path = REFERENCE / model / RISK / fusion / f"seed{seed}" / "metrics.json"
        report = json.loads(path.read_text())["threshold_reports"]["train_f1"]["test_metrics"]
        reports.append({
            "real_f1": report["real_positive"]["f1"],
            "real_auc": report["real_positive"]["auc"],
            "hall_aupr": report["hallucination_positive"]["aupr"],
        })
    return {key: {"mean": float(np.mean([row[key] for row in reports])),
                  "std": float(np.std([row[key] for row in reports]))}
            for key in ("real_f1", "real_auc", "hall_aupr")}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    final_rows = []
    grid_rows = []
    split_sha = set()
    for model in MODELS:
        data = json.loads((OUT / model / "summary.json").read_text())
        if data["model"] != model or data["risk"] != RISK or data["candidate_widths"] != list(WIDTHS):
            raise ValueError(f"{model}: summary protocol mismatch")
        split = data["split"]
        if (split["fit_images"], split["validation_images"], split["test_images"]) != (2560, 640, 800):
            raise ValueError(f"{model}: image split counts changed")
        split_sha.add(split["source_train_image_sha256"])
        for fusion in FUSIONS:
            case = data["feature_sets"][fusion]
            if {item["width"] for item in case["candidates"]} != set(WIDTHS):
                raise ValueError(f"{model}/{fusion}: incomplete width grid")
            for item in case["candidates"]:
                grid_rows.append({
                    "model": model, "fusion": fusion, "width": item["width"],
                    "selected": item["width"] == case["selected_width"],
                    **{f"val_{key}_{stat}": item["metrics"][key][stat]
                       for key in ("real_f1", "real_auc", "hall_aupr")
                       for stat in ("mean", "std")},
                })
            selected = case["selected_width"]
            expected = max(case["candidates"], key=lambda row: (
                row["metrics"]["hall_aupr"]["mean"],
                row["metrics"]["real_auc"]["mean"],
                -row["width"],
            ))["width"]
            if selected != expected:
                raise ValueError(f"{model}/{fusion}: selected width differs from validation ranking")
            test = case["test_metrics"]
            ref = baseline(model, fusion)
            final_rows.append({
                "model": model, "fusion": fusion, "selected_width": selected,
                "fit_targets": split["fit_targets"],
                "validation_targets": split["validation_targets"],
                "test_targets": split["test_targets"],
                **{f"selected_val_{key}_{stat}": case["validation_metrics"][key][stat]
                   for key in ("real_f1", "real_auc", "hall_aupr") for stat in ("mean", "std")},
                **{f"selected_test_{key}_{stat}": test[key][stat]
                   for key in ("real_f1", "real_auc", "hall_aupr") for stat in ("mean", "std")},
                **{f"width128_test_{key}_{stat}": ref[key][stat]
                   for key in ("real_f1", "real_auc", "hall_aupr") for stat in ("mean", "std")},
                "delta_test_real_auc": test["real_auc"]["mean"] - ref["real_auc"]["mean"],
                "delta_test_hall_aupr": test["hall_aupr"]["mean"] - ref["hall_aupr"]["mean"],
            })
    if len(split_sha) != 1 or len(final_rows) != 12 or len(grid_rows) != 60:
        raise ValueError("Model splits or result counts differ")
    write_csv(OUT / "validation_grid.csv", grid_rows)
    write_csv(OUT / "final_comparison.csv", final_rows)
    lines = [
        "# COCO4000-512 softmax_target_penalty：单隐藏层宽度调参",
        "",
        "四模型均沿用相同的 3200/800 图像级训练/测试划分；从 3200 张训练图像中固定抽出 640 张作为宽度选择验证集，其余 2560 张拟合候选模型。四模型的原训练图像集合相同，验证划分使用同一 seed 20260918。测试图像没有参与宽度选择。候选宽度：32、64、128、256、512；每个宽度、特征组合以 seeds 43–45 训练。",
        "",
        "每个模型的 risk-only、risk+attention、risk+I 分别按三种子验证集 Hall AUPR 均值选宽度；同分依次比较 Real AUC、较小宽度。选定后在全部 3200 张训练图像上重训三种子，最后计算 800 张测试图像的结果。输入逐列使用拟合数据的均值和总体标准差 z-score，网络无 BatchNorm；ReLU、dropout 0.3 和其他训练设置与先前的单隐藏层实验相同。",
        "",
        "与先前完整训练集宽度 128 的测试结果并排，仅作参照；这些测试集此前已用于其他实验，因此最终数值仍属探索性。调宽度的选择过程本身只用了验证集。",
        "",
        "| 模型 | 特征 | 选定宽度 | 验证 Hall AUPR | 测试 Real F1 | 测试 Real AUC | 测试 Hall AUPR | 宽度128 AUC / Hall AUPR | Δ AUC / Δ Hall AUPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {DISPLAY[row['model']]} | {row['fusion']} | {row['selected_width']} | "
            f"{row['selected_val_hall_aupr_mean']:.3f} ± {row['selected_val_hall_aupr_std']:.3f} | "
            f"{row['selected_test_real_f1_mean']:.3f} ± {row['selected_test_real_f1_std']:.3f} | "
            f"{row['selected_test_real_auc_mean']:.3f} ± {row['selected_test_real_auc_std']:.3f} | "
            f"{row['selected_test_hall_aupr_mean']:.3f} ± {row['selected_test_hall_aupr_std']:.3f} | "
            f"{row['width128_test_real_auc_mean']:.3f} / {row['width128_test_hall_aupr_mean']:.3f} | "
            f"{row['delta_test_real_auc']:+.4f} / {row['delta_test_hall_aupr']:+.4f} |"
        )
    lines += [
        "",
        "候选宽度的完整验证集 Real F1、Real AUC、Hall AUPR 均值及标准差见 [validation_grid.csv](validation_grid.csv)；最终对照的完整数值见 [final_comparison.csv](final_comparison.csv)。各模型的选择记录与逐种子指标在对应目录：",
        "",
    ]
    for model in MODELS:
        lines.append(f"- [{DISPLAY[model]}]({model}/summary.json)")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
