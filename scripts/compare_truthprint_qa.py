#!/usr/bin/env python3
"""Compare finished TruthPrInt QA runs with the matched 8:1:1 Log-S runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRUTHPRINT = ROOT / "outputs/truthprint_baseline"
REFERENCE = ROOT / "outputs/qa_benchmarks/LOGS-K32-TAU003-NATIVE6-FULL-811"
MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def fmt(mean: float, std: float) -> str:
    return f"{100 * mean:.2f}±{100 * std:.2f}"


def main() -> None:
    with (TRUTHPRINT / "complete_results.csv").open(newline="") as handle:
        truthprint = {(row["model"], row["task"]): row for row in csv.DictReader(handle)}
    reference = json.loads((REFERENCE / "complete_results.json").read_text())
    amber_auroc = {}
    for line in (REFERENCE / "amber3_dhcp_811.md").read_text().splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0] in MODELS.values() and cells[1] in {
            "existence", "attribute", "relation"
        }:
            amber_auroc[(cells[0], cells[1])] = cells[3]

    lines = [
        "# TruthPrInt vs Log-S: QA 8:1:1",
        "",
        "同一图片级 train/validation/test 划分、同一 `prompt_last_token` 与 "
        "`answer_correctness_all` 标签；seeds 43/44/45。两种方法均在验证集选择阈值。"
        "数值为测试集百分数均值±总体标准差。",
        "",
        "| Model | Task | Test Q | TruthPrInt AUROC | Log-S AUROC | TruthPrInt Hall AUPR | Log-S Hall AUPR | TruthPrInt Hall F1 | Log-S Hall F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, display in MODELS.items():
        for task, label in (("clevr_exist_9k", "CLEVR-Exist-9K"),
                            ("amber_discriminative", "AMBER overall")):
            ours = truthprint[(model, task)]
            previous = reference[f"{model}/{task}/prompt_last_token/answer_correctness_all"]
            if int(ours["test_n"]) != int(previous["test_counts"]["method"]):
                raise ValueError(f"Mismatched QA test rows: {model}/{task}")
            logs = previous["detectors"]["log_s"]
            lines.append("| " + " | ".join((
                display, label, ours["test_n"],
                fmt(float(ours["auroc_mean"]), float(ours["auroc_std"])),
                fmt(logs["auc"]["mean"], logs["auc"]["std"]),
                fmt(float(ours["hall_aupr_mean"]), float(ours["hall_aupr_std"])),
                fmt(logs["hall_aupr"]["mean"], logs["hall_aupr"]["std"]),
                fmt(float(ours["hall_f1_mean"]), float(ours["hall_f1_std"])),
                fmt(logs["hall_f1"]["mean"], logs["hall_f1"]["std"]),
            )) + " |")
    lines += [
        "",
        "AMBER 三类沿用各自方法在整体 AMBER validation 选出的模型和阈值；"
        "类别 AUROC 的 Log-S 数值来自现有同划分分项评估。",
        "",
        "| Model | AMBER type | Test Q | TruthPrInt AUROC | Log-S AUROC | TruthPrInt Hall F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, display in MODELS.items():
        for dimension in ("existence", "attribute", "relation"):
            ours = truthprint[(model, f"amber_{dimension}")]
            if (display, dimension) not in amber_auroc:
                raise ValueError(f"Missing Log-S category: {display}/{dimension}")
            lines.append("| " + " | ".join((
                display, dimension, ours["test_n"],
                fmt(float(ours["auroc_mean"]), float(ours["auroc_std"])),
                amber_auroc[(display, dimension)],
                fmt(float(ours["hall_f1_mean"]), float(ours["hall_f1_std"])),
            )) + " |")
    output = TRUTHPRINT / "qa_comparison.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
