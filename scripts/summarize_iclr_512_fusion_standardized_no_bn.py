#!/usr/bin/env python3
"""Summarize the standardized no-BatchNorm ICLR fusion architecture comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_iclr_512_fusion_standardized_no_bn import ARCHITECTURES, OUT
from scripts.train_iclr_endac_attention_i_fusion import FUSIONS, MODELS, RISK_KEYS, ROOT, SEEDS

OLD = ROOT / "outputs/iclr_512_endac_attention_i_fusion"
PROTOCOLS = ("old_3hidden_bn_raw", *ARCHITECTURES)
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "internvl_2_5_8b": "InternVL-2.5-8B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}


def metrics_for(protocol: str, model: str, risk: str, fusion: str, seed: int) -> dict:
    root = OLD / model if protocol == "old_3hidden_bn_raw" else OUT / protocol / model
    folder = root / risk / fusion / f"seed{seed}"
    config = json.loads((folder / "config.json").read_text())
    is_new = protocol != "old_3hidden_bn_raw"
    if bool(config.get("standardize_input", False)) != is_new:
        raise ValueError(f"Incorrect standardization protocol: {folder}")
    if bool(config.get("batch_norm", True)) == is_new:
        raise ValueError(f"Incorrect BatchNorm protocol: {folder}")
    if is_new:
        if config["hidden_sizes"] != list(ARCHITECTURES[protocol]):
            raise ValueError(f"Incorrect hidden sizes: {folder}")
        with np.load(folder / "input_scaler.npz") as scaler:
            if (len(scaler["mean"]) != len(scaler["scale"])
                    or not np.isfinite(scaler["mean"]).all()
                    or not np.isfinite(scaler["scale"]).all()
                    or np.any(scaler["scale"] <= 0)):
                raise ValueError(f"Invalid saved input scaler: {folder}")
    state = json.loads((folder / "metrics.json").read_text())
    report = state["threshold_reports"]["train_f1"]["test_metrics"]
    return {
        "real_f1": float(report["real_positive"]["f1"]),
        "real_auc": float(report["real_positive"]["auc"]),
        "hall_aupr": float(report["hallucination_positive"]["aupr"]),
    }


def main() -> None:
    detailed = []
    for model in MODELS:
        for risk in RISK_KEYS:
            for fusion in FUSIONS:
                for protocol in PROTOCOLS:
                    seeds = [metrics_for(protocol, model, risk, fusion, seed) for seed in SEEDS]
                    detailed.append({
                        "model": model, "risk": risk, "fusion": fusion, "protocol": protocol,
                        **{f"{metric}_{stat}": float(fn([row[metric] for row in seeds]))
                           for metric in ("real_f1", "real_auc", "hall_aupr")
                           for stat, fn in (("mean", np.mean), ("std", np.std))},
                    })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "detailed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)

    lines = [
        "# COCO4000-512 risk + attention / I：输入标准化与无 BatchNorm 的 MLP 对照",
        "",
        "数据、标签、图像级 3200/800 划分和四种 risk 分支与原融合实验完全相同；所有组合使用 seeds 43–45。输入的每一列只用训练集均值及总体标准差做 z-score，常数列的 scale 设为 1；测试集使用同一统计量。每个种子保存 input_scaler.npz。两个新网络均无 BatchNorm；单隐藏层为 128，三隐藏层为 128/64/32，均保留 ReLU 和 dropout 0.3。模型按最小训练损失选 checkpoint；Real F1 阈值从训练集选取。",
        "",
        "旧实验 old_3hidden_bn_raw 的输入未标准化，三隐藏层且含 BatchNorm，仅作参照。新旧实验同时改变了标准化和 BatchNorm，不能从两者的差值单独归因于其中一个改动。四个 risk 分支下的测试指标均为探索性结果。",
        "",
        "## 四种 risk 分别报告",
        "",
        "每格为 Real AUC / Hall AUPR，取 seeds 43–45 的均值；没有跨 risk 求平均。risk-only、risk+attention、risk+I 均使用该行相同的 risk 分支和训练设置。Real F1 与逐种子标准差见 detailed.csv 及下方逐模型报告。",
        "",
        "| 模型 | Risk | 训练设置 | risk-only | risk+attention | risk+I |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for risk in RISK_KEYS:
            for protocol in PROTOCOLS:
                values = {
                    row["fusion"]: row for row in detailed
                    if row["model"] == model and row["risk"] == risk
                    and row["protocol"] == protocol
                }
                def cell(fusion: str) -> str:
                    row = values[fusion]
                    return f"{row['real_auc_mean']:.3f} / {row['hall_aupr_mean']:.3f}"
                lines.append(f"| {DISPLAY[model]} | {risk} | {protocol} | "
                             f"{cell('risk')} | {cell('risk_attention')} | {cell('risk_I')} |")

    lines += ["", "## 逐分支结果与训练产物", "",
              "[全部组合的均值与标准差](detailed.csv)。每个模型在两种新结构下另有完整三种子表：", ""]
    for model in MODELS:
        lines.append(f"- {DISPLAY[model]}：[单隐藏层](one_hidden_128/{model}/summary.md)；"
                     f"[三隐藏层](three_hidden_128_64_32/{model}/summary.md)。")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
