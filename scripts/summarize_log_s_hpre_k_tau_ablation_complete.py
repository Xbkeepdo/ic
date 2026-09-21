#!/usr/bin/env python3
"""Build a readable complete report for the log-s hpre K/tau ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_hpre_k_tau_ablation"
BASE = ROOT / "outputs/linear_cost_log_s_risk_four_f_fusion/summary.csv"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
DISPLAY = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
TOP_K = (16, 32, 64, 128)
TAUS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
GROUPS = ("risk", "risk_f_log")
METRICS = ("real_f1", "real_auc", "hall_aupr")


def load_validation() -> list[dict]:
    rows = []
    for model in MODELS:
        with (OUT / model / "validation_summary.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                typed = dict(row)
                typed["top_k"] = int(row["top_k"])
                typed["tau"] = float(row["tau"])
                for metric in METRICS:
                    typed[f"{metric}_mean"] = float(row[f"{metric}_mean"])
                    typed[f"{metric}_std"] = float(row[f"{metric}_std"])
                rows.append(typed)
    return rows


def load_baselines() -> dict[tuple[str, str], dict]:
    with BASE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = {
        "risk": "risk__log_s_hpre",
        "risk_f_log": "log_s_hpre__log_residual_fusion",
        "f_log": "f__log_residual_fusion",
    }
    output = {}
    for model in MODELS:
        for group, feature in names.items():
            row = next(
                value for value in rows
                if value["model"] == model
                and value["threshold"] == "train_f1"
                and value["feature_group"] == feature
            )
            output[(model, group)] = {
                metric: {
                    "mean": float(row[f"{metric}_mean"]),
                    "std": float(row[f"{metric}_std"]),
                }
                for metric in METRICS
            }
    return output


def grid(rows: list[dict], group: str, metric: str, model: str | None = None) -> np.ndarray:
    selected = [row for row in rows if row["group"] == group]
    if model is not None:
        selected = [row for row in selected if row["model"] == model]
    values = []
    for top_k in TOP_K:
        current = []
        for tau in TAUS:
            matches = [
                row[f"{metric}_mean"] for row in selected
                if row["top_k"] == top_k and abs(row["tau"] - tau) < 1e-12
            ]
            expected = 1 if model is not None else len(MODELS)
            if len(matches) != expected:
                raise ValueError((model, group, metric, top_k, tau, len(matches)))
            current.append(float(np.mean(matches)))
        values.append(current)
    return np.asarray(values)


def add_grid(lines: list[str], title: str, values: np.ndarray) -> None:
    lines += [f"### {title}", ""]
    lines.append("| K \\ tau | " + " | ".join(f"{tau:.2f}" for tau in TAUS) + " |")
    lines.append("|---:|" + "---:|" * len(TAUS))
    for top_k, row in zip(TOP_K, values):
        lines.append(f"| {top_k} | " + " | ".join(f"{value:.4f}" for value in row) + " |")
    lines.append("")


def main() -> None:
    validation = load_validation()
    baselines = load_baselines()
    selection = json.loads((OUT / "shared_selection.json").read_text())
    selected = selection["selected"]
    final = {
        model: json.loads((OUT / model / "final_test_summary.json").read_text())["metrics"]
        for model in MODELS
    }

    with (OUT / "validation_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validation[0]))
        writer.writeheader()
        writer.writerows(validation)

    comparison_rows = []
    for model in MODELS:
        for group in GROUPS:
            row = {"model": model, "feature": group}
            for metric in METRICS:
                current = final[model][group][metric]
                baseline = baselines[(model, group)][metric]
                row[f"selected_{metric}_mean"] = current["mean"]
                row[f"selected_{metric}_std"] = current["std"]
                row[f"baseline_{metric}_mean"] = baseline["mean"]
                row[f"baseline_{metric}_std"] = baseline["std"]
                row[f"delta_{metric}"] = current["mean"] - baseline["mean"]
            comparison_rows.append(row)
    with (OUT / "final_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    lines = [
        "# Log-s hpre 的 K/tau 消融：完整结果", "",
        "## 结论", "",
        f"内部验证选择的四模型共享设置为 **K={selected['top_k']}, tau={selected['tau']:.2f}**。",
        f"它的四模型平均验证AUROC为 **{selected['mean_validation_real_auc']:.4f}**，"
        f"平均HALL-AUPR为 **{selected['mean_validation_hall_aupr']:.4f}**，"
        f"平均模型排名为 **{selected['mean_model_rank']:.2f}**。", "",
        "该设置在内部验证上相对原基准`K=64, tau=0.07`四个模型全部提高。"
        "在最终测试上，融合特征的收益集中在Qwen3和InternVL；Qwen2.5基本持平，LLaVA AUROC基本持平但HALL-AUPR提高。", "",
        "## 实验口径", "",
        "- 28组完整网格只在原训练集内部固定80/20图片级划分上比较。",
        "- 共享设置确定后，才在固定800图测试集上评估。",
        "- 表中测试数值为seeds 43/44/45的均值±总体标准差；F1使用训练集选择的阈值。",
        "- MLP固定为单隐藏层128、batch 256、dropout 0.3、LR 1e-3、WD 1e-5、patience 10、输入标准化、无BatchNorm。", "",
        "## 最终测试：共享设置与原基准", "",
        "原基准为`K=64, tau=0.07`，共享新设置为`K=128, tau=0.03`。", "",
    ]

    for group, title in (("risk_f_log", "Risk + F_log"), ("risk", "Risk-only")):
        lines += [f"### {title}", ""]
        lines.append("| 模型 | 新F1 | 新AUC | 基准AUC | ΔAUC | 新HALL-AUPR | 基准HALL-AUPR | ΔAUPR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        deltas_auc, deltas_aupr = [], []
        for model in MODELS:
            current = final[model][group]
            base = baselines[(model, group)]
            da = current["real_auc"]["mean"] - base["real_auc"]["mean"]
            dp = current["hall_aupr"]["mean"] - base["hall_aupr"]["mean"]
            deltas_auc.append(da)
            deltas_aupr.append(dp)
            lines.append(
                f"| {DISPLAY[model]} | {current['real_f1']['mean']:.4f}±{current['real_f1']['std']:.4f} | "
                f"{current['real_auc']['mean']:.4f}±{current['real_auc']['std']:.4f} | "
                f"{base['real_auc']['mean']:.4f} | {da:+.4f} | "
                f"{current['hall_aupr']['mean']:.4f}±{current['hall_aupr']['std']:.4f} | "
                f"{base['hall_aupr']['mean']:.4f} | {dp:+.4f} |"
            )
        lines.append(
            f"| **四模型平均** | — | — | — | **{np.mean(deltas_auc):+.4f}** | — | — | **{np.mean(deltas_aupr):+.4f}** |"
        )
        lines.append("")

    lines += ["### 固定 F_log-only 参照", ""]
    lines.append("| 模型 | Real F1 | Real AUC | HALL-AUPR |")
    lines.append("|---|---:|---:|---:|")
    for model in MODELS:
        value = baselines[(model, "f_log")]
        lines.append(
            f"| {DISPLAY[model]} | {value['real_f1']['mean']:.4f}±{value['real_f1']['std']:.4f} | "
            f"{value['real_auc']['mean']:.4f}±{value['real_auc']['std']:.4f} | "
            f"{value['hall_aupr']['mean']:.4f}±{value['hall_aupr']['std']:.4f} |"
        )
    lines += ["", "## 每模型内部验证最优项", ""]
    lines.append("| 模型 | 模型最优 K/tau | 最优AUC | 最优HALL-AUPR | 共享设置AUC（模型排名） | 基准AUC（模型排名） |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for model in MODELS:
        rows = sorted(
            [row for row in validation if row["model"] == model and row["group"] == "risk_f_log"],
            key=lambda row: (row["real_auc_mean"], row["hall_aupr_mean"]),
            reverse=True,
        )
        best = rows[0]
        shared_rank, shared_row = next(
            (rank, row) for rank, row in enumerate(rows, 1)
            if row["top_k"] == selected["top_k"] and abs(row["tau"] - selected["tau"]) < 1e-12
        )
        base_rank, base_row = next(
            (rank, row) for rank, row in enumerate(rows, 1)
            if row["top_k"] == 64 and abs(row["tau"] - 0.07) < 1e-12
        )
        lines.append(
            f"| {DISPLAY[model]} | `{best['top_k']}/{best['tau']:.2f}` | {best['real_auc_mean']:.4f} | "
            f"{best['hall_aupr_mean']:.4f} | {shared_row['real_auc_mean']:.4f}（#{shared_rank}） | "
            f"{base_row['real_auc_mean']:.4f}（#{base_rank}） |"
        )

    lines += ["", "## 四模型平均内部验证网格", ""]
    add_grid(lines, "Risk + F_log：平均AUROC", grid(validation, "risk_f_log", "real_auc"))
    add_grid(lines, "Risk + F_log：平均HALL-AUPR", grid(validation, "risk_f_log", "hall_aupr"))
    add_grid(lines, "Risk-only：平均AUROC", grid(validation, "risk", "real_auc"))
    add_grid(lines, "Risk-only：平均HALL-AUPR", grid(validation, "risk", "hall_aupr"))

    lines += ["## 各模型完整内部验证网格", ""]
    for model in MODELS:
        lines += [f"## {DISPLAY[model]}", ""]
        add_grid(lines, "Risk + F_log AUROC", grid(validation, "risk_f_log", "real_auc", model))
        add_grid(lines, "Risk + F_log HALL-AUPR", grid(validation, "risk_f_log", "hall_aupr", model))
        add_grid(lines, "Risk-only AUROC", grid(validation, "risk", "real_auc", model))
        add_grid(lines, "Risk-only HALL-AUPR", grid(validation, "risk", "hall_aupr", model))

    lines += [
        "## 配套文件", "",
        "- `shared_selection.md`：28组共享排名。",
        "- `validation_all.csv`：4模型×28设置×2特征组的全部验证均值和标准差。",
        "- `final_comparison.csv`：最终测试相对原基准的完整数值。",
        "- `figures/`：各模型与四模型平均热力图，以及固定K/固定tau切片图。", "",
    ]
    (OUT / "complete_results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
