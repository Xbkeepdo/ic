#!/usr/bin/env python3
"""Summarize COCO Real/Hall classification metrics for 8:1:1 and 8:2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DUAL = OUTPUTS / "COCO4000-512-ENDAC-BASELINES-811-DUAL"
STRICT = OUTPUTS / "COCO4000-512-ENDAC-BASELINES-V2"
GRID = OUTPUTS / "log_s_hpre_k_tau_ablation"
MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
METHODS = (
    ("metatoken_gb", "MetaToken"),
    ("svar", "SVAR"),
    ("projectaway", "ProjectAway"),
    ("token_grounding_mlp", "TokenGrounding (MLP)"),
    ("inslen", "InsLen"),
    ("ours", "Ours"),
)
SEEDS = (43, 44, 45)
KEYS = (
    "real_precision", "real_recall", "real_f1", "accuracy_real",
    "hall_precision", "hall_recall", "hall_f1", "accuracy_hall",
)
KEYS_82 = (
    "real_precision", "real_recall", "real_f1",
    "hall_precision", "hall_recall", "hall_f1", "accuracy",
)


def stat(values: list[float]) -> tuple[float, float]:
    if len(values) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} seeds, got {len(values)}")
    return mean(values), pstdev(values)


def fmt(row: dict, key: str) -> str:
    return f"{100 * float(row[key + '_mean']):.2f} ± {100 * float(row[key + '_std']):.2f}"


def write_811() -> None:
    with (DUAL / "complete_results.csv").open(newline="") as handle:
        source = list(csv.DictReader(handle))
    lookup = {(r["model"], r["method"]): r for r in source}
    assert len(source) == len(MODELS) * len(METHODS)
    lines = [
        "# COCO 8:1:1：Ours 与 baseline 的分类指标", "",
        "图片级划分：3,200 train / 400 validation / 400 test。Real 和 Hall 的阈值分别在验证集最大化对应类别 F1；因此两列 Accuracy 也各自对应不同阈值。数值为 seeds 43/44/45 的测试集百分数均值 ± 总体标准差。Ours 使用 K=32、τ=0.03 的 Risk+F_log；MetaToken 使用 GB 检测器，TokenGrounding 使用 MLP 检测器。不包含 DHCP。", "",
    ]
    for model, display in MODELS:
        lines += [f"## {display}", "",
                  "| Method | Real Precision | Real Recall | Real F1 | Acc@Real | Hall Precision | Hall Recall | Hall F1 | Acc@Hall |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method, name in METHODS:
            r = lookup[(model, method)]
            assert int(r["num_seeds"]) == len(SEEDS)
            parts = [fmt(r, key.replace("accuracy_real", "accuracy_real_threshold").replace("accuracy_hall", "accuracy_hall_threshold")) for key in KEYS]
            lines.append("| " + name + " | " + " | ".join(parts) + " |")
        lines.append("")
    lines += ["逐种子原始结果及 AUROC/AUPR 见 [complete_results.md](complete_results.md) 与各模型的 `seed*/<method>/result.json`。", ""]
    path = DUAL / "f1_recall_precision_accuracy_summary.md"
    path.write_text("\n".join(lines))
    print(path)


def baseline_entry(method: str, saved: dict) -> dict:
    methods = saved["methods"]
    return {
        "metatoken_gb": lambda: methods["metatoken"]["gb"],
        "svar": lambda: methods["svar"],
        "projectaway": lambda: methods["projectaway"],
        "token_grounding_mlp": lambda: methods["token_grounding"]["mlp"],
        "inslen": lambda: methods["instruct_lens"],
    }[method]()


def flatten(test: dict) -> dict[str, float]:
    real, hall = test["real_positive"], test["hallucination_positive"]
    return {
        "real_precision": float(real["precision"]),
        "real_recall": float(real["recall"]),
        "real_f1": float(real["f1"]),
        "accuracy": float(test["accuracy"]),
        "hall_precision": float(hall["precision"]),
        "hall_recall": float(hall["recall"]),
        "hall_f1": float(hall["f1"]),
    }


def strict_metrics(model: str, method: str, seed: int) -> dict[str, float]:
    if method == "ours":
        path = GRID / model / "test_grid" / "log_s_hpre_k032_tau003" / "risk_f_log" / f"seed{seed}" / "metrics.json"
        saved = json.loads(path.read_text())
        test = saved["threshold_reports"]["train_f1"]["test_metrics"]
    else:
        path = OUTPUTS / model / "COCO4000-512-ENDAC-BASELINES-V2" / "baseline" / "results" / f"seed{seed}" / f"{model}_baselines.json"
        saved = json.loads(path.read_text())
        entry = baseline_entry(method, saved)
        test = entry["threshold_reports"]["train_f1"]["test_metrics"]
        if abs(float(test["real_positive"]["f1"]) - float(entry["test_metrics"]["real_positive"]["f1"])) > 1e-12:
            raise ValueError(f"{model}/{method}/{seed}: selected test metrics differ")
    return flatten(test)


def write_82() -> None:
    rows = []
    methods = METHODS
    for model, display in MODELS:
        for method, name in methods:
            values = [strict_metrics(model, method, seed) for seed in SEEDS]
            row = {"model": model, "model_display": display, "method": method, "method_display": name, "num_seeds": len(SEEDS)}
            for key in KEYS_82:
                row[key + "_mean"], row[key + "_std"] = stat([v[key] for v in values])
            rows.append(row)
    path = STRICT / "f1_recall_precision_accuracy_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# COCO 8:2：Ours 与 baseline 的分类指标", "",
        "图片级划分：3,200 train / 800 test，**无验证集**。从保存的逐种子结果读取训练集 Real-F1 所选的单一阈值，并原样应用于测试集；Real/Hall Precision、Recall、F1 和 Accuracy 均使用这个相同阈值。因此这张表的 Hall 指标不采用独立的 Hall 阈值，不能与 8:1:1 双阈值表直接比较。数值为 seeds 43/44/45 的测试集百分数均值 ± 总体标准差。", "",
        "`Ours` 为 K=32、τ=0.03 的 Risk+F_log，与 8:1:1 对比使用同一组特征设置。MetaToken 使用 GB 检测器，TokenGrounding 使用 MLP 检测器。与 8:1:1 对比保持相同 baseline 范围，不包含 DHCP。", "",
    ]
    for model, display in MODELS:
        lines += [f"## {display}", "",
                  "| Method | Real Precision | Real Recall | Real F1 | Hall Precision | Hall Recall | Hall F1 | Accuracy |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in (x for x in rows if x["model"] == model):
            lines.append("| " + r["method_display"] + " | " + " | ".join(fmt(r, key) for key in KEYS_82) + " |")
        lines.append("")
    lines += ["完整精度见 [f1_recall_precision_accuracy_summary.csv](f1_recall_precision_accuracy_summary.csv)；其他 AUROC/AUPR 结果见各方法逐种子 JSON 与 [baseline 完整结果](complete_results.md)。", ""]
    md = STRICT / "f1_recall_precision_accuracy_summary.md"
    md.write_text("\n".join(lines))
    print(md)
    print(path)


def write_mixed() -> None:
    """Display saved 8:2 baselines alongside saved 8:1:1 Ours results."""
    with (STRICT / "f1_recall_precision_accuracy_summary.csv").open(newline="") as handle:
        strict_rows = list(csv.DictReader(handle))
    with (DUAL / "complete_results.csv").open(newline="") as handle:
        dual_rows = list(csv.DictReader(handle))
    strict = {(r["model"], r["method"]): r for r in strict_rows}
    dual_ours = {r["model"]: r for r in dual_rows if r["method"] == "ours"}
    if len(dual_ours) != len(MODELS):
        raise ValueError("Missing 8:1:1 Ours model results")

    metric_keys = (
        "real_precision", "real_recall", "real_f1", "accuracy_real",
        "hall_precision", "hall_recall", "hall_f1", "accuracy_hall",
    )
    rows = []
    for model, model_display in MODELS:
        for method, method_display in METHODS:
            is_ours = method == "ours"
            source = dual_ours[model] if is_ours else strict[(model, method)]
            row = {
                "model": model, "model_display": model_display,
                "method": method, "method_display": method_display,
                "source_split": "8:1:1" if is_ours else "8:2",
                "threshold_rule": "validation_dual_f1" if is_ours else "train_real_f1",
                "num_seeds": int(source["num_seeds"]),
            }
            if row["num_seeds"] != len(SEEDS):
                raise ValueError(f"Unexpected seed count: {model}/{method}")
            for key in metric_keys:
                source_key = key
                if key.startswith("accuracy_"):
                    source_key = (key + "_threshold") if is_ours else "accuracy"
                row[key + "_mean"] = float(source[source_key + "_mean"])
                row[key + "_std"] = float(source[source_key + "_std"])
            rows.append(row)

    csv_path = STRICT / "mixed_82_baselines_811_ours.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# COCO：8:2 baseline + 8:1:1 Ours（混合展示）", "",
        "**来源不同，不能将这一表的差值作为同一测试集上的方法提升。** Baseline 使用图片级 8:2 划分（3,200 train / 800 test），阈值由训练集 Real-F1 选择一次。Ours 使用图片级 8:1:1 划分（3,200 train / 400 validation / 400 test），Real/Hall 阈值分别由验证集对应类别的 F1 选择。两者测试图片数及阈值协议不同。", "",
        "Ours 为 Risk+F_log，K=32、τ=0.03；MetaToken 使用 GB，TokenGrounding 使用 MLP。均为 seeds 43/44/45 的测试集百分数均值 ± 总体标准差，不包含 DHCP。Baseline 的 Acc@Real 与 Acc@Hall 数值相同，因为使用同一个阈值。", "",
    ]
    for model, display in MODELS:
        lines += [f"## {display}", "",
                  "| Method | Split | Real Precision | Real Recall | Real F1 | Acc@Real | Hall Precision | Hall Recall | Hall F1 | Acc@Hall |",
                  "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in (r for r in rows if r["model"] == model):
            values = [fmt(row, key) for key in metric_keys]
            lines.append("| " + row["method_display"] + " | " + row["source_split"] + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += [
        "数据来源：[8:2 baseline 汇总](f1_recall_precision_accuracy_summary.md)、[8:1:1 Ours 完整结果](../COCO4000-512-ENDAC-BASELINES-811-DUAL/complete_results.md)。[合并 CSV](mixed_82_baselines_811_ours.csv) 保留来源划分和阈值字段。", "",
    ]
    md_path = STRICT / "mixed_82_baselines_811_ours.md"
    md_path.write_text("\n".join(lines))
    print(md_path)
    print(csv_path)


def main() -> None:
    write_811()
    write_82()
    write_mixed()


if __name__ == "__main__":
    main()
