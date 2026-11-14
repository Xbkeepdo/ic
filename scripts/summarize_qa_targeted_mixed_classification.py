#!/usr/bin/env python3
"""CLEVR prompt-last and AMBER-three-way mixed QA classification table."""

from __future__ import annotations

import csv
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from detection.baselines import (  # noqa: E402
    SVARMLP,
    build_dense_baseline_matrix,
    evaluate_detection_scores,
    sklearn_hallucination_scores,
    torch_hallucination_scores,
)
from features.baseline import get_baseline_payload  # noqa: E402
from features.qa_baseline import baseline_record_for_position, qa_image_identity  # noqa: E402
from scripts.summarize_qa_mixed_811_baselines_82_ours_classification import (  # noqa: E402
    METRICS,
    _numbers,
    _own_path,
    _variants,
    collect,
)
from scripts.train_qa_811_cached import _assignment  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


BASE = ROOT / "outputs" / "qa_benchmarks"
OLD = "LOGS-K32-TAU003-NATIVE6-FULL"
NEW = f"{OLD}-811"
MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
SEEDS = (43, 44, 45)
CATEGORIES = ("existence", "attribute", "relation")
DETECTORS = ("Log-S", "MetaToken-LR", "MetaToken-GB", "SVAR", "DHCP",
             "ProjectAway", "TokenGrounding-MLP", "InsLens")
STEM = "mixed_811_baselines_82_ours_clevr_amber3_classification"


def _records(model: str) -> list[dict]:
    path = (BASE / model / OLD / "amber_discriminative" / "baseline"
            / "answer_correctness_all" / "features.pkl")
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    assignment = _assignment("amber_discriminative")
    selected = [baseline_record_for_position(row, "prompt_last_token") for row in rows
                if assignment[qa_image_identity(row)] == "test"]
    if len(selected) != 1437 or any(row is None for row in selected):
        raise ValueError(f"Incomplete AMBER 8:1:1 prompt-last cohort: {model}")
    if len({row["key"] for row in selected}) != len(selected):
        raise ValueError(f"Duplicate AMBER test questions: {model}")
    return selected


def _matrix(records: list[dict], method: str, *, start: int = 5,
            end: int = 19) -> np.ndarray:
    matrix, labels, kept = build_dense_baseline_matrix(
        records, method, svar_layer_start=start, svar_layer_end=end
    )
    if len(kept) != len(records) or [row["key"] for row in kept] != [row["key"] for row in records]:
        raise ValueError(f"Dropped or reordered AMBER baseline rows: {method}")
    if not np.array_equal(labels, [row["label"] for row in records]):
        raise ValueError(f"AMBER label mismatch: {method}")
    return matrix


def _scores(name: str, variant: dict, matrices: dict[str, np.ndarray],
            records: list[dict]) -> np.ndarray:
    if name in ("MetaToken-LR", "MetaToken-GB"):
        classifier = load_pkl(variant["checkpoint"])
        return sklearn_hallucination_scores(classifier, matrices["metatoken"])
    if name == "TokenGrounding-MLP":
        classifier = load_pkl(variant["checkpoint"])
        return sklearn_hallucination_scores(classifier, matrices["token_grounding"])
    if name == "SVAR":
        saved = torch.load(variant["checkpoint"], map_location="cpu", weights_only=False)
        matrix = matrices["svar"]
        if int(saved["input_dim"]) != matrix.shape[1]:
            raise ValueError("SVAR checkpoint input width differs from AMBER features")
        model = SVARMLP(matrix.shape[1], hidden_dim=int(saved["hidden_dim"]))
        model.load_state_dict(saved["state_dict"])
        return torch_hallucination_scores(model, matrix, torch.device("cpu"))
    if name == "ProjectAway":
        return np.asarray([float(get_baseline_payload(row, "projectaway")["hallucination_score"])
                           for row in records], dtype=np.float64)
    if name == "InsLens":
        return -np.asarray([float(get_baseline_payload(row, "instruct_lens")["score"])
                            for row in records], dtype=np.float64)
    raise ValueError(f"Unexpected score-requiring detector: {name}")


def _row(model: str, task: str, detector: str, split: str, count: int,
         seed: int, threshold: float, values: dict) -> dict:
    if count <= 0 or not math.isfinite(float(threshold)):
        raise ValueError(f"Invalid cohort or threshold: {model}/{task}/{detector}/{seed}")
    return {
        "model": model, "task": task,
        "position": "prompt_last_token", "label_protocol": "answer_correctness_all",
        "detector": detector, "split": split, "test_questions": count,
        "seed": seed, "threshold": float(threshold), **values,
    }


def _compare_metrics(actual: dict, expected: dict, context: str) -> None:
    for name in METRICS:
        if abs(actual[name] - expected[name]) > 1e-6:
            raise ValueError(f"Overall metric mismatch: {context}/{name}: "
                             f"{actual[name]} != {expected[name]}")


def amber_rows(model: str) -> list[dict]:
    records = _records(model)
    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    masks = {category: np.asarray([row["source_split"] == category for row in records])
             for category in CATEGORIES}
    expected_counts = {"existence": 498, "attribute": 766, "relation": 173}
    for category, mask in masks.items():
        if int(mask.sum()) != expected_counts[category] or set(labels[mask]) != {0, 1}:
            raise ValueError(f"Invalid AMBER test category: {model}/{category}")
    root = (BASE / model / NEW / "amber_discriminative" / "baseline"
            / "answer_correctness_all" / "prompt_last_token")
    dhcp = json.loads((BASE / NEW / "amber3_dhcp_811.json").read_text())[model]
    if dhcp["test_rows"] != len(records):
        raise ValueError(f"DHCP test cohort differs: {model}")
    first = json.loads((root / "seed_43" / "result.json").read_text())
    svar_checkpoint = torch.load(first["methods"]["svar"]["checkpoint"],
                                 map_location="cpu", weights_only=False)
    matrices = {
        "metatoken": _matrix(records, "metatoken"),
        "token_grounding": _matrix(records, "token_grounding"),
        "svar": _matrix(records, "svar", start=int(svar_checkpoint["layer_start"]),
                        end=int(svar_checkpoint["layer_end_exclusive"])),
    }
    rows = []
    for seed in SEEDS:
        baseline = json.loads((root / f"seed_{seed}" / "result.json").read_text())
        if baseline["counts"]["test"] != len(records) or baseline["threshold_selection"] != "validation_real_f1":
            raise ValueError(f"Unexpected AMBER 8:1:1 baseline protocol: {model}/seed_{seed}")
        variants = _variants(baseline)
        for name, variant in variants.items():
            if name == "DHCP":
                cached = dhcp["seeds"][str(seed)]
                if abs(cached["threshold_real"] - variant["threshold"]) > 1e-12:
                    raise ValueError(f"DHCP threshold mismatch: {model}/seed_{seed}")
                if abs(cached["overall_auroc"] - variant["test_metrics"]["real_positive"]["auc"]) > 1e-6:
                    raise ValueError(f"DHCP overall AUROC mismatch: {model}/seed_{seed}")
                category_metrics = cached["categories"]
            else:
                scores = _scores(name, variant, matrices, records)
                overall = evaluate_detection_scores(labels, scores, variant["threshold"],
                                                    positive_class="real")
                _compare_metrics(_numbers(overall, own=False),
                                 _numbers(variant["test_metrics"], own=False),
                                 f"{model}/{name}/seed_{seed}")
                category_metrics = {
                    category: evaluate_detection_scores(
                        labels[mask], scores[mask], variant["threshold"], positive_class="real"
                    ) for category, mask in masks.items()
                }
            for category, mask in masks.items():
                metrics = category_metrics[category]
                count = int(mask.sum())
                if (metrics["confusion_matrix_hallucination_positive"][0][0]
                        + metrics["confusion_matrix_hallucination_positive"][0][1]
                        + metrics["confusion_matrix_hallucination_positive"][1][0]
                        + metrics["confusion_matrix_hallucination_positive"][1][1]) != count:
                    raise ValueError(f"Category confusion matrix count mismatch: {model}/{name}/{category}")
                rows.append(_row(model, f"AMBER {category}", name, "8:1:1",
                                 count, seed, variant["threshold"],
                                 _numbers(metrics, own=False)))
        own = json.loads(_own_path(model, "amber_discriminative", "answer_correctness_all",
                                   "prompt_last_token", seed).read_text())
        if own["split_protocol"] != "strict_82_no_validation" or own["threshold_selection"] != "train_f1":
            raise ValueError(f"Unexpected AMBER Log-S 8:2 protocol: {model}/seed_{seed}")
        for category in CATEGORIES:
            metrics = own["test_groups"]["source_split"][category]
            selected = own["threshold_reports"]["train_f1"]["test_groups"]["source_split"][category]
            if metrics != selected:
                raise ValueError(f"AMBER Log-S selected category metrics differ: {model}/{category}/seed_{seed}")
            count = metrics["real"]["support"] + metrics["hallucination"]["support"]
            if count != {"existence": 987, "attribute": 1500, "relation": 344}[category]:
                raise ValueError(f"Unexpected AMBER Log-S 8:2 category size: {model}/{category}")
            rows.append(_row(model, f"AMBER {category}", "Log-S", "8:2",
                             count, seed, own["threshold"], _numbers(metrics, own=True)))
        print(f"{model}/seed_{seed}: seven baseline overall metrics matched", flush=True)
    return rows


def clevr_rows() -> list[dict]:
    rows = []
    for row in collect():
        if (row["dataset"] == "clevr_exist_9k"
                and row["position"] == "prompt_last_token"
                and row["label_protocol"] == "answer_correctness_all"):
            rows.append(_row(row["model"], "CLEVR-Exist-9K", row["detector"],
                             row["split"], row["test_questions"], row["seed"],
                             row["threshold"], {name: row[name] for name in METRICS}))
    if len(rows) != len(MODELS) * len(DETECTORS) * len(SEEDS):
        raise ValueError(f"Incomplete CLEVR target rows: {len(rows)}")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["model"], row["task"], row["detector"])
        grouped.setdefault(key, []).append(row)
    if len(rows) != 384 or len(grouped) != 128:
        raise ValueError(f"Expected 384 seed and 128 summary rows: {len(rows)}, {len(grouped)}")
    result = []
    for key, seeds in grouped.items():
        if {row["seed"] for row in seeds} != set(SEEDS):
            raise ValueError(f"Missing seeds: {key}")
        for field in ("split", "test_questions", "position", "label_protocol"):
            if len({row[field] for row in seeds}) != 1:
                raise ValueError(f"Inconsistent {field}: {key}")
        summary = {field: seeds[0][field] for field in ("model", "task", "position",
                                                        "label_protocol", "detector",
                                                        "split", "test_questions")}
        for metric in METRICS:
            values = [row[metric] * 100 for row in seeds]
            summary[f"{metric}_mean_pct"] = statistics.mean(values)
            summary[f"{metric}_std_pct"] = statistics.pstdev(values)
        result.append(summary)
    order = {name: index for index, name in enumerate(DETECTORS)}
    tasks = {"CLEVR-Exist-9K": 0, "AMBER existence": 1,
             "AMBER attribute": 2, "AMBER relation": 3}
    result.sort(key=lambda row: (MODELS.index(row["model"]), tasks[row["task"]],
                                 order[row["detector"]]))
    return result


def write(rows: list[dict], summary: list[dict]) -> None:
    output = BASE / NEW
    seed_fields = ["model", "task", "position", "label_protocol", "detector",
                   "split", "test_questions", "seed", "threshold", *METRICS]
    with (output / f"{STEM}_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_fields = ["model", "task", "position", "label_protocol", "detector",
                      "split", "test_questions"]
    summary_fields += [f"{metric}_{stat}_pct" for metric in METRICS
                       for stat in ("mean", "std")]
    with (output / f"{STEM}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# QA 指定范围：baseline 8:1:1 + Log-S 8:2 分类指标",
        "",
        "只包含 CLEVR-Exist-9K `prompt_last_token | answer_correctness_all`，"
        "以及 AMBER `prompt_last_token | answer_correctness_all` 的 existence、"
        "attribute、relation 三类。四模型、每项 3 seeds；数值为测试集百分数均值±总体标准差。"
        "MetaToken 的 LR/GB 分开列出。",
        "",
        "Log-S 8:2 使用训练集 Real-F1 阈值；baseline 8:1:1 使用验证集 Real-F1 阈值。"
        "Real/Hall 共用每个结果的单一阈值，Accuracy 只列一次。两侧测试图片和阈值协议不同，"
        "表中差值不能当作同题配对提升。AMBER 各类别沿用整体 AMBER 所选的模型和阈值，"
        "未按类别重新训练或选阈值。",
        "",
        "| Model | Task | Detector | Split | Test Q | Real P | Real R | Real F1 | Hall P | Hall R | Hall F1 | Accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        metrics = [f"{row[f'{name}_mean_pct']:.2f}±{row[f'{name}_std_pct']:.2f}"
                   for name in METRICS]
        lines.append("| " + " | ".join((row["model"], row["task"], row["detector"],
                                       row["split"], str(row["test_questions"]),
                                       *metrics)) + " |")
    lines += [
        "",
        "所有非 DHCP AMBER baseline 的整体分类指标均由保存模型重新推理，并逐 seed 与原"
        " `result.json` 核对；DHCP 的整体指标已在 [DHCP 分项报告](amber3_dhcp_811.md)"
        "核对。CLEVR 取自完整[跨划分分类结果](mixed_811_baselines_82_ours_classification.md)。",
        "",
        f"[汇总 CSV]({STEM}.csv) ｜ [逐种子 CSV]({STEM}_seeds.csv)",
    ]
    (output / f"{STEM}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote 128 summary / 384 seed rows: {output / f'{STEM}.md'}")


def main() -> None:
    rows = clevr_rows()
    for model in MODELS:
        rows.extend(amber_rows(model))
    summary = summarize(rows)
    write(rows, summary)


if __name__ == "__main__":
    main()
