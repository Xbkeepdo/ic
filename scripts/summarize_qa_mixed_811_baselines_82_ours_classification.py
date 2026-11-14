#!/usr/bin/env python3
"""Join formal QA 8:1:1 baselines with 8:2 Log-S classification results."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "qa_benchmarks"
OLD = "LOGS-K32-TAU003-NATIVE6-FULL"
NEW = f"{OLD}-811"
MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
DATASETS = ("pope", "clevr_exist_9k", "amber_discriminative")
PROTOCOLS = ("answer_correctness_all", "object_hallucination_yes_only")
SEEDS = (43, 44, 45)
DETECTORS = ("Log-S", "MetaToken-LR", "MetaToken-GB", "SVAR", "DHCP",
             "ProjectAway", "TokenGrounding-MLP", "InsLens")
METRICS = ("real_precision", "real_recall", "real_f1", "hall_precision",
           "hall_recall", "hall_f1", "accuracy")
FIELDS = ("model", "dataset", "position", "label_protocol", "detector", "split",
          "train_questions", "validation_questions", "test_questions",
          "selection_protocol", "threshold_selection", "seed", "threshold",
          *METRICS)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _own_path(model: str, dataset: str, protocol: str, position: str, seed: int) -> Path:
    return (BASE / model / OLD / dataset / "results" / protocol / position
            / f"logs_risk_f_log__{position}" / f"seed_{seed}" / "result.json")


def _baseline_path(model: str, dataset: str, protocol: str, position: str,
                   seed: int) -> Path:
    return (BASE / model / NEW / dataset / "baseline" / protocol / position
            / f"seed_{seed}" / "result.json")


def _numbers(metrics: dict, *, own: bool) -> dict[str, float]:
    real = metrics["real" if own else "real_positive"]
    hall = metrics["hallucination" if own else "hallucination_positive"]
    result = {
        "real_precision": real["precision"], "real_recall": real["recall"],
        "real_f1": real["f1"], "hall_precision": hall["precision"],
        "hall_recall": hall["recall"], "hall_f1": hall["f1"],
        "accuracy": metrics["accuracy"],
    }
    if any(value is None or not math.isfinite(float(value)) or not 0 <= value <= 1
           for value in result.values()):
        raise ValueError(f"Invalid classification metric: {result}")
    return {key: float(value) for key, value in result.items()}


def _variants(baseline: dict) -> dict[str, dict]:
    methods = baseline["methods"]
    if set(methods) != {"metatoken", "svar", "dhcp", "projectaway",
                        "token_grounding", "instruct_lens"}:
        raise ValueError(f"Unexpected baseline method set: {sorted(methods)}")
    return {
        "MetaToken-LR": methods["metatoken"]["lr"],
        "MetaToken-GB": methods["metatoken"]["gb"],
        "SVAR": methods["svar"],
        "DHCP": methods["dhcp"],
        "ProjectAway": methods["projectaway"],
        "TokenGrounding-MLP": methods["token_grounding"]["mlp"],
        "InsLens": methods["instruct_lens"],
    }


def _selection_protocol(name: str, variant: dict) -> str:
    if name in ("ProjectAway", "InsLens"):
        return "training_free_validation_threshold"
    recorded = (variant.get("selection_protocol")
                or variant.get("paper_config", {}).get("selection_protocol"))
    if not recorded:
        raise ValueError(f"Missing baseline selection protocol: {name}")
    return str(recorded)


def _row(*, model: str, dataset: str, position: str, protocol: str,
         detector: str, split: str, counts: dict, selection_protocol: str,
         threshold_selection: str, seed: int, threshold: float,
         metrics: dict, own: bool) -> dict:
    return {
        "model": model, "dataset": dataset, "position": position,
        "label_protocol": protocol, "detector": detector, "split": split,
        "train_questions": int(counts["train"]),
        "validation_questions": int(counts["val"]),
        "test_questions": int(counts["test"]),
        "selection_protocol": selection_protocol,
        "threshold_selection": threshold_selection,
        "seed": seed, "threshold": float(threshold),
        **_numbers(metrics, own=own),
    }


def collect() -> list[dict]:
    rows = []
    groups = 0
    for model in MODELS:
        for dataset in DATASETS:
            positions = (("prompt_last_token",) if dataset == "amber_discriminative"
                         else ("prompt_last_token", "question_object_pre_token"))
            for position in positions:
                for protocol in PROTOCOLS:
                    groups += 1
                    for seed in SEEDS:
                        own = _read(_own_path(model, dataset, protocol, position, seed))
                        baseline = _read(_baseline_path(model, dataset, protocol,
                                                       position, seed))
                        if (own["split_protocol"] != "strict_82_no_validation"
                                or own["checkpoint_selection"] != "minimum_train_loss"
                                or own["threshold_selection"] != "train_f1"
                                or own["positive_class"] != "real"
                                or own["feature_set"] != f"logs_risk_f_log@{position}"
                                or own["label_protocol"] != protocol):
                            raise ValueError(f"Unexpected Log-S 8:2 protocol: {model}/{dataset}/{position}/{protocol}/seed_{seed}")
                        if (own["test_metrics"] != own["threshold_reports"]["train_f1"]["test_metrics"]
                                or own["counts"]["val"] != 0):
                            raise ValueError(f"Unexpected Log-S 8:2 selected metrics: {model}/{dataset}/{position}/{protocol}/seed_{seed}")
                        if (baseline["split_protocol"] != "strict_outer_image_level_811"
                                or baseline["threshold_selection"] != "validation_real_f1"
                                or baseline["model"] != model
                                or baseline["dataset"] != dataset
                                or baseline["position"] != position
                                or baseline["label_protocol"] != protocol
                                or baseline["seed"] != seed):
                            raise ValueError(f"Unexpected baseline 8:1:1 protocol: {model}/{dataset}/{position}/{protocol}/seed_{seed}")
                        if (baseline["counts"]["val"] <= 0
                                or baseline["counts"]["test"] <= 0
                                or baseline["counts"]["test"] > own["counts"]["test"]):
                            raise ValueError(f"Invalid cross-split cohort counts: {model}/{dataset}/{position}/{protocol}/seed_{seed}")
                        rows.append(_row(
                            model=model, dataset=dataset, position=position,
                            protocol=protocol, detector="Log-S", split="8:2",
                            counts=own["counts"],
                            selection_protocol=own["checkpoint_selection"],
                            threshold_selection="train_real_f1", seed=seed,
                            threshold=own["threshold"], metrics=own["test_metrics"],
                            own=True,
                        ))
                        for name, variant in _variants(baseline).items():
                            rows.append(_row(
                                model=model, dataset=dataset, position=position,
                                protocol=protocol, detector=name, split="8:1:1",
                                counts=baseline["counts"],
                                selection_protocol=_selection_protocol(name, variant),
                                threshold_selection="validation_real_f1", seed=seed,
                                threshold=variant["threshold"],
                                metrics=variant["test_metrics"], own=False,
                            ))
    if groups != 40 or len(rows) != 960:
        raise AssertionError(f"Expected 40 groups and 960 seed rows: {groups}, {len(rows)}")
    return rows


def summarize(seed_rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in seed_rows:
        key = tuple(row[field] for field in ("model", "dataset", "position",
                                             "label_protocol", "detector"))
        grouped.setdefault(key, []).append(row)
    if len(grouped) != 320:
        raise AssertionError(f"Expected 320 detector groups, got {len(grouped)}")
    result = []
    for key, rows in grouped.items():
        if {row["seed"] for row in rows} != set(SEEDS):
            raise ValueError(f"Incomplete seeds: {key}")
        fields = ("split", "train_questions", "validation_questions", "test_questions",
                  "selection_protocol", "threshold_selection")
        if any(len({row[field] for row in rows}) != 1 for field in fields):
            raise ValueError(f"Inconsistent seed metadata: {key}")
        summary = dict(zip(("model", "dataset", "position", "label_protocol", "detector"), key))
        summary.update({field: rows[0][field] for field in fields})
        for metric in METRICS:
            scores = [row[metric] * 100 for row in rows]
            summary[f"{metric}_mean_pct"] = statistics.mean(scores)
            summary[f"{metric}_std_pct"] = statistics.pstdev(scores)
        result.append(summary)
    return result


def write(seed_rows: list[dict], summary: list[dict]) -> None:
    output = BASE / NEW
    stem = "mixed_811_baselines_82_ours_classification"
    with (output / f"{stem}_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(seed_rows)
    headers = ["model", "dataset", "position", "label_protocol", "detector", "split",
               "train_questions", "validation_questions", "test_questions",
               "selection_protocol", "threshold_selection"]
    headers += [f"{metric}_{stat}_pct" for metric in METRICS for stat in ("mean", "std")]
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# QA benchmark：baseline 8:1:1 + Log-S 8:2 分类指标",
        "",
        "完整范围：4 模型 × POPE/CLEVR/AMBER × 可用位置 × 2 标签协议，共 40 个条件；"
        "每条件有 Log-S 与 7 个 baseline 检测器（MetaToken 分 LR/GB），seeds 43/44/45。"
        "数值为测试集百分数 mean±population std。",
        "",
        "Log-S：原图片级 8:2，训练集最小 loss 选 checkpoint、训练集 Real-F1 选阈值；"
        "baseline：图片级 8:1:1，验证集选 checkpoint（适用时）和 Real-F1 阈值。"
        "每个结果只有一个阈值，Real/Hall 共用它，Accuracy 也只列一次。"
        "8:1:1 的 test 是原 8:2 test 的一部分，测试题与阈值协议不同，"
        "因此表内跨划分差值不能当作同题配对提升。",
        "",
        "AMBER 在本表按正式 QA 的整体测试集汇总；已单独完成的 DHCP 8:1:1 "
        "[existence/attribute/relation 分项](amber3_dhcp_811.md)另列。",
        "",
        "| Model | Dataset | Position | Label protocol | Detector | Split | Test Q | Real P | Real R | Real F1 | Hall P | Hall R | Hall F1 | Accuracy |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        metrics = [f"{row[f'{name}_mean_pct']:.2f}±{row[f'{name}_std_pct']:.2f}"
                   for name in METRICS]
        lines.append("| " + " | ".join((
            row["model"], row["dataset"], row["position"], row["label_protocol"],
            row["detector"], row["split"], str(row["test_questions"]), *metrics,
        )) + " |")
    lines += [
        "",
        f"[汇总 CSV]({stem}.csv) 含 train/validation/test 题数、方法选择协议与阈值来源；"
        f"[逐种子 CSV]({stem}_seeds.csv) 含每次测试的原始指标和所选阈值。",
    ]
    (output / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audited 40 conditions, 320 detector rows, 960 seed rows -> {output / f'{stem}.md'}")


def main() -> None:
    seed_rows = collect()
    summary = summarize(seed_rows)
    write(seed_rows, summary)


if __name__ == "__main__":
    main()
