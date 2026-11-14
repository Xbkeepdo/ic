#!/usr/bin/env python3
"""Audit and summarize the cached formal QA 8:1:1 rerun."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "qa_benchmarks"
NAME = "LOGS-K32-TAU003-NATIVE6-FULL-811"
MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
DATASETS = ("pope", "clevr_exist_9k", "amber_discriminative")
PROTOCOLS = ("answer_correctness_all", "object_hallucination_yes_only")
SEEDS = (43, 44, 45)
BASELINES = ("metatoken_lr", "metatoken_gb", "svar", "dhcp", "projectaway", "token_grounding_mlp", "instruct_lens")


def _metric(value: dict, field: str) -> float:
    if field == "auc":
        result = value["real_positive"]["auc"]
    elif field == "hall_aupr":
        result = value["hallucination_positive"]["aupr"]
    else:
        result = value["hallucination_positive"]["f1"]
    if result is None or not math.isfinite(float(result)):
        raise ValueError(f"Non-finite {field}: {result}")
    return float(result)


def _aggregate(scores: list[dict]) -> dict:
    return {
        metric: {
            "mean": statistics.mean(row[metric] for row in scores),
            "std": statistics.pstdev(row[metric] for row in scores),
            "seeds": [row[metric] for row in scores],
        }
        for metric in ("auc", "hall_aupr", "hall_f1")
    }


def _format(value: dict) -> str:
    return f"{value['mean']:.4f}±{value['std']:.4f}"


def main() -> None:
    output = BASE / NAME
    report = {}
    lines = [
        "# QA 8:1:1 rerun: Log-S and six native baselines",
        "",
        "All methods use the same image-disjoint train/validation/test assignment. "
        "Validation selects checkpoints where applicable and Real-F1 thresholds; "
        "test is used only for final evaluation. The run reuses the formal "
        "8:2 generation, labels, and extracted features; the old outer-test "
        "cohort is divided into validation and test without changing training rows.",
        "",
        "AUROC and Hall AUPR are mean±population std over seeds 43/44/45. "
        "Hall F1 uses the validation-selected threshold.",
        "",
        "| Model | Dataset | Position | Label protocol | Detector | Test rows | AUROC | Hall AUPR | Hall F1 |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    expected = 0
    for model in MODELS:
        for dataset in DATASETS:
            split = json.loads((output / "splits" / f"{dataset}.json").read_text())
            if set(split["images"]["train"]) & set(split["images"]["val"]):
                raise ValueError(f"Train/val image leakage: {dataset}")
            if set(split["images"]["train"]) & set(split["images"]["test"]):
                raise ValueError(f"Train/test image leakage: {dataset}")
            if set(split["images"]["val"]) & set(split["images"]["test"]):
                raise ValueError(f"Val/test image leakage: {dataset}")
            positions = ("prompt_last_token",) if dataset == "amber_discriminative" else (
                "prompt_last_token", "question_object_pre_token"
            )
            for position in positions:
                for protocol in PROTOCOLS:
                    expected += 1
                    scores = {name: [] for name in ("log_s", *BASELINES)}
                    counts = {}
                    for seed in SEEDS:
                        root = BASE / model / NAME / dataset
                        own_path = root / "method" / protocol / position / f"seed_{seed}" / "result.json"
                        baseline_path = root / "baseline" / protocol / position / f"seed_{seed}" / "result.json"
                        own = json.loads(own_path.read_text())
                        baseline = json.loads(baseline_path.read_text())
                        if own["split_protocol"] != "strict_outer_image_level_811" or baseline["split_protocol"] != "strict_outer_image_level_811":
                            raise ValueError(f"Wrong split protocol: {model}/{dataset}/{position}/{protocol}")
                        if own["checkpoint_selection"] != "minimum_validation_loss" or own["threshold_selection"] != "validation_real_f1":
                            raise ValueError(f"Wrong own-method selection: {own_path}")
                        if baseline["threshold_selection"] != "validation_real_f1":
                            raise ValueError(f"Wrong baseline threshold selection: {baseline_path}")
                        if set(baseline["methods"]) != {"metatoken", "svar", "dhcp", "projectaway", "token_grounding", "instruct_lens"}:
                            raise ValueError(f"Incomplete baseline methods: {baseline_path}")
                        if own["counts"]["train"] <= 0 or own["counts"]["val"] <= 0 or own["counts"]["test"] <= 0:
                            raise ValueError(f"Empty own-method split: {own_path}")
                        if any(baseline["counts"][split] <= 0 for split in ("train", "val", "test")):
                            raise ValueError(f"Empty baseline split: {baseline_path}")
                        values = own["metrics"]["test"]
                        scores["log_s"].append({
                            "auc": float(values["auroc"]),
                            "hall_aupr": float(values["hallucination_aupr"]),
                            "hall_f1": float(values["hallucination"]["f1"]),
                        })
                        variants = {
                            "metatoken_lr": baseline["methods"]["metatoken"]["lr"],
                            "metatoken_gb": baseline["methods"]["metatoken"]["gb"],
                            "svar": baseline["methods"]["svar"],
                            "dhcp": baseline["methods"]["dhcp"],
                            "projectaway": baseline["methods"]["projectaway"],
                            "token_grounding_mlp": baseline["methods"]["token_grounding"]["mlp"],
                            "instruct_lens": baseline["methods"]["instruct_lens"],
                        }
                        for name, variant in variants.items():
                            scores[name].append({field: _metric(variant["test_metrics"], field)
                                                 for field in ("auc", "hall_aupr", "hall_f1")})
                        counts["method"] = own["counts"]["test"]
                        counts["baseline"] = baseline["counts"]["test"]
                        if counts["method"] != counts["baseline"]:
                            raise ValueError(
                                f"Method/baseline test cohort differs for "
                                f"{model}/{dataset}/{position}/{protocol}: {counts}"
                            )
                    group_key = f"{model}/{dataset}/{position}/{protocol}"
                    report[group_key] = {
                        "test_counts": counts,
                        "detectors": {name: _aggregate(values) for name, values in scores.items()},
                    }
                    for name, values in report[group_key]["detectors"].items():
                        lines.append("| " + " | ".join((
                            model, dataset, position, protocol, name,
                            str(counts["method"] if name == "log_s" else counts["baseline"]),
                            _format(values["auc"]), _format(values["hall_aupr"]),
                            _format(values["hall_f1"]),
                        )) + " |")
    if expected != 40:
        raise AssertionError(f"Expected 40 model/dataset/position/label groups, got {expected}")
    (output / "complete_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "complete_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {expected} groups and {expected * len(SEEDS)} seed pairs -> {output / 'complete_results.md'}")


if __name__ == "__main__":
    main()
