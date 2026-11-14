#!/usr/bin/env python3
"""Write one audited table from completed TruthPrInt COCO and QA runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/truthprint_baseline"
MODELS = ("llava_1_5_7b", "qwen2_5_vl_7b", "internvl_2_5_8b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
TASKS = ("coco", "clevr_exist_9k", "amber_discriminative")
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
           "accuracy_real", "accuracy_hall")


def one_metrics(result: dict, task: str, dimension: str | None = None) -> dict[str, float]:
    if task == "coco":
        test = result["dual_threshold"]["test"]
        return {
            "auroc": float(test["auroc"]),
            "real_aupr": float(test["real"]["aupr"]),
            "hall_aupr": float(test["hall"]["aupr"]),
            "real_f1": float(test["real"]["f1"]),
            "hall_f1": float(test["hall"]["f1"]),
            "accuracy_real": float(test["accuracy_real_threshold"]),
            "accuracy_hall": float(test["accuracy_hall_threshold"]),
        }
    test = result["amber_dimensions"][dimension] if dimension else result["test"]
    return {
        "auroc": float(test["auroc"]),
        "real_aupr": float(test["real_aupr"]),
        "hall_aupr": float(test["hallucination_aupr"]),
        "real_f1": float(test["real"]["f1"]),
        "hall_f1": float(test["hallucination"]["f1"]),
        "accuracy_real": float(test["accuracy"]),
        "accuracy_hall": float(test["accuracy"]),
    }


def format_metric(row: dict, metric: str) -> str:
    return f"{100 * row[metric + '_mean']:.2f} ± {100 * row[metric + '_std']:.2f}"


def aggregate() -> list[dict]:
    rows = []
    for task in TASKS:
        for model in MODELS:
            results = []
            for seed in SEEDS:
                path = OUT / model / task / f"seed{seed}" / "result.json"
                result = json.loads(path.read_text())
                if int(result["seed"]) != seed or int(result.get("layer", 16)) != 16:
                    raise ValueError(f"Unexpected TruthPrInt run metadata: {path}")
                results.append(result)
            dimensions = (None,) if task != "amber_discriminative" else (
                None, "existence", "attribute", "relation")
            for dimension in dimensions:
                values = [one_metrics(result, task, dimension) for result in results]
                test_n = (results[0]["counts"]["test"] if dimension is None else
                          sum(int(results[0]["amber_dimensions"][dimension][label]["support"])
                              for label in ("real", "hallucination")))
                row = {
                    "model": model,
                    "task": task if dimension is None else f"amber_{dimension}",
                    "split": "8:1:1",
                    "test_n": test_n,
                    "threshold": ("validation_dual_class_f1" if task == "coco" else
                                  "validation_real_f1"),
                }
                for metric in METRICS:
                    row[metric + "_mean"] = mean(item[metric] for item in values)
                    row[metric + "_std"] = pstdev(item[metric] for item in values)
                rows.append(row)
    return rows


def main() -> None:
    rows = aggregate()
    csv_path = OUT / "complete_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# TruthPrInt hallucination detection",
        "",
        "[TruthPrInt](https://github.com/jinhaoduan/TruthPrInt) commit `c61af74`; post-block-16 previous-token hidden state; 128/64 ReLU MLP; balanced training; seeds 43/44/45.",
        "Each model is trained in-domain with `subspace_align=None`; validation TPR@1% FPR selects the checkpoint.",
        "COCO uses the shared 3,200/400/400 image split and validation-selected Real and Hall F1 thresholds.",
        "QA uses the existing 8:1:1 image split, prompt-last position, answer_correctness_all labels, and one validation-selected Real F1 threshold.",
        "AMBER's three categories reuse the overall AMBER model and threshold.",
        "",
        "| Model | Task | Test | AUROC | Hall AUPR | Real F1 | Hall F1 | Accuracy@Real | Accuracy@Hall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join((
            row["model"], row["task"], str(row["test_n"]),
            format_metric(row, "auroc"), format_metric(row, "hall_aupr"),
            format_metric(row, "real_f1"), format_metric(row, "hall_f1"),
            format_metric(row, "accuracy_real"), format_metric(row, "accuracy_hall"),
        )) + " |")
    (OUT / "complete_results.md").write_text("\n".join(lines) + "\n")
    print(csv_path)
    print(OUT / "complete_results.md")


if __name__ == "__main__":
    main()
