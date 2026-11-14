#!/usr/bin/env python3
"""Evaluate last-layer risk/F training-free scores on COCO ENDAC 8:1:1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.evaluate_log_s_hpre_training_free_811 import (  # noqa: E402
    evaluate_score,
    load_split,
    percent,
    stable_sigmoid,
)


MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
FEATURE_ROOT = ROOT / "outputs/log_s_hpre_k_tau_ablation"
OUTPUT = ROOT / "outputs/log_s_hpre_training_free_last_layer_811"
RISK_KEY = "log_s_hpre_k032_tau003"
F_KEY = "f_log"
EPSILON = 1e-12


def standardize(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    reference = values[train_mask]
    scale = max(float(reference.std()), 1e-8)
    return np.clip((values - float(reference.mean())) / scale, -5.0, 5.0)


def robust_standardize(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    reference = values[train_mask]
    median = float(np.median(reference))
    scale = float(np.percentile(reference, 75) - np.percentile(reference, 25))
    scale = max(scale / 1.349, 1e-8)
    return np.clip((values - median) / scale, -5.0, 5.0)


def scalar_ecdf(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(values[train_mask], dtype=np.float64))
    ranks = np.searchsorted(reference, values, side="right") - 0.5
    return np.clip(ranks / len(reference), 0.0, 1.0)


def build_last_layer_scores(
    risk: np.ndarray,
    f_log: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    risk_last = np.asarray(risk[:, -1], dtype=np.float64)
    f_last = np.asarray(f_log[:, -1], dtype=np.float64)
    if np.any(f_last <= 0):
        raise ValueError("F_last must be positive before log(F_last)")
    log_f_last = np.log(f_last + EPSILON)

    risk_z = standardize(risk_last, train_mask)
    f_z = standardize(f_last, train_mask)
    log_f_z = standardize(log_f_last, train_mask)
    risk_robust = robust_standardize(risk_last, train_mask)
    log_f_robust = robust_standardize(log_f_last, train_mask)

    risk_rank = scalar_ecdf(risk_last, train_mask)
    f_rank = scalar_ecdf(f_last, train_mask)

    return {
        "last_risk_zscore": stable_sigmoid(risk_z),
        "last_f_raw_zscore": stable_sigmoid(-f_z),
        "last_f_log_zscore": stable_sigmoid(-log_f_z),
        "last_fusion_raw_zscore": stable_sigmoid(risk_z - f_z),
        "last_fusion_log_zscore": stable_sigmoid(risk_z - log_f_z),
        "last_fusion_log_robust_zscore": stable_sigmoid(
            risk_robust - log_f_robust
        ),
        "last_fusion_ecdf": 0.5 * (risk_rank + 1.0 - f_rank),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "score",
        "protocol",
        "threshold_real",
        "threshold_hall",
        "auroc",
        "real_precision",
        "real_recall",
        "real_f1",
        "real_aupr",
        "hall_precision",
        "hall_recall",
        "hall_f1",
        "hall_aupr",
        "accuracy_real_threshold",
        "accuracy_hall_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Last-layer training-free risk/F results (COCO ENDAC 8:1:1)",
        "",
        "- Only the final decoder layer of `log_s_hpre_k032_tau003` and `f_log` is used.",
        "- Scale statistics use unlabeled features from the 3,200-image train split.",
        "- `log(F)` means `log(F + 1e-12)`; all observed final-layer F values are positive.",
        "- A larger risk and a smaller F are fixed as Hall evidence.",
        "- Fixed 0.5 is label-free; calibrated thresholds use only the 400-image validation labels.",
        "- The 400-image test split is evaluation-only.",
    ]
    display = dict(MODELS)
    for protocol, title in (
        ("validation_calibrated_dual", "Validation-calibrated Hall threshold"),
        ("fixed_0.5", "Strict label-free fixed threshold 0.5"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | Score | AUROC | Real F1 | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUPR |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model, _ in MODELS:
            for score_name, payload in result["models"][model]["scores"].items():
                metrics = payload[protocol]
                lines.append(
                    f"| {display[model]} | `{score_name}` | "
                    f"{percent(metrics['auroc'])} | {percent(metrics['real_f1'])} | "
                    f"{percent(metrics['real_aupr'])} | "
                    f"{percent(metrics['hall_precision'])} | "
                    f"{percent(metrics['hall_recall'])} | "
                    f"{percent(metrics['hall_f1'])} | "
                    f"{percent(metrics['hall_aupr'])} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split = load_split()
    result: dict[str, Any] = {
        "experiment": "log_s_hpre_training_free_last_layer_811",
        "split": "shared_image_level_3200_400_400",
        "risk_key": RISK_KEY,
        "f_key": F_KEY,
        "models": {},
    }
    csv_rows: list[dict[str, Any]] = []

    for model, _display in MODELS:
        feature_path = FEATURE_ROOT / model / "compact_features.npz"
        with np.load(feature_path, allow_pickle=False) as saved:
            risk = np.asarray(saved[RISK_KEY], dtype=np.float64)
            f_log = np.asarray(saved[F_KEY], dtype=np.float64)
            labels = np.asarray(saved["label"], dtype=np.int64)
            image_ids = np.asarray(saved["image_id"], dtype=np.int64)
        masks = {
            name: np.isin(image_ids, list(image_set))
            for name, image_set in split.items()
        }
        if sum(int(mask.sum()) for mask in masks.values()) != labels.size:
            raise ValueError(f"{model}: split does not cover all tokens")

        model_result = {
            "feature_path": str(feature_path),
            "last_layer_index": int(risk.shape[1] - 1),
            "tokens": {name: int(mask.sum()) for name, mask in masks.items()},
            "scores": {},
        }
        for score_name, hall_score in build_last_layer_scores(
            risk, f_log, masks["train"]
        ).items():
            if not np.isfinite(hall_score).all():
                raise ValueError(f"{model}/{score_name}: non-finite score")
            payload = evaluate_score(labels, masks, hall_score)
            model_result["scores"][score_name] = payload
            for protocol in ("fixed_0.5", "validation_calibrated_dual"):
                metrics = payload[protocol]
                thresholds = payload["thresholds"]
                row = {
                    "model": model,
                    "score": score_name,
                    "protocol": protocol,
                    "threshold_real": (
                        thresholds["fixed"]
                        if protocol == "fixed_0.5"
                        else thresholds["validation_real_f1"]
                    ),
                    "threshold_hall": (
                        thresholds["fixed"]
                        if protocol == "fixed_0.5"
                        else thresholds["validation_hall_f1"]
                    ),
                    "accuracy_real_threshold": metrics.get(
                        "accuracy_real_threshold", metrics.get("accuracy")
                    ),
                    "accuracy_hall_threshold": metrics.get(
                        "accuracy_hall_threshold", metrics.get("accuracy")
                    ),
                }
                row.update(
                    {
                        key: metrics[key]
                        for key in (
                            "auroc",
                            "real_precision",
                            "real_recall",
                            "real_f1",
                            "real_aupr",
                            "hall_precision",
                            "hall_recall",
                            "hall_f1",
                            "hall_aupr",
                        )
                    }
                )
                csv_rows.append(row)
        result["models"][model] = model_result

    (OUTPUT / "complete_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT / "complete_results.csv", csv_rows)
    write_markdown(OUTPUT / "complete_results.md", result)
    print(OUTPUT / "complete_results.md")


if __name__ == "__main__":
    main()
