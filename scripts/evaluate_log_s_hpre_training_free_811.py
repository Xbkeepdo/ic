#!/usr/bin/env python3
"""Evaluate label-free scale fusion of log-s risk and F_log on COCO ENDAC.

The score construction never reads labels from the 3,200-image training split.
It reports both a fixed 0.5 operating point and thresholds selected on the
400-image validation split for comparison with the supervised detector table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from detection.baselines import (  # noqa: E402
    evaluate_detection_scores,
    select_detection_threshold,
)


MODELS = (
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
)
SPLIT_PATH = GF_ROOT / "outputs/coco4000_512_endac_811/image_splits.json"
FEATURE_ROOT = ROOT / "outputs/log_s_hpre_k_tau_ablation"
OUTPUT = ROOT / "outputs/log_s_hpre_training_free_811"
RISK_KEY = "log_s_hpre_k032_tau003"
F_KEY = "f_log"


def load_split() -> dict[str, set[int]]:
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split = {
        name: {int(value) for value in payload[name]}
        for name in ("train", "val", "test")
    }
    if {name: len(values) for name, values in split.items()} != {
        "train": 3200,
        "val": 400,
        "test": 400,
    }:
        raise ValueError("Expected the shared 3200/400/400 image split")
    return split


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_layer_ecdf(train: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(train, dtype=np.float64), axis=0)


def apply_layer_ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    count = int(reference.shape[0])
    for layer in range(reference.shape[1]):
        output[:, layer] = (
            np.searchsorted(reference[:, layer], values[:, layer], side="right")
            - 0.5
        ) / count
    return np.clip(output, 0.0, 1.0)


def fit_scalar_ecdf(train: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(train, dtype=np.float64).reshape(-1))


def apply_scalar_ecdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    output = (
        np.searchsorted(reference, values, side="right") - 0.5
    ) / len(reference)
    return np.clip(output, 0.0, 1.0)


def build_scores(
    risk: np.ndarray,
    f_log: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return hallucination scores; high risk and low F_log indicate Hall."""

    risk_reference = fit_layer_ecdf(risk[train_mask])
    f_reference = fit_layer_ecdf(f_log[train_mask])
    risk_percentile = apply_layer_ecdf(risk_reference, risk)
    f_percentile = apply_layer_ecdf(f_reference, f_log)

    risk_mean = risk.mean(axis=1)
    f_mean = f_log.mean(axis=1)
    risk_scalar_reference = fit_scalar_ecdf(risk_mean[train_mask])
    f_scalar_reference = fit_scalar_ecdf(f_mean[train_mask])

    risk_train = risk[train_mask]
    f_train = f_log[train_mask]
    risk_mean_layer = risk_train.mean(axis=0)
    f_mean_layer = f_train.mean(axis=0)
    risk_std_layer = np.maximum(risk_train.std(axis=0), 1e-8)
    f_std_layer = np.maximum(f_train.std(axis=0), 1e-8)

    risk_median = np.median(risk_train, axis=0)
    f_median = np.median(f_train, axis=0)
    risk_iqr = (
        np.percentile(risk_train, 75, axis=0)
        - np.percentile(risk_train, 25, axis=0)
    ) / 1.349
    f_iqr = (
        np.percentile(f_train, 75, axis=0)
        - np.percentile(f_train, 25, axis=0)
    ) / 1.349
    risk_iqr = np.maximum(risk_iqr, 1e-8)
    f_iqr = np.maximum(f_iqr, 1e-8)

    risk_z = np.clip((risk - risk_mean_layer) / risk_std_layer, -5.0, 5.0)
    f_z = np.clip((f_log - f_mean_layer) / f_std_layer, -5.0, 5.0)
    risk_robust = np.clip((risk - risk_median) / risk_iqr, -5.0, 5.0)
    f_robust = np.clip((f_log - f_median) / f_iqr, -5.0, 5.0)

    return {
        "risk_layer_ecdf": risk_percentile.mean(axis=1),
        "f_log_layer_ecdf": (1.0 - f_percentile).mean(axis=1),
        "fusion_layer_ecdf": 0.5
        * (risk_percentile.mean(axis=1) + (1.0 - f_percentile).mean(axis=1)),
        "fusion_scalar_ecdf": 0.5
        * (
            apply_scalar_ecdf(risk_scalar_reference, risk_mean)
            + 1.0
            - apply_scalar_ecdf(f_scalar_reference, f_mean)
        ),
        "fusion_layer_zscore": stable_sigmoid(
            risk_z.mean(axis=1) - f_z.mean(axis=1)
        ),
        "fusion_layer_robust_zscore": stable_sigmoid(
            risk_robust.mean(axis=1) - f_robust.mean(axis=1)
        ),
    }


def compact_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(report["accuracy"]),
        "auroc": float(report["real_positive"]["auc"]),
        "real_precision": float(report["real_positive"]["precision"]),
        "real_recall": float(report["real_positive"]["recall"]),
        "real_f1": float(report["real_positive"]["f1"]),
        "real_aupr": float(report["real_positive"]["aupr"]),
        "hall_precision": float(report["hallucination_positive"]["precision"]),
        "hall_recall": float(report["hallucination_positive"]["recall"]),
        "hall_f1": float(report["hallucination_positive"]["f1"]),
        "hall_aupr": float(report["hallucination_positive"]["aupr"]),
    }


def evaluate_score(
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    hall_score: np.ndarray,
) -> dict[str, Any]:
    val_labels = labels[masks["val"]]
    val_scores = hall_score[masks["val"]]
    test_labels = labels[masks["test"]]
    test_scores = hall_score[masks["test"]]

    fixed = evaluate_detection_scores(
        test_labels, test_scores, 0.5, positive_class="hallucination"
    )
    real_threshold = select_detection_threshold(
        val_labels, val_scores, positive_class="real"
    )
    hall_threshold = select_detection_threshold(
        val_labels, val_scores, positive_class="hallucination"
    )
    calibrated_real = evaluate_detection_scores(
        test_labels, test_scores, real_threshold, positive_class="real"
    )
    calibrated_hall = evaluate_detection_scores(
        test_labels, test_scores, hall_threshold, positive_class="hallucination"
    )
    real_metrics = compact_metrics(calibrated_real)
    hall_metrics = compact_metrics(calibrated_hall)
    calibrated = dict(real_metrics)
    for key in ("hall_precision", "hall_recall", "hall_f1", "hall_aupr"):
        calibrated[key] = hall_metrics[key]
    calibrated["accuracy_real_threshold"] = real_metrics["accuracy"]
    calibrated["accuracy_hall_threshold"] = hall_metrics["accuracy"]
    calibrated.pop("accuracy", None)

    return {
        "thresholds": {
            "fixed": 0.5,
            "validation_real_f1": float(real_threshold),
            "validation_hall_f1": float(hall_threshold),
        },
        "fixed_0.5": compact_metrics(fixed),
        "validation_calibrated_dual": calibrated,
        "test_counts": {
            "tokens": int(masks["test"].sum()),
            "hall": int((test_labels == 0).sum()),
            "real": int((test_labels == 1).sum()),
        },
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


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Training-free fusion of log-s risk and F_log (COCO ENDAC 8:1:1)",
        "",
        "- Features: `log_s_hpre_k032_tau003` and `f_log`.",
        "- Scale fitting uses only unlabeled features from the 3,200-image train split.",
        "- Score direction is fixed from the definitions: larger risk and smaller F_log indicate Hall.",
        "- ECDF fusion uses equal 1:1 component weights and equal weights over layers.",
        "- Fixed 0.5 uses no labels at any stage; validation-calibrated thresholds use the 400-image validation labels but fit no classifier.",
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
        "experiment": "log_s_hpre_training_free_811",
        "split": "shared_image_level_3200_400_400",
        "split_path": str(SPLIT_PATH),
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
        if risk.shape != f_log.shape or risk.shape[0] != labels.size:
            raise ValueError(f"{model}: feature shape mismatch")
        masks = {
            name: np.isin(image_ids, list(image_set))
            for name, image_set in split.items()
        }
        if sum(int(mask.sum()) for mask in masks.values()) != labels.size:
            raise ValueError(f"{model}: split does not cover all tokens")

        scores = build_scores(risk, f_log, masks["train"])
        model_result = {
            "feature_path": str(feature_path),
            "tokens": {name: int(mask.sum()) for name, mask in masks.items()},
            "layers": int(risk.shape[1]),
            "scores": {},
        }
        for score_name, hall_score in scores.items():
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
