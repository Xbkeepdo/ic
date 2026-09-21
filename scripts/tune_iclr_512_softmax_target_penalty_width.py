#!/usr/bin/env python3
"""Tune one-hidden-layer width on training images, then evaluate held-out test."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_iclr_endac_attention_i_fusion import (
    FUSIONS, MODELS, ROOT, SEEDS, _matrix, _probe_config, load_aligned,
)
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe
from utils.config_utils import load_config

OUT = ROOT / "outputs/iclr_512_softmax_target_penalty_width_tuning"
RISK = "softmax_target_penalty"
WIDTHS = (32, 64, 128, 256, 512)
VAL_IMAGE_COUNT = 640
SPLIT_SEED = 20260918


def metrics_at(path: Path) -> dict[str, float]:
    metrics = json.loads(path.read_text())
    report = metrics["threshold_reports"]["train_f1"]["test_metrics"]
    return {
        "real_f1": float(report["real_positive"]["f1"]),
        "real_auc": float(report["real_positive"]["auc"]),
        "hall_aupr": float(report["hallucination_positive"]["aupr"]),
    }


def summarize(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(np.mean([row[name] for row in values])),
               "std": float(np.std([row[name] for row in values]))}
        for name in ("real_f1", "real_auc", "hall_aupr")
    }


def fit_case(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    width: int,
    seed: int,
    config: dict,
    device: torch.device,
    folder: Path,
) -> dict[str, float]:
    path = folder / "metrics.json"
    if not path.exists():
        probe = replace(
            _probe_config(config, seed),
            hidden_sizes=(width,),
            batch_norm=False,
            standardize_input=True,
        )
        metrics = train_and_evaluate_probe(
            X_train=features[train_mask],
            y_train=labels[train_mask],
            X_val=np.empty((0, features.shape[1]), dtype=np.float32),
            y_val=np.empty((0,), dtype=np.int32),
            X_test=features[eval_mask],
            y_test=labels[eval_mask],
            config=probe,
            device=device,
            output_dir=str(folder),
        )
        path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
    else:
        saved = json.loads((folder / "config.json").read_text())
        if (saved["hidden_sizes"] != [width] or saved["seed"] != seed
                or saved.get("batch_norm") is not False
                or saved.get("standardize_input") is not True):
            raise ValueError(f"Existing case has different protocol: {folder}")
    return metrics_at(path)


def tune(model: str, device: str, out: Path) -> None:
    torch.set_num_threads(1)
    audit_path = ROOT / "outputs/iclr_512_endac_attention_i_fusion" / model / "alignment.json"
    expected = json.loads(audit_path.read_text())
    data, audit = load_aligned(model, expected["risk_run_name"], True)
    if audit != expected:
        raise ValueError(f"{model}: aligned cohort changed")
    model_out = out / model
    model_out.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "outputs" / model / audit["risk_run_name"] / "image_splits.json"
    original = json.loads(split_path.read_text())
    train_images = np.asarray(sorted(map(int, original["train"])), dtype=np.int64)
    test_images = set(map(int, original["test"]))
    if len(train_images) != 3200 or len(test_images) != 800 or set(train_images) & test_images:
        raise ValueError(f"{model}: invalid original image split")
    rng = np.random.default_rng(SPLIT_SEED)
    permuted = rng.permutation(train_images)
    val_images = sorted(map(int, permuted[:VAL_IMAGE_COUNT]))
    fit_images = sorted(map(int, permuted[VAL_IMAGE_COUNT:]))
    image_hash = hashlib.sha256(",".join(map(str, train_images)).encode()).hexdigest()
    split = {
        "source_train_image_sha256": image_hash,
        "seed": SPLIT_SEED,
        "fit_images": fit_images,
        "validation_images": val_images,
        "test_images": sorted(test_images),
    }
    output_split = model_out / "width_selection_image_split.json"
    if output_split.exists() and json.loads(output_split.read_text()) != split:
        raise ValueError(f"{model}: existing width-selection split differs")
    output_split.write_text(json.dumps(split, indent=2) + "\n")

    ids = data["image_id"]
    fit_mask = data["train_mask"] & np.isin(ids, fit_images)
    val_mask = data["train_mask"] & np.isin(ids, val_images)
    if np.any(fit_mask & val_mask) or np.any((fit_mask | val_mask) != data["train_mask"]):
        raise ValueError(f"{model}: fit/validation masks do not partition original train")
    if np.any(np.isin(ids[data["test_mask"]], fit_images + val_images)):
        raise ValueError(f"{model}: held-out test images entered width selection")
    for mask in (fit_mask, val_mask, data["test_mask"]):
        if set(np.unique(data["label"][mask])) != {0, 1}:
            raise ValueError(f"{model}: each split needs both labels")

    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    output_device = torch.device(device)
    model_summary = {
        "model": model,
        "risk": RISK,
        "candidate_widths": list(WIDTHS),
        "selection_metric": "mean validation Hall AUPR; tie: Real AUC, then smaller width",
        "split": {
            "fit_images": len(fit_images),
            "validation_images": len(val_images),
            "test_images": len(test_images),
            "fit_targets": int(fit_mask.sum()),
            "validation_targets": int(val_mask.sum()),
            "test_targets": int(data["test_mask"].sum()),
            "source_train_image_sha256": image_hash,
        },
        "feature_sets": {},
    }
    for fusion in FUSIONS:
        features = _matrix(data, RISK, fusion)
        candidates = []
        for width in WIDTHS:
            scores = []
            for seed in SEEDS:
                folder = model_out / "validation" / fusion / f"width{width}" / f"seed{seed}"
                scores.append(fit_case(
                    features=features, labels=data["label"], train_mask=fit_mask,
                    eval_mask=val_mask, width=width, seed=seed, config=config,
                    device=output_device, folder=folder,
                ))
                print("VALIDATION", model, fusion, width, seed, flush=True)
            candidates.append({"width": width, "metrics": summarize(scores)})
        best = max(candidates, key=lambda row: (
            row["metrics"]["hall_aupr"]["mean"],
            row["metrics"]["real_auc"]["mean"],
            -row["width"],
        ))
        selection = {
            "fusion": fusion,
            "candidates": candidates,
            "selected_width": best["width"],
            "validation_metrics": best["metrics"],
        }
        selection_path = model_out / f"{fusion}_selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
            raise ValueError(f"{model}/{fusion}: existing width selection differs")
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        test_scores = []
        for seed in SEEDS:
            folder = model_out / "final" / fusion / f"seed{seed}"
            test_scores.append(fit_case(
                features=features, labels=data["label"],
                train_mask=data["train_mask"], eval_mask=data["test_mask"],
                width=best["width"], seed=seed, config=config,
                device=output_device, folder=folder,
            ))
            print("FINAL", model, fusion, best["width"], seed, flush=True)
        model_summary["feature_sets"][fusion] = {
            **selection,
            "test_metrics": summarize(test_scores),
        }
    (model_out / "summary.json").write_text(json.dumps(model_summary, indent=2) + "\n")
    print("SUMMARY", model_out / "summary.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    tune(args.model, args.device, args.output_root.resolve())


if __name__ == "__main__":
    main()
