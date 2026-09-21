#!/usr/bin/env python3
"""Tune the one-layer log_s_hpre + log-residual probe without test leakage."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402


SOURCE = ROOT / "outputs/linear_cost_log_s_risk_four_f_fusion"
OUT = ROOT / "outputs/log_s_hpre_log_residual_fusion_tuning"
MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
    "minigpt4_7b",
    "shikra_7b",
)
SEEDS = (43, 44, 45)
SPLIT_SEED = 20260921
SCREEN_SEED = 43
TOP_K = 8


@dataclass(frozen=True)
class Candidate:
    name: str
    width: int
    batch_size: int
    dropout: float
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int


BASE = dict(width=128, batch_size=256, dropout=0.3,
            learning_rate=1e-3, weight_decay=1e-5,
            early_stopping_patience=10)


def _candidate(name: str, **changes: float | int) -> Candidate:
    values = {**BASE, **changes}
    return Candidate(name=name, **values)


CANDIDATES = (
    _candidate("baseline"),
    _candidate("width16", width=16),
    _candidate("width32", width=32),
    _candidate("width64", width=64),
    _candidate("width256", width=256),
    _candidate("width512", width=512),
    _candidate("batch64", batch_size=64),
    _candidate("batch128", batch_size=128),
    _candidate("batch512", batch_size=512),
    _candidate("dropout00", dropout=0.0),
    _candidate("dropout01", dropout=0.1),
    _candidate("dropout02", dropout=0.2),
    _candidate("dropout04", dropout=0.4),
    _candidate("dropout05", dropout=0.5),
    _candidate("lr1e-4", learning_rate=1e-4),
    _candidate("lr3e-4", learning_rate=3e-4),
    _candidate("lr5e-4", learning_rate=5e-4),
    _candidate("lr2e-3", learning_rate=2e-3),
    _candidate("lr3e-3", learning_rate=3e-3),
    _candidate("wd0", weight_decay=0.0),
    _candidate("wd1e-6", weight_decay=1e-6),
    _candidate("wd1e-4", weight_decay=1e-4),
    _candidate("wd1e-3", weight_decay=1e-3),
    _candidate("patience5", early_stopping_patience=5),
    _candidate("patience20", early_stopping_patience=20),
    _candidate("mix_h256_b128_d02_lr5e-4", width=256, batch_size=128,
               dropout=0.2, learning_rate=5e-4),
    _candidate("mix_h256_b256_d01_lr5e-4_wd0", width=256, dropout=0.1,
               learning_rate=5e-4, weight_decay=0.0),
    _candidate("mix_h512_b128_d03_lr3e-4", width=512, batch_size=128,
               learning_rate=3e-4),
    _candidate("mix_h64_b128_d00_lr1e-3_wd0", width=64, batch_size=128,
               dropout=0.0, weight_decay=0.0),
    _candidate("mix_h64_b256_d01_lr3e-4_wd0", width=64, batch_size=256,
               dropout=0.1, learning_rate=3e-4, weight_decay=0.0),
    _candidate("mix_h128_b128_d01_lr5e-4_wd0", width=128, batch_size=128,
               dropout=0.1, learning_rate=5e-4, weight_decay=0.0),
    _candidate("mix_h256_b256_d02_lr3e-4_wd1e-6", width=256, batch_size=256,
               dropout=0.2, learning_rate=3e-4, weight_decay=1e-6),
    _candidate("mix_h256_b512_d00_lr1e-3_wd0", width=256, batch_size=512,
               dropout=0.0, learning_rate=1e-3, weight_decay=0.0),
    _candidate("mix_h512_b256_d01_lr5e-4_wd1e-6", width=512, batch_size=256,
               dropout=0.1, learning_rate=5e-4, weight_decay=1e-6),
    _candidate("mix_h512_b512_d03_lr1e-3", width=512, batch_size=512,
               dropout=0.3, learning_rate=1e-3),
    _candidate("mix_h32_b128_d00_lr2e-3_wd0", width=32, batch_size=128,
               dropout=0.0, learning_rate=2e-3, weight_decay=0.0),
    _candidate("mix_h128_b512_d00_lr2e-3_wd0", width=128, batch_size=512,
               dropout=0.0, learning_rate=2e-3, weight_decay=0.0),
    _candidate("mix_h256_b1024_d01_lr3e-3_wd0", width=256, batch_size=1024,
               dropout=0.1, learning_rate=3e-3, weight_decay=0.0),
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2) + "\n")
    os.replace(temporary, path)


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
        metric: {
            "mean": float(np.mean([row[metric] for row in values])),
            "std": float(np.std([row[metric] for row in values])),
        }
        for metric in ("real_f1", "real_auc", "hall_aupr")
    }


def fit_case(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    candidate: Candidate,
    seed: int,
    root_config: dict,
    device: torch.device,
    folder: Path,
) -> dict[str, float]:
    metrics_path = folder / "metrics.json"
    expected = {
        "hidden_sizes": [candidate.width],
        "batch_size": candidate.batch_size,
        "dropout": candidate.dropout,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "early_stopping_patience": candidate.early_stopping_patience,
        "seed": seed,
        "batch_norm": False,
        "standardize_input": True,
    }
    if metrics_path.exists():
        saved = json.loads((folder / "config.json").read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing case has a different configuration: {folder}")
        return metrics_at(metrics_path)
    probe = replace(
        _probe_config(root_config, seed),
        hidden_sizes=(candidate.width,),
        batch_size=candidate.batch_size,
        dropout=candidate.dropout,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        early_stopping_patience=candidate.early_stopping_patience,
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
    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
    return metrics_at(metrics_path)


def load_data(model: str) -> dict[str, np.ndarray]:
    path = SOURCE / model / "compact_features.npz"
    with np.load(path, allow_pickle=False) as saved:
        data = {key: np.asarray(saved[key]) for key in saved.files}
    required = {
        "log_s_hpre", "log_residual_fusion", "label", "image_id",
        "train_mask", "strict_test_mask",
    }
    if not required.issubset(data):
        raise ValueError(f"{model}: missing compact feature keys: {sorted(required - set(data))}")
    features = np.concatenate(
        (data["log_s_hpre"], data["log_residual_fusion"]), axis=1
    ).astype(np.float32)
    if data["log_s_hpre"].shape != data["log_residual_fusion"].shape:
        raise ValueError(f"{model}: risk/F shapes differ")
    if not np.isfinite(features).all():
        raise ValueError(f"{model}: nonfinite tuning input")
    data["features"] = features
    data["test_mask"] = np.asarray(data["strict_test_mask"], dtype=bool)
    return data


def make_split(model: str, data: dict[str, np.ndarray], model_out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    ids = data["image_id"]
    train_images = np.unique(ids[data["train_mask"]])
    test_images = np.unique(ids[data["test_mask"]])
    if np.intersect1d(train_images, test_images).size:
        raise ValueError(f"{model}: train/test image overlap")
    rng = np.random.default_rng(SPLIT_SEED)
    permuted = rng.permutation(np.sort(train_images))
    validation_count = max(1, round(0.2 * len(train_images)))
    val_images = np.sort(permuted[:validation_count])
    fit_images = np.sort(permuted[validation_count:])
    fit_mask = data["train_mask"] & np.isin(ids, fit_images)
    val_mask = data["train_mask"] & np.isin(ids, val_images)
    if np.any(fit_mask & val_mask) or not np.array_equal(fit_mask | val_mask, data["train_mask"]):
        raise ValueError(f"{model}: internal split does not partition training rows")
    split = {
        "seed": SPLIT_SEED,
        "source_train_image_sha256": hashlib.sha256(
            ",".join(map(str, np.sort(train_images))).encode()
        ).hexdigest(),
        "fit_images": fit_images.tolist(),
        "validation_images": val_images.tolist(),
        "test_images": np.sort(test_images).tolist(),
        "counts": {
            "fit_images": len(fit_images),
            "validation_images": len(val_images),
            "test_images": len(test_images),
            "fit_targets": int(fit_mask.sum()),
            "validation_targets": int(val_mask.sum()),
            "test_targets": int(data["test_mask"].sum()),
        },
    }
    split_path = model_out / "selection_image_split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError(f"{model}: existing selection split differs")
    atomic_json(split_path, split)
    return fit_mask, val_mask, split


def candidate_key(row: dict) -> tuple[float, float, int, float]:
    metrics = row["validation_metrics"]
    cfg = row["config"]
    return (
        metrics["real_auc"]["mean"],
        metrics["hall_aupr"]["mean"],
        -int(cfg["width"]),
        -float(cfg["dropout"]),
    )


def old_baseline(model: str) -> dict[str, dict[str, float]]:
    values = [
        metrics_at(
            SOURCE / model / "log_s_hpre__log_residual_fusion"
            / f"seed{seed}" / "metrics.json"
        )
        for seed in SEEDS
    ]
    return summarize(values)


def write_model_report(model_out: Path, summary: dict) -> None:
    rows = sorted(summary["validation_candidates"], key=candidate_key, reverse=True)
    csv_path = model_out / "validation_candidates.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank", "candidate", "width", "batch_size", "dropout", "learning_rate",
            "weight_decay", "early_stopping_patience", "seeds",
            "val_real_auc_mean", "val_real_auc_std",
            "val_hall_aupr_mean", "val_hall_aupr_std",
        ])
        for rank, row in enumerate(rows, 1):
            cfg, metrics = row["config"], row["validation_metrics"]
            writer.writerow([
                rank, cfg["name"], cfg["width"], cfg["batch_size"], cfg["dropout"],
                cfg["learning_rate"], cfg["weight_decay"], cfg["early_stopping_patience"],
                ",".join(map(str, row["seeds"])),
                metrics["real_auc"]["mean"], metrics["real_auc"]["std"],
                metrics["hall_aupr"]["mean"], metrics["hall_aupr"]["std"],
            ])
    selected = summary["selected_config"]
    test = summary["test_metrics"]
    old = summary["old_test_baseline"]
    lines = [
        f"# {summary['model']}: log_s_hpre + log_residual_fusion tuning",
        "",
        "Selection uses a fixed 80/20 image split inside the original training partition. "
        "The strict test partition is used only after selecting a configuration.",
        "All inputs use training-only column z-score; BatchNorm is disabled; all Linear layers use "
        "Kaiming uniform initialization for ReLU.",
        "",
        "## Selected configuration",
        "",
        f"- Hidden width: {selected['width']}",
        f"- Batch size: {selected['batch_size']}",
        f"- Dropout: {selected['dropout']}",
        f"- Learning rate: {selected['learning_rate']}",
        f"- Weight decay: {selected['weight_decay']}",
        f"- Early-stopping patience: {selected['early_stopping_patience']}",
        "",
        "| Result | Real F1 | Real AUC | Hall AUPR |",
        "|---|---:|---:|---:|",
        f"| Tuned test | {test['real_f1']['mean']:.4f}±{test['real_f1']['std']:.4f} | "
        f"{test['real_auc']['mean']:.4f}±{test['real_auc']['std']:.4f} | "
        f"{test['hall_aupr']['mean']:.4f}±{test['hall_aupr']['std']:.4f} |",
        f"| Original test baseline | {old['real_f1']['mean']:.4f}±{old['real_f1']['std']:.4f} | "
        f"{old['real_auc']['mean']:.4f}±{old['real_auc']['std']:.4f} | "
        f"{old['hall_aupr']['mean']:.4f}±{old['hall_aupr']['std']:.4f} |",
        f"| Delta | {test['real_f1']['mean'] - old['real_f1']['mean']:+.4f} | "
        f"{test['real_auc']['mean'] - old['real_auc']['mean']:+.4f} | "
        f"{test['hall_aupr']['mean'] - old['hall_aupr']['mean']:+.4f} |",
        "",
        "## Validation ranking",
        "",
        "The screening seed is 43. The top eight screening candidates are ranked below using seeds 43–45.",
        "",
        "| Rank | Candidate | Width | Batch | Dropout | LR | WD | Patience | Val AUC | Val Hall AUPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        cfg, metric = row["config"], row["validation_metrics"]
        lines.append(
            f"| {rank} | {cfg['name']} | {cfg['width']} | {cfg['batch_size']} | "
            f"{cfg['dropout']:.2f} | {cfg['learning_rate']:.1e} | {cfg['weight_decay']:.1e} | "
            f"{cfg['early_stopping_patience']} | "
            f"{metric['real_auc']['mean']:.4f}±{metric['real_auc']['std']:.4f} | "
            f"{metric['hall_aupr']['mean']:.4f}±{metric['hall_aupr']['std']:.4f} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")


def tune(model: str, device_name: str, out: Path) -> None:
    torch.set_num_threads(1)
    model_out = out / model
    model_out.mkdir(parents=True, exist_ok=True)
    data = load_data(model)
    fit_mask, val_mask, split = make_split(model, data, model_out)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    device = torch.device(device_name)
    features = data["features"]
    labels = data["label"]
    progress_path = model_out / "progress.json"
    completed = 0
    total = len(CANDIDATES) + 2 * TOP_K + len(SEEDS)

    screened = []
    for candidate in CANDIDATES:
        folder = model_out / "validation" / candidate.name / f"seed{SCREEN_SEED}"
        score = fit_case(
            features=features, labels=labels, train_mask=fit_mask, eval_mask=val_mask,
            candidate=candidate, seed=SCREEN_SEED, root_config=root_config,
            device=device, folder=folder,
        )
        screened.append((candidate, score))
        completed += 1
        atomic_json(progress_path, {"phase": "screen", "completed": completed, "total": total,
                                    "candidate": candidate.name, "seed": SCREEN_SEED})
        print("SCREEN", model, candidate.name, score["real_auc"], flush=True)

    top = sorted(screened, key=lambda pair: (
        pair[1]["real_auc"], pair[1]["hall_aupr"], -pair[0].width
    ), reverse=True)[:TOP_K]
    validation_rows = []
    for candidate, seed43_score in top:
        scores = [seed43_score]
        for seed in SEEDS[1:]:
            folder = model_out / "validation" / candidate.name / f"seed{seed}"
            scores.append(fit_case(
                features=features, labels=labels, train_mask=fit_mask, eval_mask=val_mask,
                candidate=candidate, seed=seed, root_config=root_config,
                device=device, folder=folder,
            ))
            completed += 1
            atomic_json(progress_path, {"phase": "confirm", "completed": completed, "total": total,
                                        "candidate": candidate.name, "seed": seed})
            print("CONFIRM", model, candidate.name, seed, scores[-1]["real_auc"], flush=True)
        validation_rows.append({
            "config": asdict(candidate),
            "seeds": list(SEEDS),
            "validation_metrics": summarize(scores),
        })
    best = max(validation_rows, key=candidate_key)
    selected = Candidate(**best["config"])

    test_scores = []
    for seed in SEEDS:
        folder = model_out / "final" / selected.name / f"seed{seed}"
        test_scores.append(fit_case(
            features=features, labels=labels, train_mask=data["train_mask"],
            eval_mask=data["test_mask"], candidate=selected, seed=seed,
            root_config=root_config, device=device, folder=folder,
        ))
        completed += 1
        atomic_json(progress_path, {"phase": "final", "completed": completed, "total": total,
                                    "candidate": selected.name, "seed": seed})
        print("FINAL", model, selected.name, seed, test_scores[-1]["real_auc"], flush=True)

    summary = {
        "schema": "log-s-hpre-log-residual-fusion-tuning-v1",
        "model": model,
        "feature": "concat(log_s_hpre, log_residual_fusion)",
        "selection_protocol": {
            "screen": f"all {len(CANDIDATES)} candidates with seed {SCREEN_SEED}",
            "confirm": f"top {TOP_K} by validation Real AUC with seeds {list(SEEDS)}",
            "select": "mean validation Real AUC; tie Hall AUPR, smaller width, lower dropout",
            "final": "retrain selected configuration on the full training partition; evaluate the strict test partition",
            "input": "training-only column z-score; no BatchNorm",
            "initialization": "kaiming_uniform_relu for every Linear layer; zero bias",
        },
        "split": split["counts"] | {
            "seed": split["seed"],
            "source_train_image_sha256": split["source_train_image_sha256"],
        },
        "screening": [
            {"config": asdict(candidate), "seed": SCREEN_SEED, "metrics": score}
            for candidate, score in screened
        ],
        "validation_candidates": validation_rows,
        "selected_config": asdict(selected),
        "selected_validation_metrics": best["validation_metrics"],
        "test_metrics": summarize(test_scores),
        "old_test_baseline": old_baseline(model),
    }
    atomic_json(model_out / "summary.json", summary)
    write_model_report(model_out, summary)
    atomic_json(progress_path, {"phase": "complete", "completed": total, "total": total,
                                "candidate": selected.name})
    print("SUMMARY", model_out / "summary.md", flush=True)


def write_aggregate(out: Path) -> None:
    summaries = []
    for model in MODELS:
        path = out / model / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))
    lines = [
        "# log_s_hpre + log_residual_fusion hyperparameter tuning",
        "",
        "Selection uses only an image-level 20% validation subset of the original training partition. "
        "Final values are mean±std over seeds 43–45 on the untouched strict test partition.",
        "",
        "| Model | Width | Batch | Dropout | LR | WD | Patience | Tuned Real AUC | Δ vs original | Tuned Hall AUPR | Δ vs original |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cfg, test, old = row["selected_config"], row["test_metrics"], row["old_test_baseline"]
        lines.append(
            f"| {row['model']} | {cfg['width']} | {cfg['batch_size']} | {cfg['dropout']:.2f} | "
            f"{cfg['learning_rate']:.1e} | {cfg['weight_decay']:.1e} | "
            f"{cfg['early_stopping_patience']} | "
            f"{test['real_auc']['mean']:.4f}±{test['real_auc']['std']:.4f} | "
            f"{test['real_auc']['mean'] - old['real_auc']['mean']:+.4f} | "
            f"{test['hall_aupr']['mean']:.4f}±{test['hall_aupr']['std']:.4f} | "
            f"{test['hall_aupr']['mean'] - old['hall_aupr']['mean']:+.4f} |"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def select_shared_config(out: Path) -> None:
    summaries = {
        model: json.loads((out / model / "summary.json").read_text())
        for model in MODELS
    }
    candidate_rows: dict[str, dict] = {}
    ranks: dict[tuple[str, str], int] = {}
    for model, summary in summaries.items():
        screening = sorted(
            summary["screening"],
            key=lambda row: (row["metrics"]["real_auc"], row["metrics"]["hall_aupr"]),
            reverse=True,
        )
        if len(screening) != len(CANDIDATES):
            raise ValueError(f"{model}: incomplete shared-config screening")
        for rank, row in enumerate(screening, 1):
            name = row["config"]["name"]
            ranks[(model, name)] = rank
            current = candidate_rows.setdefault(name, {
                "config": row["config"], "model_scores": {},
            })
            if current["config"] != row["config"]:
                raise ValueError(f"{name}: candidate configuration changed across models")
            current["model_scores"][model] = row["metrics"]
    ranked = []
    for name, row in candidate_rows.items():
        model_ranks = {model: ranks[(model, name)] for model in MODELS}
        aucs = [row["model_scores"][model]["real_auc"] for model in MODELS]
        auprs = [row["model_scores"][model]["hall_aupr"] for model in MODELS]
        ranked.append({
            **row,
            "model_ranks": model_ranks,
            "mean_rank": float(np.mean(list(model_ranks.values()))),
            "mean_validation_real_auc": float(np.mean(aucs)),
            "mean_validation_hall_aupr": float(np.mean(auprs)),
        })
    ranked.sort(key=lambda row: (
        row["mean_rank"], -row["mean_validation_real_auc"],
        row["config"]["width"], row["config"]["dropout"],
    ))
    selection = {
        "selection": "lowest mean within-model rank across six seed-43 validation screens; tie mean Real AUC",
        "selected_config": ranked[0]["config"],
        "candidates": ranked,
    }
    atomic_json(out / "shared_selection.json", selection)
    lines = [
        "# Shared hyperparameter selection", "",
        "Selected only from the six internal validation screens; strict test results are not used.", "",
        "| Rank | Candidate | Mean model rank | Mean validation AUC | Mean validation Hall AUPR |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | `{row['config']['name']}` | {row['mean_rank']:.2f} | "
            f"{row['mean_validation_real_auc']:.4f} | "
            f"{row['mean_validation_hall_aupr']:.4f} |"
        )
    (out / "shared_selection.md").write_text("\n".join(lines) + "\n")


def evaluate_shared(model: str, device_name: str, out: Path) -> None:
    selection = json.loads((out / "shared_selection.json").read_text())
    candidate = Candidate(**selection["selected_config"])
    data = load_data(model)
    features = data["features"]
    labels = data["label"]
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    device = torch.device(device_name)
    scores = []
    for seed in SEEDS:
        folder = out / "shared_final" / model / candidate.name / f"seed{seed}"
        scores.append(fit_case(
            features=features, labels=labels, train_mask=data["train_mask"],
            eval_mask=data["test_mask"], candidate=candidate, seed=seed,
            root_config=root_config, device=device, folder=folder,
        ))
        print("SHARED_FINAL", model, candidate.name, seed, scores[-1]["real_auc"], flush=True)
    result = {
        "model": model,
        "config": asdict(candidate),
        "test_metrics": summarize(scores),
        "old_test_baseline": old_baseline(model),
    }
    atomic_json(out / "shared_final" / model / "summary.json", result)


def write_shared_aggregate(out: Path) -> None:
    rows = [
        json.loads((out / "shared_final" / model / "summary.json").read_text())
        for model in MODELS
    ]
    config = rows[0]["config"]
    if any(row["config"] != config for row in rows):
        raise ValueError("Shared final evaluations used different configurations")
    lines = [
        "# Shared configuration: strict-test results", "",
        f"Configuration: `{config}`", "",
        "| Model | Real AUC | ΔAUC | Hall AUPR | ΔAUPR | Real F1 | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test, old = row["test_metrics"], row["old_test_baseline"]
        lines.append(
            f"| {row['model']} | {test['real_auc']['mean']:.4f}±{test['real_auc']['std']:.4f} | "
            f"{test['real_auc']['mean'] - old['real_auc']['mean']:+.4f} | "
            f"{test['hall_aupr']['mean']:.4f}±{test['hall_aupr']['std']:.4f} | "
            f"{test['hall_aupr']['mean'] - old['hall_aupr']['mean']:+.4f} | "
            f"{test['real_f1']['mean']:.4f}±{test['real_f1']['std']:.4f} | "
            f"{test['real_f1']['mean'] - old['real_f1']['mean']:+.4f} |"
        )
    (out / "shared_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--select-shared", action="store_true")
    parser.add_argument("--shared-model", choices=MODELS)
    parser.add_argument("--summarize-shared", action="store_true")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if args.summarize:
        write_aggregate(out)
        return
    if args.select_shared:
        select_shared_config(out)
        return
    if args.shared_model is not None:
        evaluate_shared(args.shared_model, args.device, out)
        return
    if args.summarize_shared:
        write_shared_aggregate(out)
        return
    if args.model is None:
        parser.error("--model is required unless --summarize is used")
    tune(args.model, args.device, out)
    write_aggregate(out)


if __name__ == "__main__":
    main()
