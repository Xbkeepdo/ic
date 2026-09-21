#!/usr/bin/env python3
"""Tune the one-layer risk + log-gate/log-write-norm probe without test leakage."""

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

from scripts.train_iclr_endac_attention_i_fusion import (  # noqa: E402
    MODELS,
    _probe_config,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402


SOURCE = ROOT / "outputs/fixed_wcost_log_gate_log_am"
OUT = ROOT / "outputs/fixed_wcost_log_gate_log_am_tuning"
SEEDS = (43, 44, 45)
SPLIT_SEED = 20260920
VAL_IMAGE_COUNT = 640
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


BASE = dict(width=128, batch_size=256, dropout=0.3,
            learning_rate=1e-3, weight_decay=1e-5)


def _candidate(name: str, **changes: float | int) -> Candidate:
    values = {**BASE, **changes}
    return Candidate(name=name, **values)


CANDIDATES = (
    _candidate("baseline"),
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
    _candidate("mix_h256_b128_d02_lr5e-4", width=256, batch_size=128,
               dropout=0.2, learning_rate=5e-4),
    _candidate("mix_h256_b256_d01_lr5e-4_wd0", width=256, dropout=0.1,
               learning_rate=5e-4, weight_decay=0.0),
    _candidate("mix_h512_b128_d03_lr3e-4", width=512, batch_size=128,
               learning_rate=3e-4),
    _candidate("mix_h64_b128_d00_lr1e-3_wd0", width=64, batch_size=128,
               dropout=0.0, weight_decay=0.0),
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
    required = {"feature", "risk", "label", "image_id", "train_mask", "test_mask"}
    if set(data) != required:
        raise ValueError(f"{model}: unexpected compact feature keys: {sorted(data)}")
    features = np.concatenate((data["risk"], data["feature"]), axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{model}: nonfinite tuning input")
    data["features"] = features
    return data


def make_split(model: str, data: dict[str, np.ndarray], model_out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    ids = data["image_id"]
    source_split = json.loads((
        ROOT / "outputs" / model / "COCO4000-512-WCOST-ALL" / "image_splits.json"
    ).read_text())
    train_images = np.asarray(sorted(map(int, source_split["train"])), dtype=np.int64)
    test_images = np.asarray(sorted(map(int, source_split["test"])), dtype=np.int64)
    if len(train_images) != 3200 or len(test_images) != 800:
        raise ValueError(f"{model}: expected 3200/800 image split")
    if np.intersect1d(train_images, test_images).size:
        raise ValueError(f"{model}: train/test image overlap")
    if not np.all(np.isin(ids[data["train_mask"]], train_images)):
        raise ValueError(f"{model}: compact train rows differ from source split")
    if not np.all(np.isin(ids[data["test_mask"]], test_images)):
        raise ValueError(f"{model}: compact test rows differ from source split")
    rng = np.random.default_rng(SPLIT_SEED)
    permuted = rng.permutation(np.sort(train_images))
    val_images = np.sort(permuted[:VAL_IMAGE_COUNT])
    fit_images = np.sort(permuted[VAL_IMAGE_COUNT:])
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
        metrics_at(SOURCE / model / "risk_log_gate_log_am_sum" / f"seed{seed}" / "metrics.json")
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
            "weight_decay", "seeds", "val_real_auc_mean", "val_real_auc_std",
            "val_hall_aupr_mean", "val_hall_aupr_std",
        ])
        for rank, row in enumerate(rows, 1):
            cfg, metrics = row["config"], row["validation_metrics"]
            writer.writerow([
                rank, cfg["name"], cfg["width"], cfg["batch_size"], cfg["dropout"],
                cfg["learning_rate"], cfg["weight_decay"], ",".join(map(str, row["seeds"])),
                metrics["real_auc"]["mean"], metrics["real_auc"]["std"],
                metrics["hall_aupr"]["mean"], metrics["hall_aupr"]["std"],
            ])
    selected = summary["selected_config"]
    test = summary["test_metrics"]
    old = summary["old_test_baseline"]
    lines = [
        f"# {summary['model']}: risk + F_l hyperparameter tuning",
        "",
        "Selection uses a fixed 2560/640 image split inside the original 3200 training images. "
        "The original 800 test images are used only after selecting a configuration.",
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
        "| Rank | Candidate | Width | Batch | Dropout | LR | WD | Val AUC | Val Hall AUPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        cfg, metric = row["config"], row["validation_metrics"]
        lines.append(
            f"| {rank} | {cfg['name']} | {cfg['width']} | {cfg['batch_size']} | "
            f"{cfg['dropout']:.2f} | {cfg['learning_rate']:.1e} | {cfg['weight_decay']:.1e} | "
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
        "schema": "fixed-wcost-log-gate-log-am-tuning-v1",
        "model": model,
        "feature": "risk + F_l, F_l=sum_m log1p(G_lm)*log1p(||a_lm||_2)",
        "selection_protocol": {
            "screen": f"all {len(CANDIDATES)} candidates with seed {SCREEN_SEED}",
            "confirm": f"top {TOP_K} by validation Real AUC with seeds {list(SEEDS)}",
            "select": "mean validation Real AUC; tie Hall AUPR, smaller width, lower dropout",
            "final": "retrain selected configuration on all 3200 training images; evaluate 800 test images",
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
        "# Risk + log-gate/log-write-norm hyperparameter tuning",
        "",
        "Selection is performed only on a 640-image validation subset of the original training set. "
        "Final values are mean±std over seeds 43–45 on the untouched 800-image test set.",
        "",
        "| Model | Width | Batch | Dropout | LR | WD | Tuned Real AUC | Δ vs original | Tuned Hall AUPR | Δ vs original |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cfg, test, old = row["selected_config"], row["test_metrics"], row["old_test_baseline"]
        lines.append(
            f"| {row['model']} | {cfg['width']} | {cfg['batch_size']} | {cfg['dropout']:.2f} | "
            f"{cfg['learning_rate']:.1e} | {cfg['weight_decay']:.1e} | "
            f"{test['real_auc']['mean']:.4f}±{test['real_auc']['std']:.4f} | "
            f"{test['real_auc']['mean'] - old['real_auc']['mean']:+.4f} | "
            f"{test['hall_aupr']['mean']:.4f}±{test['hall_aupr']['std']:.4f} | "
            f"{test['hall_aupr']['mean'] - old['hall_aupr']['mean']:+.4f} |"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if args.summarize:
        write_aggregate(out)
        return
    if args.model is None:
        parser.error("--model is required unless --summarize is used")
    tune(args.model, args.device, out)
    write_aggregate(out)


if __name__ == "__main__":
    main()
