#!/usr/bin/env python3
"""Train/evaluate the TruthPrInt single-layer MLP on existing 8:1:1 cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.baselines import select_detection_threshold  # noqa: E402
from detection.qa_probe import classification_metrics  # noqa: E402
from features.pickle_journal import iter_pickle_records  # noqa: E402
from features.qa_baseline import qa_image_identity  # noqa: E402
from scripts.run_endac_baselines_811_dual_threshold import dual_threshold_report  # noqa: E402


SEEDS = (43, 44, 45)
COCO_SOURCE = "COCO4000-512-ENDAC-BASELINES-V2"
QA_SOURCE = "LOGS-K32-TAU003-NATIVE6-FULL"
QA_SPLITS = "LOGS-K32-TAU003-NATIVE6-FULL-811"
COCO_SPLITS = ROOT.parent / "gfchair/outputs/coco4000_512_endac_811/image_splits.json"


class TruthPrIntMLP(nn.Module):
    """The repository's [hidden, 128, 64, 1] ReLU probe, single branch."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten()


def row_key(row: dict, scope: str):
    if scope == "coco":
        return int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"])
    return str(row["key"])


def load_expected(model: str, scope: str, dataset: str | None) -> dict:
    if scope == "coco":
        path = ROOT / "outputs" / model / COCO_SOURCE / "baseline/features.pkl"
    else:
        path = ROOT / "outputs/qa_benchmarks" / model / QA_SOURCE / str(dataset) / "features.pkl"
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    expected = {row_key(row, scope): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError(f"Duplicate source rows: {path}")
    return expected


def load_states(root: Path, model: str, scope: str, dataset: str | None,
                expected: dict, layer: int) -> dict:
    name = "coco" if scope == "coco" else str(dataset)
    parts = sorted((root / model / name).glob("part*-of-*.pkl"))
    if not parts:
        raise FileNotFoundError(f"No TruthPrInt shards for {model}/{name}")
    extracted = {}
    for part in parts:
        for row in iter_pickle_records(part):
            key = row_key(row, scope)
            if key in extracted or key not in expected:
                raise ValueError(f"Duplicate or unexpected TruthPrInt key: {key}")
            if int(row["layer"]) != layer or int(row["label"]) != int(expected[key]["label"]):
                raise ValueError(f"Layer/label mismatch: {key}")
            state = np.asarray(row["state"], dtype=np.float32)
            if state.ndim != 1 or not np.isfinite(state).all():
                raise ValueError(f"Invalid hidden state: {key}")
            extracted[key] = state
    if set(extracted) != set(expected):
        missing = set(expected) - set(extracted)
        raise ValueError(f"Incomplete extraction for {model}/{name}: {len(missing)} missing of {len(expected)}")
    widths = {len(state) for state in extracted.values()}
    if len(widths) != 1:
        raise ValueError(f"Mixed state widths: {widths}")
    return extracted


def split_assignment(scope: str, dataset: str | None) -> dict:
    if scope == "coco":
        payload = json.loads(COCO_SPLITS.read_text())
        groups = {name: set(int(value) for value in payload[name]) for name in ("train", "val", "test")}
        if {name: len(values) for name, values in groups.items()} != {"train": 3200, "val": 400, "test": 400}:
            raise ValueError("Unexpected COCO 8:1:1 split")
        return {image_id: split for split, ids in groups.items() for image_id in ids}
    path = ROOT / "outputs/qa_benchmarks" / QA_SPLITS / "splits" / f"{dataset}.json"
    payload = json.loads(path.read_text())
    if payload["protocol"] != "strict_outer_image_level_811_from_existing_outer82":
        raise ValueError(f"Unexpected QA split: {path}")
    assignment = {identity: split for split in ("train", "val", "test")
                  for identity in payload["images"][split]}
    if len(assignment) != sum(len(payload["images"][split]) for split in ("train", "val", "test")):
        raise ValueError("Overlapping QA images")
    return assignment


def prepare_matrices(expected: dict, extracted: dict, assignment: dict,
                     scope: str) -> tuple[dict, dict, dict]:
    groups = {split: [] for split in ("train", "val", "test")}
    for key, row in expected.items():
        identity = int(row["image_id"]) if scope == "coco" else qa_image_identity(row)
        split = assignment[identity]
        groups[split].append(key)
    matrices, labels = {}, {}
    for split, keys in groups.items():
        matrices[split] = np.stack([extracted[key] for key in keys]).astype(np.float32)
        labels[split] = np.asarray([int(expected[key]["label"]) for key in keys], dtype=np.int64)
        if set(labels[split]) != {0, 1}:
            raise ValueError(f"{split} lacks one class")
    return groups, matrices, labels


def center_unit(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = x - mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-12).any():
        raise ValueError("Invalid centered hidden-state norm")
    return (centered / norms).astype(np.float32)


def tpr_at_one_percent_fpr(y_hall: np.ndarray, hall_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_hall, hall_scores)
    valid = tpr[fpr <= 0.01 + 1e-12]
    return float(valid.max()) if len(valid) else 0.0


def score(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            logits = model(torch.from_numpy(x[start:start + 2048]).to(device))
            batches.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(batches).astype(np.float64)


def train_seed(seed: int, groups: dict, matrices: dict, labels: dict,
               expected: dict, scope: str, device: str, output: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    y_train_hall = 1 - labels["train"]
    hall = np.flatnonzero(y_train_hall == 1)
    real = np.flatnonzero(y_train_hall == 0)
    count = min(len(hall), len(real))
    balanced = np.concatenate((rng.choice(hall, count, replace=False),
                               rng.choice(real, count, replace=False)))
    rng.shuffle(balanced)
    mean = matrices["train"][balanced].mean(axis=0, keepdims=True)
    scaled = {split: center_unit(matrix, mean) for split, matrix in matrices.items()}
    model = TruthPrIntMLP(scaled["train"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(
        torch.from_numpy(scaled["train"][balanced]),
        torch.from_numpy(y_train_hall[balanced].astype(np.float32))),
        batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    best = (-1.0, -1.0)
    best_state = None
    best_epoch = -1
    for epoch in range(30):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch.to(device)), y_batch.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite TruthPrInt loss")
            loss.backward()
            optimizer.step()
        val_hall = score(model, scaled["val"], device)
        quality = (tpr_at_one_percent_fpr(1 - labels["val"], val_hall),
                   float(roc_auc_score(1 - labels["val"], val_hall)))
        if quality > best:
            best = quality
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    hall_scores = {split: score(model, scaled[split], device) for split in ("train", "val", "test")}
    val_real_scores = 1.0 - hall_scores["val"]
    real_threshold = select_detection_threshold(labels["val"], hall_scores["val"], positive_class="real")
    result = {
        "seed": seed,
        "train_balanced_class_count": count,
        "counts": {split: len(groups[split]) for split in groups},
        "class_counts": {split: np.bincount(labels[split], minlength=2).tolist() for split in groups},
        "training": {"epochs": 30, "batch_size": 512, "learning_rate": 0.001,
                     "architecture": "single_branch_mlp_128_64_1_relu",
                     "preprocessing": "training_balanced_mean_center_then_l2_unit",
                     "selection": "validation_tpr_at_fpr_0.01_tiebreak_auroc",
                     "best_epoch": best_epoch, "validation_tpr_at_fpr_0.01": best[0],
                     "validation_auroc": best[1]},
        "real_f1_threshold": float(real_threshold),
        "fixed_0.5_test": classification_metrics(labels["test"], 1.0 - hall_scores["test"], 0.5),
        "test": classification_metrics(labels["test"], 1.0 - hall_scores["test"], real_threshold),
    }
    if scope == "coco":
        result["dual_threshold"] = dual_threshold_report(
            val_labels=labels["val"], val_hall_scores=hall_scores["val"],
            test_labels=labels["test"], test_hall_scores=hall_scores["test"])
    else:
        result["amber_dimensions"] = {}
        if any(expected[key].get("source_split") in {"existence", "attribute", "relation"}
               for key in groups["test"]):
            for dimension in ("existence", "attribute", "relation"):
                mask = np.asarray([expected[key].get("source_split") == dimension for key in groups["test"]])
                if mask.any():
                    result["amber_dimensions"][dimension] = classification_metrics(
                        labels["test"][mask], 1.0 - hall_scores["test"][mask], real_threshold)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "train_mean": mean,
                "layer": 16, "seed": seed, "best_epoch": best_epoch}, output / "checkpoint.pt")
    np.savez_compressed(output / "test_scores.npz",
                        keys=np.asarray([str(key) for key in groups["test"]]),
                        labels=labels["test"], hall_scores=hall_scores["test"])
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def summarize(model: str, name: str, root: Path) -> None:
    results = [json.loads((root / model / name / f"seed{seed}/result.json").read_text()) for seed in SEEDS]
    lines = [f"# TruthPrInt {model} / {name}", "",
             "Source: jinhaoduan/TruthPrInt commit c61af74; post-block-16 previous-token state; 128/64 MLP.",
             "Training balances classes within the train split; model selection and threshold use validation only.", "",
             "| Metric | Mean ± std |", "|---|---:|"]
    if name == "coco":
        lines[3] += " Real and Hall metrics use their own validation-selected F1 thresholds."
        tests = [item["dual_threshold"]["test"] for item in results]
        values_by_metric = {
            "auroc": [test["auroc"] for test in tests],
            "real_aupr": [test["real"]["aupr"] for test in tests],
            "hallucination_aupr": [test["hall"]["aupr"] for test in tests],
            "accuracy_at_real_threshold": [test["accuracy_real_threshold"] for test in tests],
            "accuracy_at_hall_threshold": [test["accuracy_hall_threshold"] for test in tests],
        }
        for label, key in (("real", "real"), ("hallucination", "hall")):
            for metric in ("precision", "recall", "f1"):
                values_by_metric[f"{label} {metric}"] = [test[key][metric] for test in tests]
    else:
        values_by_metric = {
            metric: [item["test"][metric] for item in results]
            for metric in ("auroc", "real_aupr", "hallucination_aupr", "accuracy")
        }
        for label in ("real", "hallucination"):
            for metric in ("precision", "recall", "f1"):
                values_by_metric[f"{label} {metric}"] = [
                    item["test"][label][metric] for item in results
                ]
    for metric, values in values_by_metric.items():
        lines.append(f"| {metric} | {100*np.mean(values):.2f} ± {100*np.std(values):.2f} |")
    if name == "amber_discriminative":
        for dimension in ("existence", "attribute", "relation"):
            for metric in ("auroc", "accuracy"):
                values = [float(item["amber_dimensions"][dimension][metric]) for item in results]
                lines.append(f"| {dimension} {metric} | {100*np.mean(values):.2f} ± {100*np.std(values):.2f} |")
            for label in ("real", "hallucination"):
                values = [float(item["amber_dimensions"][dimension][label]["f1"]) for item in results]
                lines.append(f"| {dimension} {label} F1 | {100*np.mean(values):.2f} ± {100*np.std(values):.2f} |")
    path = root / model / name / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scope", choices=("coco", "qa"), required=True)
    parser.add_argument("--dataset", choices=("clevr_exist_9k", "amber_discriminative"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/truthprint_baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()
    if args.scope == "qa" and not args.dataset:
        parser.error("--dataset is required for QA")
    name = "coco" if args.scope == "coco" else args.dataset
    expected = load_expected(args.model, args.scope, args.dataset)
    states = load_states(args.output_root, args.model, args.scope, args.dataset, expected, args.layer)
    groups, matrices, labels = prepare_matrices(expected, states,
                                               split_assignment(args.scope, args.dataset), args.scope)
    for seed in args.seeds:
        output = args.output_root / args.model / name / f"seed{seed}"
        if (output / "result.json").exists():
            print(f"[TruthPrInt] Existing result: {output}")
            continue
        result = train_seed(seed, groups, matrices, labels, expected, args.scope, args.device, output)
        print(f"[TruthPrInt] {args.model}/{name} seed={seed}: AUROC={result['test']['auroc']:.4f}", flush=True)
    if tuple(args.seeds) == SEEDS:
        summarize(args.model, name, args.output_root)


if __name__ == "__main__":
    main()
