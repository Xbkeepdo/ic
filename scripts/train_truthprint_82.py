#!/usr/bin/env python3
"""TruthPrInt COCO 8:2 detector: train-selected threshold, no validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean, pstdev
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.baselines import select_detection_threshold
from detection.qa_probe import classification_metrics
from scripts.train_truthprint import (
    ROOT, SEEDS, TruthPrIntMLP, center_unit, load_expected, load_states,
    prepare_matrices, score,
)


SPLIT = ROOT / "outputs/shared/COCO4000_seed42_image_splits_82.json"
OUT = ROOT / "outputs/truthprint_baseline"
MODELS = ("llava_1_5_7b", "qwen2_5_vl_7b", "internvl_2_5_8b", "qwen3_vl_8b")


def assignment() -> dict[int, str]:
    payload = json.loads(SPLIT.read_text())
    if payload["val"] or len(payload["train"]) != 3200 or len(payload["test"]) != 800:
        raise ValueError("Unexpected COCO 8:2 split")
    mapping = {int(image_id): split for split in ("train", "test")
               for image_id in payload[split]}
    if len(mapping) != 4000:
        raise ValueError("8:2 images overlap or are missing")
    return mapping


def matrices_82(expected: dict, states: dict, owner: dict):
    groups = {split: [] for split in ("train", "test")}
    for key in expected:
        groups[owner[key[0]]].append(key)
    x = {split: np.stack([states[key] for key in keys]).astype(np.float32)
         for split, keys in groups.items()}
    y = {split: np.asarray([int(expected[key]["label"]) for key in keys], dtype=np.int64)
         for split, keys in groups.items()}
    for split in groups:
        if set(y[split]) != {0, 1}:
            raise ValueError(f"Missing COCO class in {split}")
    return groups, x, y


def train_seed(seed: int, groups: dict, x: dict, y: dict,
               device: str, path: Path) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    y_hall = 1 - y["train"]
    hall = np.flatnonzero(y_hall == 1)
    real = np.flatnonzero(y_hall == 0)
    count = min(len(hall), len(real))
    balanced = np.concatenate((rng.choice(hall, count, replace=False),
                               rng.choice(real, count, replace=False)))
    rng.shuffle(balanced)
    train_mean = x["train"][balanced].mean(axis=0, keepdims=True)
    scaled = {split: center_unit(matrix, train_mean) for split, matrix in x.items()}
    model = TruthPrIntMLP(scaled["train"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(
        torch.from_numpy(scaled["train"][balanced]),
        torch.from_numpy(y_hall[balanced].astype(np.float32))),
        batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    for epoch in range(30):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch.to(device)), y_batch.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite TruthPrInt training loss")
            loss.backward()
            optimizer.step()
    train_hall = score(model, scaled["train"], device)
    test_hall = score(model, scaled["test"], device)
    threshold = select_detection_threshold(y["train"], train_hall,
                                           positive_class="real")
    result = {
        "seed": seed,
        "layer": 16,
        "split_protocol": "strict_image_level_82_no_validation",
        "checkpoint_selection": "last_epoch_30",
        "threshold_selection": "train_real_f1",
        "balanced_class_count": count,
        "counts": {split: len(keys) for split, keys in groups.items()},
        "class_counts": {split: np.bincount(y[split], minlength=2).tolist()
                         for split in groups},
        "real_score_threshold": float(threshold),
        "train": classification_metrics(y["train"], 1.0 - train_hall, threshold),
        "test": classification_metrics(y["test"], 1.0 - test_hall, threshold),
        "fixed_0.5_test": classification_metrics(y["test"], 1.0 - test_hall, 0.5),
    }
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "train_mean": train_mean,
                "layer": 16, "seed": seed, "epoch": 30}, path / "checkpoint.pt")
    np.savez_compressed(path / "test_scores.npz",
                        keys=np.asarray([str(key) for key in groups["test"]]),
                        labels=y["test"], hall_scores=test_hall)
    (path / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def summarize(models: tuple[str, ...]) -> Path:
    lines = ["# TruthPrInt COCO 8:2 detection", "",
             "Shared 3,200/800 image split, no validation; final epoch after 30 training epochs; Real F1 threshold selected on train. Seeds 43/44/45, post-block-16 previous-token state and 128/64 MLP. Percent test mean ± population std.", "",
             "| Model | Test tokens | AUROC | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        results = [json.loads((OUT / model / "coco_82" / f"seed{seed}/result.json").read_text())
                   for seed in SEEDS]
        def fmt(values):
            return f"{100 * mean(values):.2f} ± {100 * pstdev(values):.2f}"
        lines.append("| " + " | ".join((
            model, str(results[0]["counts"]["test"]),
            fmt([item["test"]["auroc"] for item in results]),
            fmt([item["test"]["hallucination_aupr"] for item in results]),
            fmt([item["test"]["real"]["f1"] for item in results]),
            fmt([item["test"]["hallucination"]["f1"] for item in results]),
            fmt([item["test"]["accuracy"] for item in results]),
        )) + " |")
    output = OUT / "coco_82_results.md"
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    owner = assignment()
    for model in args.models:
        expected = load_expected(model, "coco", None)
        states = load_states(OUT, model, "coco", None, expected, 16)
        groups, x, y = matrices_82(expected, states, owner)
        for seed in args.seeds:
            path = OUT / model / "coco_82" / f"seed{seed}"
            if (path / "result.json").exists():
                print(f"[TruthPrInt 8:2] existing {model} seed={seed}", flush=True)
                continue
            result = train_seed(seed, groups, x, y, args.device, path)
            print(f"[TruthPrInt 8:2] {model} seed={seed} AUROC={result['test']['auroc']:.4f}", flush=True)
    if tuple(args.seeds) == SEEDS and set(args.models) == set(MODELS):
        print(summarize(MODELS))


if __name__ == "__main__":
    main()
