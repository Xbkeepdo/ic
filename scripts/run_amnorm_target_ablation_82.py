#!/usr/bin/env python3
"""Run the attention-vs-a_m log-s target ablation on the strict 8:2 split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_endac_baselines_811_dual_threshold import (  # noqa: E402
    atomic_json,
    dual_threshold_report,
    predict_real_probability,
    resolve_device,
    set_seed,
)
from scripts.run_llava_amnorm_target_ablation_811 import (  # noqa: E402
    CASES,
    DISPLAY,
    MODEL_NAMES,
    MODELS,
    SEEDS,
)
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe  # noqa: E402


METRICS = (
    "accuracy_real_threshold",
    "accuracy_hall_threshold",
    "auroc",
    "real_f1",
    "real_recall",
    "real_aupr",
    "hall_f1",
    "hall_recall",
    "hall_aupr",
)


def source_811_root(model: str) -> Path:
    name = (
        "llava_log_s_hpre_amnorm_target_811_ablation"
        if model == "llava_1_5_7b"
        else f"{model}_log_s_hpre_amnorm_target_811_ablation"
    )
    return ROOT / "outputs" / name


def output_root(model: str) -> Path:
    return ROOT / "outputs" / f"{model}_log_s_hpre_amnorm_target_82_ablation"


def prepare(model: str, out: Path) -> dict[str, np.ndarray]:
    cache = out / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return {name: np.asarray(saved[name]) for name in saved.files}
    source_path = source_811_root(model) / "compact_features.npz"
    mask_path = ROOT / "outputs/log_s_hpre_k_tau_ablation" / model / "compact_features.npz"
    if not source_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"{model}: source compact features are incomplete")
    with np.load(source_path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(mask_path, allow_pickle=False) as saved:
        reference = {name: np.asarray(saved[name]) for name in (
            "label", "image_id", "response_token_idx", "target_token_id",
            "train_mask", "strict_test_mask",
        )}
    for name in ("label", "image_id", "response_token_idx", "target_token_id"):
        if not np.array_equal(data[name], reference[name]):
            raise ValueError(f"{model}: 8:1:1 compact order differs for {name}")
    train_mask = np.asarray(reference["train_mask"], dtype=bool)
    test_mask = np.asarray(reference["strict_test_mask"], dtype=bool)
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise ValueError(f"{model}: invalid strict 8:2 masks")
    data.update({"train_mask": train_mask, "test_mask": test_mask})
    out.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(cache)
    image_ids = np.asarray(data["image_id"], dtype=np.int64)
    atomic_json(out / "alignment.json", {
        "model": model,
        "mention_alignment": "exact with extracted new risk and original strict 8:2 masks",
        "samples": int(len(image_ids)),
        "image_counts": {
            "train": int(len(np.unique(image_ids[train_mask]))),
            "test": int(len(np.unique(image_ids[test_mask]))),
        },
        "token_counts": {
            "train": int(train_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": "minimum_train_loss",
        "thresholds": {
            "real": "maximize training Real-F1",
            "hall": "maximize training Hall-F1",
        },
    })
    return data


def train_case(
    *,
    model: str,
    case: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int,
    device: str,
    out: Path,
) -> dict[str, Any]:
    train_x, train_y = matrix[train_mask], labels[train_mask]
    test_x, test_y = matrix[test_mask], labels[test_mask]
    mean = train_x.astype(np.float64).mean(axis=0)
    scale = train_x.astype(np.float64).std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_z = ((train_x.astype(np.float64) - mean) / scale).astype(np.float32)
    test_z = ((test_x.astype(np.float64) - mean) / scale).astype(np.float32)

    set_seed(seed)
    chosen_device = torch.device(device)
    net = DGSTStyleProbe(
        input_dim=matrix.shape[1], hidden_sizes=(128,), dropout=0.3, batch_norm=False
    ).to(chosen_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_z), torch.from_numpy(train_y.astype(np.float32))
        ),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    for epoch in range(1, 101):
        net.train()
        loss_sum, count = 0.0, 0
        for values, targets in loader:
            values, targets = values.to(chosen_device), targets.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(values).squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            count += len(targets)
        train_loss = loss_sum / count
        improved = train_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = float(train_loss), epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in net.state_dict().items()
            }
        else:
            stale += 1
        scheduler.step(train_loss)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "is_best": improved,
        })
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError(f"{model}/{case}/seed{seed}: no finite checkpoint")
    net.load_state_dict(best_state)
    _, train_real = predict_real_probability(
        net, train_z, train_y, criterion, chosen_device
    )
    _, test_real = predict_real_probability(net, test_z, test_y, criterion, chosen_device)
    report = dual_threshold_report(
        val_labels=train_y,
        val_hall_scores=1.0 - train_real,
        test_labels=test_y,
        test_hall_scores=1.0 - test_real,
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    np.savez(out / "input_scaler.npz", mean=mean, scale=scale)
    np.savez_compressed(
        out / "scores.npz",
        train_label=train_y.astype(np.int8),
        train_hall_score=(1.0 - train_real).astype(np.float64),
        test_label=test_y.astype(np.int8),
        test_hall_score=(1.0 - test_real).astype(np.float64),
    )
    atomic_json(out / "history.json", history)
    config = {
        "model": model,
        "case": case,
        "input_dim": int(matrix.shape[1]),
        "hidden_sizes": [128],
        "batch_norm": False,
        "standardize_input": True,
        "dropout": 0.3,
        "batch_size": 256,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "lr_factor": 0.5,
        "lr_patience": 5,
        "early_stopping_patience": 10,
        "checkpoint_selection": "minimum_train_loss",
        "split_protocol": "strict_82_no_validation",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
    }
    atomic_json(out / "config.json", config)
    result = {
        "model": model,
        "case": case,
        "display": DISPLAY[case],
        "seed": seed,
        "split": "strict_82_no_validation",
        "token_counts": {"train": int(train_mask.sum()), "test": int(test_mask.sum())},
        "threshold_selection": {
            "real": "maximize training Real-F1",
            "hall": "maximize training Hall-F1",
        },
        "thresholds": report["thresholds"],
        "training_threshold_metrics": report["validation"],
        "test": report["test"],
        "training": config,
    }
    atomic_json(out / "result.json", result)
    return result


def flatten(result: dict[str, Any]) -> dict[str, float]:
    test = result["test"]
    return {
        "accuracy_real_threshold": float(test["accuracy_real_threshold"]),
        "accuracy_hall_threshold": float(test["accuracy_hall_threshold"]),
        "auroc": float(test["auroc"]),
        "real_f1": float(test["real"]["f1"]),
        "real_recall": float(test["real"]["recall"]),
        "real_aupr": float(test["real"]["aupr"]),
        "hall_f1": float(test["hall"]["f1"]),
        "hall_recall": float(test["hall"]["recall"]),
        "hall_aupr": float(test["hall"]["aupr"]),
    }


def summarize(model: str, out: Path) -> None:
    rows, missing = [], []
    for case in CASES:
        values = []
        for seed in SEEDS:
            path = out / case / f"seed{seed}" / "result.json"
            if not path.exists():
                missing.append(str(path))
            else:
                values.append(flatten(json.loads(path.read_text())))
        if not values:
            continue
        row: dict[str, Any] = {
            "case": case, "display": DISPLAY[case], "num_seeds": len(values)
        }
        for metric in METRICS:
            observed = np.asarray([value[metric] for value in values])
            row[f"{metric}_mean"] = float(observed.mean())
            row[f"{metric}_std"] = float(observed.std())
        rows.append(row)
    by_case = {row["case"]: row for row in rows}
    comparisons = []
    for suffix in ("risk", "risk_f_log"):
        old = by_case.get(f"attention_target_{suffix}")
        new = by_case.get(f"amnorm_target_{suffix}")
        if old is not None and new is not None:
            comparisons.append({
                "feature": suffix,
                **{
                    f"delta_{metric}": new[f"{metric}_mean"] - old[f"{metric}_mean"]
                    for metric in METRICS
                },
            })
    atomic_json(out / "complete_results.json", {
        "model": model,
        "protocol": "strict 8:2 without validation; train-loss checkpoint; dual train thresholds",
        "rows": rows,
        "comparisons": comparisons,
        "missing": missing,
    })
    if rows:
        fields = list(rows[0])
        with (out / "complete_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    lines = [
        f"# {MODEL_NAMES[model]} log-s target ablation — strict 8:2",
        "",
        "- 3200 train / 800 test images; no validation split.",
        "- Checkpoint minimizes training loss.",
        "- Real and Hall thresholds independently maximize their training F1.",
        "- K=32, tau=0.03; mean ± population std over seeds 43/44/45.",
        "",
        "| Feature | Acc@Real | Acc@Hall | AUROC | Real F1 | Real Recall | Real AUPR | Hall F1 | Hall Recall | Hall AUPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            f"{100*row[f'{metric}_mean']:.2f} ± {100*row[f'{metric}_std']:.2f}"
            for metric in METRICS
        ]
        lines.append(f"| {row['display']} | " + " | ".join(values) + " |")
    if comparisons:
        lines.extend([
            "", "## New target minus old target", "",
            "| Feature | ΔAUROC | ΔReal F1 | ΔReal AUPR | ΔHall F1 | ΔHall AUPR |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in comparisons:
            values = [100*row[f"delta_{metric}"] for metric in (
                "auroc", "real_f1", "real_aupr", "hall_f1", "hall_aupr"
            )]
            lines.append(
                f"| {row['feature']} | " + " | ".join(f"{value:+.2f}" for value in values) + " |"
            )
    if missing:
        lines.extend(["", "## Missing", "", *[f"- `{path}`" for path in missing]])
    (out / "complete_results.md").write_text("\n".join(lines) + "\n")
    completed, total = len(CASES)*len(SEEDS)-len(missing), len(CASES)*len(SEEDS)
    blocks = round(30*completed/total)
    (out / "progress.md").write_text(
        f"# {MODEL_NAMES[model]} strict 8:2 target ablation\n\n"
        f"`[{'█'*blocks}{'░'*(30-blocks)}] {completed}/{total} ({100*completed/total:.1f}%)`\n\n"
        f"Missing runs: {len(missing)}\n"
    )
    print(f"{model}: rows={len(rows)} missing={len(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = output_root(args.model)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args.model, out); return
    data = prepare(args.model, out)
    device = resolve_device(args.device)
    for case in CASES:
        for seed in SEEDS:
            folder = out / case / f"seed{seed}"
            if (folder / "result.json").exists() and not args.overwrite:
                print(f"skip {args.model} {case} seed={seed}", flush=True); continue
            print(f"run {args.model} {case} seed={seed} device={device}", flush=True)
            train_case(
                model=args.model,
                case=case,
                matrix=np.asarray(data[case], dtype=np.float32),
                labels=np.asarray(data["label"], dtype=np.int64),
                train_mask=np.asarray(data["train_mask"], dtype=bool),
                test_mask=np.asarray(data["test_mask"], dtype=bool),
                seed=seed,
                device=device,
                out=folder,
            )
            summarize(args.model, out)
    summarize(args.model, out)


if __name__ == "__main__":
    main()
