#!/usr/bin/env python3
"""Compare attention and ||a_m|| log-s targets with the shared 8:1:1 split."""

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
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.run_endac_baselines_811_dual_threshold import (  # noqa: E402
    atomic_json,
    dual_threshold_report,
    predict_real_probability,
    resolve_device,
    set_seed,
)
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


MODELS = (
    "llava_1_5_7b",
    "qwen2_5_vl_7b",
    "internvl_2_5_8b",
    "qwen3_vl_8b",
)
MODEL_NAMES = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}
MODEL = "llava_1_5_7b"
SEEDS = (43, 44, 45)
OLD_RISK = "log_s_hpre_k032_tau003"
NEW_RISK = "log_s_hpre_amnorm_k032_tau003"
BASE = ROOT / "outputs/log_s_hpre_k_tau_ablation/llava_1_5_7b/compact_features.npz"
EXTRACTED = (
    GF_ROOT
    / "outputs/coco4000_512_log_s_hpre_amnorm_target_811"
    / MODEL
    / "features.pkl"
)
SPLIT_PATH = GF_ROOT / "outputs/coco4000_512_endac_811/image_splits.json"
CASES = (
    "attention_target_risk",
    "attention_target_risk_f_log",
    "amnorm_target_risk",
    "amnorm_target_risk_f_log",
)
DISPLAY = {
    "attention_target_risk": r"risk: T=norm(attention*s)",
    "attention_target_risk_f_log": r"risk+F_log: T=norm(attention*s)",
    "amnorm_target_risk": r"risk: T=norm(||a_m||*s)",
    "amnorm_target_risk_f_log": r"risk+F_log: T=norm(||a_m||*s)",
}
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


def configure_model(model: str) -> None:
    global MODEL, BASE, EXTRACTED
    MODEL = model
    BASE = ROOT / "outputs/log_s_hpre_k_tau_ablation" / model / "compact_features.npz"
    EXTRACTED = (
        GF_ROOT
        / "outputs/coco4000_512_log_s_hpre_amnorm_target_811"
        / model
        / "features.pkl"
    )


def key_from_arrays(data: dict[str, np.ndarray], index: int) -> tuple[int, int, int]:
    return (
        int(data["image_id"][index]),
        int(data["response_token_idx"][index]),
        int(data["target_token_id"][index]),
    )


def row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def prepare(output_root: Path) -> dict[str, np.ndarray]:
    cache = output_root / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return {name: np.asarray(saved[name]) for name in saved.files}
    if not BASE.exists() or not EXTRACTED.exists():
        raise FileNotFoundError(
            f"Required features are incomplete: base={BASE.exists()} new={EXTRACTED.exists()}"
        )
    with np.load(BASE, allow_pickle=False) as saved:
        base = {name: np.asarray(saved[name]) for name in saved.files}
    rows = load_pkl(EXTRACTED)
    keys = [key_from_arrays(base, index) for index in range(len(base["label"]))]
    if len(set(keys)) != len(keys):
        raise ValueError("Base compact features contain duplicate mention keys")
    new_by_key = {row_key(row): row for row in rows}
    if len(new_by_key) != len(rows) or set(new_by_key) != set(keys):
        raise ValueError("Old/new mention keys differ")
    for index, key in enumerate(keys):
        row = new_by_key[key]
        if int(row["label"]) != int(base["label"][index]):
            raise ValueError(f"Label mismatch for {key}")
    new_risk = np.stack([
        np.asarray(new_by_key[key]["risks"][NEW_RISK], dtype=np.float32)
        for key in keys
    ])
    old_risk = np.asarray(base[OLD_RISK], dtype=np.float32)
    f_log = np.asarray(base["f_log"], dtype=np.float32)
    if new_risk.shape != old_risk.shape or f_log.shape != old_risk.shape:
        raise ValueError(
            f"Feature shape mismatch: new={new_risk.shape}, old={old_risk.shape}, F={f_log.shape}"
        )
    if any(not np.isfinite(value).all() for value in (new_risk, old_risk, f_log)):
        raise ValueError("Features contain non-finite values")
    result = {
        "attention_target_risk": old_risk,
        "attention_target_risk_f_log": np.concatenate((old_risk, f_log), axis=1),
        "amnorm_target_risk": new_risk,
        "amnorm_target_risk_f_log": np.concatenate((new_risk, f_log), axis=1),
        "label": np.asarray(base["label"], dtype=np.int8),
        "image_id": np.asarray(base["image_id"], dtype=np.int64),
        "response_token_idx": np.asarray(base["response_token_idx"], dtype=np.int32),
        "target_token_id": np.asarray(base["target_token_id"], dtype=np.int32),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.tmp.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(cache)
    atomic_json(output_root / "alignment.json", {
        "model": MODEL,
        "samples": int(len(rows)),
        "layers": int(old_risk.shape[1]),
        "mention_alignment": "exact (image_id,response_token_idx,target_token_id)",
        "label_alignment": "exact",
        "old_target": "renorm(attention*log_s)",
        "new_target": "renorm(am_norm*log_s)",
        "top_k": 32,
        "tau": 0.03,
        "base_path": str(BASE),
        "new_path": str(EXTRACTED),
    })
    return result


def load_split() -> dict[str, set[int]]:
    payload = json.loads(SPLIT_PATH.read_text())
    split = {
        name: {int(value) for value in payload[name]}
        for name in ("train", "val", "test")
    }
    if {name: len(values) for name, values in split.items()} != {
        "train": 3200,
        "val": 400,
        "test": 400,
    }:
        raise ValueError("Expected shared 3200/400/400 image split")
    return split


def train_case(
    *,
    case: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    split: dict[str, set[int]],
    seed: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    masks = {
        name: np.isin(image_ids, list(ids)) for name, ids in split.items()
    }
    arrays = {
        name: (matrix[mask], labels[mask]) for name, mask in masks.items()
    }
    for name, (_values, split_labels) in arrays.items():
        if set(np.unique(split_labels).tolist()) != {0, 1}:
            raise ValueError(f"{case}/{name}: both labels are required")
    mean = arrays["train"][0].astype(np.float64).mean(axis=0)
    scale = arrays["train"][0].astype(np.float64).std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = {
        name: ((values.astype(np.float64) - mean) / scale).astype(np.float32)
        for name, (values, _split_labels) in arrays.items()
    }

    set_seed(seed)
    chosen_device = torch.device(device)
    net = DGSTStyleProbe(
        input_dim=matrix.shape[1],
        hidden_sizes=(128,),
        dropout=0.3,
        batch_norm=False,
    ).to(chosen_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(standardized["train"]),
            torch.from_numpy(arrays["train"][1].astype(np.float32)),
        ),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    for epoch in range(1, 101):
        net.train()
        losses = []
        for values, targets in loader:
            values, targets = values.to(chosen_device), targets.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(values).squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _ = predict_real_probability(
            net,
            standardized["val"],
            arrays["val"][1],
            criterion,
            chosen_device,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = float(val_loss), epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in net.state_dict().items()
            }
        else:
            stale += 1
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "is_best": improved,
        })
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError(f"{case}/seed{seed}: no finite checkpoint")
    net.load_state_dict(best_state)
    _, val_real = predict_real_probability(
        net, standardized["val"], arrays["val"][1], criterion, chosen_device
    )
    _, test_real = predict_real_probability(
        net, standardized["test"], arrays["test"][1], criterion, chosen_device
    )
    report = dual_threshold_report(
        val_labels=arrays["val"][1],
        val_hall_scores=1.0 - val_real,
        test_labels=arrays["test"][1],
        test_hall_scores=1.0 - test_real,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "model.pt")
    np.savez(output_dir / "input_scaler.npz", mean=mean, scale=scale)
    np.savez_compressed(
        output_dir / "scores.npz",
        val_label=arrays["val"][1].astype(np.int8),
        val_hall_score=(1.0 - val_real).astype(np.float64),
        test_label=arrays["test"][1].astype(np.int8),
        test_hall_score=(1.0 - test_real).astype(np.float64),
    )
    atomic_json(output_dir / "history.json", history)
    config = {
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
        "checkpoint_selection": "minimum_validation_loss",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }
    atomic_json(output_dir / "config.json", config)
    result = {
        "model": MODEL,
        "case": case,
        "display": DISPLAY[case],
        "seed": seed,
        "split": "shared_image_level_811",
        "image_counts": {name: len(ids) for name, ids in split.items()},
        "token_counts": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
        "threshold_selection": {
            "real": "maximize validation Real-F1",
            "hall": "maximize validation Hall-F1",
        },
        "thresholds": report["thresholds"],
        "validation": report["validation"],
        "test": report["test"],
        "training": config,
    }
    atomic_json(output_dir / "result.json", result)
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


def summarize(output_root: Path) -> None:
    rows, missing = [], []
    for case in CASES:
        values = []
        for seed in SEEDS:
            path = output_root / case / f"seed{seed}" / "result.json"
            if not path.exists():
                missing.append(str(path))
                continue
            values.append(flatten(json.loads(path.read_text())))
        if not values:
            continue
        row: dict[str, Any] = {
            "case": case,
            "display": DISPLAY[case],
            "num_seeds": len(values),
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
        if old is None or new is None:
            continue
        comparisons.append({
            "feature": suffix,
            **{
                f"delta_{metric}": new[f"{metric}_mean"] - old[f"{metric}_mean"]
                for metric in METRICS
            },
        })
    payload = {
        "model": MODEL,
        "protocol": "shared image-level 8:1:1; seeds 43/44/45; dual validation thresholds",
        "old_target": "renorm(attention*log_s)",
        "new_target": "renorm(am_norm*log_s)",
        "top_k": 32,
        "tau": 0.03,
        "rows": rows,
        "comparisons": comparisons,
        "missing": missing,
    }
    atomic_json(output_root / "complete_results.json", payload)
    fields = ["case", "display", "num_seeds"] + [
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("mean", "std")
    ]
    with (output_root / "complete_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {MODEL_NAMES[MODEL]} log-s target: attention vs a_m norm",
        "",
        "- Old target: `renorm(attention_weight * s_m)`.",
        "- New target: `renorm(||a_m||_2 * s_m)`.",
        "- `s_m = log(1 + C p_m)`, K=32, tau=0.03; source, cost and penalty are unchanged.",
        "- Shared image split: 3200 train / 400 validation / 400 test.",
        "- Real and Hall thresholds independently maximize their validation F1.",
        "- Mean ± population standard deviation over seeds 43/44/45.",
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
            "",
            "## Change from attention target to a_m-norm target",
            "",
            "| Feature | ΔAUROC | ΔReal F1 | ΔReal AUPR | ΔHall F1 | ΔHall AUPR |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for comparison in comparisons:
            values = [
                100 * comparison[f"delta_{metric}"]
                for metric in ("auroc", "real_f1", "real_aupr", "hall_f1", "hall_aupr")
            ]
            lines.append(
                f"| {comparison['feature']} | "
                + " | ".join(f"{value:+.2f}" for value in values)
                + " |"
            )
    if missing:
        lines.extend(["", "## Missing", "", *[f"- `{path}`" for path in missing]])
    (output_root / "complete_results.md").write_text("\n".join(lines) + "\n")
    completed = len(CASES) * len(SEEDS) - len(missing)
    total = len(CASES) * len(SEEDS)
    blocks = int(round(30 * completed / total)) if total else 0
    (output_root / "progress.md").write_text(
        f"# {MODEL_NAMES[MODEL]} a_m-norm target ablation\n\n"
        f"`[{'█'*blocks}{'░'*(30-blocks)}] {completed}/{total} "
        f"({100*completed/total:.1f}%)`\n\nMissing runs: {len(missing)}\n"
    )
    print(f"summarized rows={len(rows)} missing={len(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="llava_1_5_7b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configure_model(args.model)
    default_name = (
        "llava_log_s_hpre_amnorm_target_811_ablation"
        if args.model == "llava_1_5_7b"
        else f"{args.model}_log_s_hpre_amnorm_target_811_ablation"
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else ROOT / "outputs" / default_name
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(output_root)
        return
    data = prepare(output_root)
    split = load_split()
    device = resolve_device(args.device)
    for case in CASES:
        for seed in SEEDS:
            output_dir = output_root / case / f"seed{seed}"
            result_path = output_dir / "result.json"
            if result_path.exists() and not args.overwrite:
                print(f"skip {case} seed={seed}", flush=True)
                continue
            print(f"run {case} seed={seed} device={device}", flush=True)
            train_case(
                case=case,
                matrix=np.asarray(data[case], dtype=np.float32),
                labels=np.asarray(data["label"], dtype=np.int64),
                image_ids=np.asarray(data["image_id"], dtype=np.int64),
                split=split,
                seed=seed,
                device=device,
                output_dir=output_dir,
            )
            summarize(output_root)
    summarize(output_root)


if __name__ == "__main__":
    main()
