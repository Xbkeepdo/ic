#!/usr/bin/env python3
"""Run ENDAC baselines and Ours on the shared 8:1:1 image split.

The experiment uses one fitted scorer per method and selects two independent
validation thresholds: one maximizing Real-positive F1 and one maximizing
Hallucination-positive F1.  Test labels never influence training, checkpoint
selection, hyperparameter selection, or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from detection.baselines import (  # noqa: E402
    SVARMLP,
    build_dense_baseline_matrix,
    evaluate_detection_scores,
    get_baseline_payload,
    select_detection_threshold,
    sklearn_hallucination_scores,
    torch_hallucination_scores,
)
from features.baseline import baseline_config  # noqa: E402
from scripts.train_baselines import (  # noqa: E402
    _evaluate_instruct_lens,
    _evaluate_projectaway,
    _svar_training_layer_range,
    _train_metatoken,
    _train_svar,
    _train_token_grounding,
)
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
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
METHODS = (
    "metatoken_gb",
    "svar",
    "projectaway",
    "token_grounding_mlp",
    "inslen",
    "ours",
)
METHOD_NAMES = {
    "metatoken_gb": "MetaToken (GB)",
    "svar": "SVAR",
    "projectaway": "ProjectAway",
    "token_grounding_mlp": "TokenGrounding (MLP)",
    "inslen": "InsLen",
    "ours": "Ours",
}
SEEDS = (43, 44, 45)
OUT = ROOT / "outputs/COCO4000-512-ENDAC-BASELINES-811-DUAL"
SPLIT_PATH = (
    GF_ROOT / "outputs/coco4000_512_endac_811/image_splits.json"
)
CONFIG_PATH = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
BASELINE_RUN = "COCO4000-512-ENDAC-BASELINES-V2"
OURS_RUN = ROOT / "outputs/log_s_hpre_k_tau_ablation"
METRIC_KEYS = (
    "accuracy_real_threshold",
    "accuracy_hall_threshold",
    "auroc",
    "real_precision",
    "real_f1",
    "real_recall",
    "real_aupr",
    "hall_precision",
    "hall_f1",
    "hall_recall",
    "hall_aupr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(output_root)
        return
    if args.model is None:
        raise ValueError("--model is required unless --summarize is used")
    device = resolve_device(args.device)
    audit = prepare_audit(args.model)
    model_root = output_root / args.model
    model_root.mkdir(parents=True, exist_ok=True)
    atomic_json(model_root / "audit.json", audit)
    if args.audit_only:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    split = load_split()
    records = load_baseline_records(args.model)
    split_records = split_rows(records, split)
    config = load_config(str(CONFIG_PATH))
    baseline_cfg = baseline_config(config)
    baseline_cfg = json.loads(json.dumps(baseline_cfg))
    baseline_cfg.setdefault("token_grounding", {})["classifiers"] = ["mlp"]

    for seed in args.seeds:
        seed_cfg = dict(baseline_cfg)
        seed_cfg["seed"] = int(seed)
        seed_root = model_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for method in args.methods:
            result_path = seed_root / method / "result.json"
            if result_path.exists() and not args.overwrite:
                print(f"[811] skip {args.model} seed={seed} {method}", flush=True)
                continue
            print(
                f"[811] run {args.model} seed={seed} {method} device={device}",
                flush=True,
            )
            result = run_method(
                method=method,
                model=args.model,
                seed=int(seed),
                split_records=split_records,
                split=split,
                baseline_cfg=seed_cfg,
                device=device,
                output_dir=result_path.parent,
            )
            atomic_json(result_path, result)
            write_model_progress(output_root, args.model)
    write_model_progress(output_root, args.model)


def load_split() -> dict[str, set[int]]:
    payload = json.loads(SPLIT_PATH.read_text())
    split = {
        name: {int(value) for value in payload[name]}
        for name in ("train", "val", "test")
    }
    counts = {name: len(values) for name, values in split.items()}
    if counts != {"train": 3200, "val": 400, "test": 400}:
        raise ValueError(f"Expected shared 3200/400/400 split, got {counts}")
    if split["train"] & split["val"] or split["train"] & split["test"] or split["val"] & split["test"]:
        raise ValueError("Shared 8:1:1 image partitions overlap")
    return split


def baseline_feature_path(model: str) -> Path:
    return ROOT / "outputs" / model / BASELINE_RUN / "baseline/features.pkl"


def ours_feature_path(model: str) -> Path:
    return OURS_RUN / model / "compact_features.npz"


def load_baseline_records(model: str) -> list[dict[str, Any]]:
    path = baseline_feature_path(model)
    rows = load_pkl(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid baseline feature rows: {path}")
    return rows


def key_of(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def prepare_audit(model: str) -> dict[str, Any]:
    split = load_split()
    baseline_rows = load_baseline_records(model)
    baseline_labels = {key_of(row): int(row["label"]) for row in baseline_rows}
    if len(baseline_labels) != len(baseline_rows):
        raise ValueError(f"{model}: duplicate baseline mention keys")
    with np.load(ours_feature_path(model), allow_pickle=False) as saved:
        ours_labels = {
            (int(image), int(response), int(target)): int(label)
            for image, response, target, label in zip(
                saved["image_id"],
                saved["response_token_idx"],
                saved["target_token_id"],
                saved["label"],
            )
        }
        matrix_shape = [
            int(len(saved["label"])),
            int(saved["log_s_hpre_k032_tau003"].shape[1] + saved["f_log"].shape[1]),
        ]
    if baseline_labels != ours_labels:
        raise ValueError(f"{model}: baseline and Ours keys/labels differ")
    split_counts = {
        name: sum(int(row["image_id"]) in images for row in baseline_rows)
        for name, images in split.items()
    }
    for name in ("train", "val", "test"):
        labels = {
            int(row["label"])
            for row in baseline_rows
            if int(row["image_id"]) in split[name]
        }
        if labels != {0, 1}:
            raise ValueError(f"{model}: split {name} lacks one class: {labels}")
    return {
        "model": model,
        "split_path": str(SPLIT_PATH),
        "image_counts": {name: len(values) for name, values in split.items()},
        "token_counts": split_counts,
        "total_tokens": len(baseline_rows),
        "ours_matrix_shape": matrix_shape,
        "alignment": "exact mention-key and label equality",
        "dhcp_excluded": True,
        "token_grounding_detector": "sklearn MLP native repository grid",
        "thresholds": {
            "real": "validation Real-F1 maximum",
            "hall": "validation Hall-F1 maximum",
        },
    }


def split_rows(
    rows: Sequence[dict[str, Any]], split: Mapping[str, set[int]]
) -> dict[str, list[dict[str, Any]]]:
    output = {
        name: [row for row in rows if int(row["image_id"]) in images]
        for name, images in split.items()
    }
    if sum(len(value) for value in output.values()) != len(rows):
        raise ValueError("Some baseline rows are outside the shared image split")
    return output


def run_method(
    *,
    method: str,
    model: str,
    seed: int,
    split_records: Mapping[str, list[dict[str, Any]]],
    split: Mapping[str, set[int]],
    baseline_cfg: dict[str, Any],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    native: Mapping[str, Any]
    if method == "metatoken_gb":
        native = _train_metatoken(
            split_records, checkpoint_dir, baseline_cfg, "real"
        )["gb"]
        matrices = dense_matrices(split_records, "metatoken")
        classifier = load_pickle(Path(native["checkpoint"]))
        val_scores = sklearn_hallucination_scores(classifier, matrices["val"][0])
        test_scores = sklearn_hallucination_scores(classifier, matrices["test"][0])
        val_labels, test_labels = matrices["val"][1], matrices["test"][1]
    elif method == "svar":
        native = _train_svar(
            split_records, checkpoint_dir, baseline_cfg, device, "real"
        )
        start, end = _svar_training_layer_range(baseline_cfg)
        matrices = {
            name: build_dense_baseline_matrix(
                split_records[name],
                "svar",
                svar_layer_start=start,
                svar_layer_end=end,
            )[:2]
            for name in ("train", "val", "test")
        }
        checkpoint = torch.load(native["checkpoint"], map_location="cpu")
        net = SVARMLP(
            matrices["train"][0].shape[1],
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(torch.device(device))
        net.load_state_dict(checkpoint["state_dict"])
        val_scores = torch_hallucination_scores(
            net, matrices["val"][0], torch.device(device)
        )
        test_scores = torch_hallucination_scores(
            net, matrices["test"][0], torch.device(device)
        )
        val_labels, test_labels = matrices["val"][1], matrices["test"][1]
    elif method == "projectaway":
        native = _evaluate_projectaway(
            split_records, "real", strict_82_no_validation=False
        )
        val_scores, val_labels = direct_scores(
            split_records["val"], "projectaway", "hallucination_score"
        )
        test_scores, test_labels = direct_scores(
            split_records["test"], "projectaway", "hallucination_score"
        )
    elif method == "token_grounding_mlp":
        native = _train_token_grounding(
            split_records, checkpoint_dir, baseline_cfg, device, "real"
        )["mlp"]
        matrices = dense_matrices(split_records, "token_grounding")
        classifier = load_pickle(Path(native["checkpoint"]))
        val_scores = sklearn_hallucination_scores(classifier, matrices["val"][0])
        test_scores = sklearn_hallucination_scores(classifier, matrices["test"][0])
        val_labels, test_labels = matrices["val"][1], matrices["test"][1]
    elif method == "inslen":
        native = _evaluate_instruct_lens(
            split_records, "real", strict_82_no_validation=False
        )
        val_truth, val_labels = direct_scores(
            split_records["val"], "instruct_lens", "score"
        )
        test_truth, test_labels = direct_scores(
            split_records["test"], "instruct_lens", "score"
        )
        val_scores, test_scores = -val_truth, -test_truth
    elif method == "ours":
        native, val_scores, val_labels, test_scores, test_labels = train_ours(
            model=model,
            seed=seed,
            split=split,
            device=device,
            output_dir=checkpoint_dir,
        )
    else:
        raise ValueError(method)

    report = dual_threshold_report(
        val_labels=np.asarray(val_labels, dtype=np.int64),
        val_hall_scores=np.asarray(val_scores, dtype=np.float64),
        test_labels=np.asarray(test_labels, dtype=np.int64),
        test_hall_scores=np.asarray(test_scores, dtype=np.float64),
    )
    np.savez_compressed(
        output_dir / "scores.npz",
        val_label=np.asarray(val_labels, dtype=np.int8),
        val_hall_score=np.asarray(val_scores, dtype=np.float64),
        test_label=np.asarray(test_labels, dtype=np.int8),
        test_hall_score=np.asarray(test_scores, dtype=np.float64),
    )
    return {
        "model": model,
        "method": method,
        "method_display": METHOD_NAMES[method],
        "seed": seed,
        "split_protocol": "shared_image_level_811",
        "image_counts": {name: len(values) for name, values in split.items()},
        "token_counts": {
            "val": int(len(val_labels)),
            "test": int(len(test_labels)),
        },
        "threshold_selection": {
            "real": "validation_real_f1",
            "hall": "validation_hall_f1",
        },
        "test": report["test"],
        "validation": report["validation"],
        "thresholds": report["thresholds"],
        "native_training": compact_native_metadata(native),
    }


def dense_matrices(split_records, method):
    return {
        name: build_dense_baseline_matrix(split_records[name], method)[:2]
        for name in ("train", "val", "test")
    }


def direct_scores(rows, method: str, field: str):
    scores = np.asarray(
        [float(get_baseline_payload(row, method)[field]) for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    return scores, labels


def dual_threshold_report(
    *,
    val_labels: np.ndarray,
    val_hall_scores: np.ndarray,
    test_labels: np.ndarray,
    test_hall_scores: np.ndarray,
) -> dict[str, Any]:
    if not np.isfinite(val_hall_scores).all() or not np.isfinite(test_hall_scores).all():
        raise ValueError("Scores contain non-finite values")
    real_threshold = select_detection_threshold(
        val_labels, val_hall_scores, positive_class="real"
    )
    hall_threshold = select_detection_threshold(
        val_labels, val_hall_scores, positive_class="hallucination"
    )

    def evaluate(labels, scores):
        real = evaluate_detection_scores(
            labels, scores, real_threshold, positive_class="real"
        )
        hall = evaluate_detection_scores(
            labels, scores, hall_threshold, positive_class="hallucination"
        )
        auc_gap = abs(
            float(real["real_positive"]["auc"])
            - float(hall["hallucination_positive"]["auc"])
        )
        if auc_gap > 1e-10:
            raise ValueError(f"Real/Hall AUROC mismatch: {auc_gap}")
        return {
            "accuracy_real_threshold": float(real["accuracy"]),
            "accuracy_hall_threshold": float(hall["accuracy"]),
            "auroc": float(real["real_positive"]["auc"]),
            "real": {
                key: float(real["real_positive"][key])
                for key in ("precision", "recall", "f1", "auc", "aupr")
            },
            "hall": {
                key: float(hall["hallucination_positive"][key])
                for key in ("precision", "recall", "f1", "auc", "aupr")
            },
        }

    return {
        "thresholds": {
            "real_score": float(real_threshold),
            "hall_score": float(hall_threshold),
        },
        "validation": evaluate(val_labels, val_hall_scores),
        "test": evaluate(test_labels, test_hall_scores),
    }


def train_ours(
    *,
    model: str,
    seed: int,
    split: Mapping[str, set[int]],
    device: str,
    output_dir: Path,
):
    with np.load(ours_feature_path(model), allow_pickle=False) as saved:
        matrix = np.concatenate(
            (
                np.asarray(saved["log_s_hpre_k032_tau003"], dtype=np.float32),
                np.asarray(saved["f_log"], dtype=np.float32),
            ),
            axis=1,
        )
        labels = np.asarray(saved["label"], dtype=np.int64)
        image_ids = np.asarray(saved["image_id"], dtype=np.int64)
    masks = {name: np.isin(image_ids, list(ids)) for name, ids in split.items()}
    arrays = {
        name: (matrix[mask], labels[mask]) for name, mask in masks.items()
    }
    mean = arrays["train"][0].astype(np.float64).mean(axis=0)
    scale = arrays["train"][0].astype(np.float64).std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = {
        name: ((values.astype(np.float64) - mean) / scale).astype(np.float32)
        for name, (values, _labels) in arrays.items()
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
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
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
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, 101):
        net.train()
        losses = []
        for values, targets in loader:
            values = values.to(chosen_device)
            targets = targets.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(values).squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _val_probability = predict_real_probability(
            net,
            standardized["val"],
            arrays["val"][1],
            criterion,
            chosen_device,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = float(val_loss)
            best_epoch = epoch
            stale = 0
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
        raise RuntimeError("Ours did not produce a finite checkpoint")
    net.load_state_dict(best_state)
    _, val_real = predict_real_probability(
        net,
        standardized["val"],
        arrays["val"][1],
        criterion,
        chosen_device,
    )
    _, test_real = predict_real_probability(
        net,
        standardized["test"],
        arrays["test"][1],
        criterion,
        chosen_device,
    )
    torch.save(best_state, output_dir / "model.pt")
    np.savez(output_dir / "input_scaler.npz", mean=mean, scale=scale)
    atomic_json(output_dir / "history.json", history)
    config = {
        "feature": "log_s_hpre_k032_tau003 + f_log",
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
    return (
        config,
        1.0 - val_real,
        arrays["val"][1],
        1.0 - test_real,
        arrays["test"][1],
    )


def predict_real_probability(net, matrix, labels, criterion, device):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(matrix),
            torch.from_numpy(labels.astype(np.float32)),
        ),
        batch_size=1024,
        shuffle=False,
    )
    net.eval()
    losses, probabilities = [], []
    with torch.no_grad():
        for values, targets in loader:
            values, targets = values.to(device), targets.to(device)
            logits = net(values).squeeze(-1)
            losses.append(float(criterion(logits, targets).item()) * len(targets))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return sum(losses) / len(labels), np.concatenate(probabilities).astype(np.float64)


def compact_native_metadata(native: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "paper_config",
        "input_dim",
        "best_params",
        "selection_protocol",
        "checkpoint",
        "history",
    )
    return json_ready({key: native[key] for key in keep if key in native})


def flatten_result(result: Mapping[str, Any]) -> dict[str, float]:
    test = result["test"]
    return {
        "accuracy_real_threshold": float(test["accuracy_real_threshold"]),
        "accuracy_hall_threshold": float(test["accuracy_hall_threshold"]),
        "auroc": float(test["auroc"]),
        "real_precision": float(test["real"]["precision"]),
        "real_f1": float(test["real"]["f1"]),
        "real_recall": float(test["real"]["recall"]),
        "real_aupr": float(test["real"]["aupr"]),
        "hall_precision": float(test["hall"]["precision"]),
        "hall_f1": float(test["hall"]["f1"]),
        "hall_recall": float(test["hall"]["recall"]),
        "hall_aupr": float(test["hall"]["aupr"]),
    }


def summarize(output_root: Path) -> None:
    rows = []
    missing = []
    for model in MODELS:
        for method in METHODS:
            values = []
            for seed in SEEDS:
                path = output_root / model / f"seed{seed}" / method / "result.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                values.append(flatten_result(json.loads(path.read_text())))
            if not values:
                continue
            row: dict[str, Any] = {
                "model": model,
                "model_display": MODEL_NAMES[model],
                "method": method,
                "method_display": METHOD_NAMES[method],
                "num_seeds": len(values),
            }
            for metric in METRIC_KEYS:
                metric_values = np.asarray([value[metric] for value in values])
                row[f"{metric}_mean"] = float(metric_values.mean())
                row[f"{metric}_std"] = float(metric_values.std())
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No completed result.json files")
    atomic_json(output_root / "complete_results.json", {
        "protocol": "shared image-level 8:1:1; validation-selected dual thresholds",
        "thresholds": {
            "real": "maximize validation Real-F1",
            "hall": "maximize validation Hall-F1",
        },
        "seeds": list(SEEDS),
        "rows": rows,
        "missing": missing,
    })
    fields = [
        "model", "model_display", "method", "method_display", "num_seeds"
    ] + [f"{metric}_{suffix}" for metric in METRIC_KEYS for suffix in ("mean", "std")]
    with (output_root / "complete_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "complete_results.md").write_text(
        render_markdown(rows, missing), encoding="utf-8"
    )
    write_master_progress(output_root, missing)
    print(f"[811] summarized {len(rows)} model-method rows; missing={len(missing)}")


def render_markdown(rows, missing):
    lines = [
        "# COCO4000-512 ENDAC Baselines + Ours — 8:1:1 dual thresholds",
        "",
        "- Shared image-level split: 3200 train / 400 validation / 400 test.",
        "- Real threshold maximizes validation Real-F1; Hall threshold maximizes validation Hall-F1.",
        "- Precision/Recall/F1 and Accuracy use their corresponding class threshold; AUROC/AUPR are threshold-free.",
        "- TokenGrounding uses the native sklearn MLP grid. DHCP is excluded.",
        "- Values are mean ± population standard deviation over seeds 43/44/45.",
        "",
    ]
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        lines.extend([
            f"## {MODEL_NAMES[model]}",
            "",
            "| Method | Acc@Real | Acc@Hall | AUROC | Real Precision | Real F1 | Real Recall | Real AUPR | Hall Precision | Hall F1 | Hall Recall | Hall AUPR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in selected:
            values = [format_stat(row, metric) for metric in METRIC_KEYS]
            lines.append(f"| {row['method_display']} | " + " | ".join(values) + " |")
        lines.append("")
    if missing:
        lines.extend(["## Missing", "", *[f"- `{path}`" for path in missing], ""])
    return "\n".join(lines)


def format_stat(row, metric):
    return f"{100 * row[f'{metric}_mean']:.2f} ± {100 * row[f'{metric}_std']:.2f}"


def write_model_progress(output_root: Path, model: str) -> None:
    complete = []
    for seed in SEEDS:
        methods = [
            method
            for method in METHODS
            if (output_root / model / f"seed{seed}" / method / "result.json").exists()
        ]
        complete.append((seed, methods))
    lines = [
        f"# {MODEL_NAMES[model]} 8:1:1 progress",
        "",
        "| Seed | Completed | Progress |",
        "|---:|---|---:|",
    ]
    for seed, methods in complete:
        lines.append(
            f"| {seed} | {', '.join(methods) if methods else '—'} | "
            f"{len(methods)}/{len(METHODS)} |"
        )
    atomic_text(output_root / model / "progress.md", "\n".join(lines) + "\n")


def write_master_progress(output_root: Path, missing: Sequence[str]) -> None:
    completed = len(MODELS) * len(METHODS) * len(SEEDS) - len(missing)
    total = len(MODELS) * len(METHODS) * len(SEEDS)
    width = 30
    filled = round(width * completed / total)
    bar = "█" * filled + "░" * (width - filled)
    atomic_text(
        output_root / "progress.md",
        "\n".join([
            "# ENDAC 8:1:1 dual-threshold progress",
            "",
            f"`[{bar}] {completed}/{total} ({100 * completed / total:.1f}%)`",
            "",
            f"Missing runs: {len(missing)}",
            "",
        ]),
    )


def load_pickle(path: Path):
    import pickle
    with path.open("rb") as handle:
        return pickle.load(handle)


def compact_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return compact_path(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
