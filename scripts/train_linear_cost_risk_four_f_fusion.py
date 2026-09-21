#!/usr/bin/env python3
"""Train four linear-cost risks alone and fused with four final-norm F curves."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.train_fixed_wcost_log_gate_log_am import (  # noqa: E402
    atomic_json,
    atomic_npz,
    report,
)
from scripts.train_iclr_endac_attention_i_fusion import _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
    "minigpt4_7b",
    "shikra_7b",
)
RISK_NAMES = ("log_s_hpre", "log_s_hmid", "original_hpre", "original_hmid")
F_NAMES = (
    "sqrt_residual_fusion",
    "log_residual_fusion",
    "mass_preserving_fusion",
    "asinh_sqrt_fusion",
)
SEEDS = (43, 44, 45)
SOURCE = GF_ROOT / "outputs/coco4000_512_linear_cost_log_s_risk_811"
OUT = ROOT / "outputs/linear_cost_log_s_risk_four_f_fusion"
GROUPS = (
    tuple(f"risk__{risk}" for risk in RISK_NAMES)
    + tuple(f"f__{feature}" for feature in F_NAMES)
    + tuple(f"{risk}__{feature}" for risk in RISK_NAMES for feature in F_NAMES)
)
EXPECTED = len(GROUPS) * len(SEEDS)


def _key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def _compute_four_f(
    probability: np.ndarray,
    am_norm: np.ndarray,
    vocabulary_size: int,
) -> dict[str, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    am_norm = np.asarray(am_norm, dtype=np.float64)
    if probability.shape != am_norm.shape or probability.ndim != 2:
        raise ValueError(f"p/a_m shape mismatch: {probability.shape}/{am_norm.shape}")
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(am_norm).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or (am_norm < 0).any()
    ):
        raise ValueError("invalid p or a_m values")
    mass = probability.sum(axis=1)
    root_probability = np.sqrt(probability)
    root_sum = root_probability.sum(axis=1)
    token_count = probability.shape[1]
    q = np.full_like(probability, 1.0 / token_count)
    np.divide(root_probability, root_sum[:, None], out=q, where=root_sum[:, None] > 0)
    reshaped_probability = mass[:, None] * q
    i_sum = am_norm.sum(axis=1)
    sqrt_cp = np.sqrt(float(vocabulary_size) * probability)
    values = {
        "sqrt_residual_fusion": (am_norm * (1.0 + sqrt_cp)).sum(axis=1),
        "log_residual_fusion": (
            am_norm * (1.0 + np.log1p(float(vocabulary_size) * probability))
        ).sum(axis=1),
        "mass_preserving_fusion": i_sum + float(vocabulary_size) * (
            am_norm * reshaped_probability
        ).sum(axis=1),
        "asinh_sqrt_fusion": (
            am_norm * (1.0 + np.arcsinh(sqrt_cp))
        ).sum(axis=1),
    }
    if not np.allclose(reshaped_probability.sum(axis=1), mass, rtol=2e-6, atol=1e-12):
        raise ValueError("mass-preserving transform changed semantic mass")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("non-finite F feature")
    return {name: value.astype(np.float32) for name, value in values.items()}


def prepare(model: str, model_out: Path) -> tuple[dict[str, np.ndarray], dict]:
    cache = model_out / "compact_features.npz"
    audit_path = model_out / "feature_audit.json"
    if cache.exists() and audit_path.exists():
        with np.load(cache, allow_pickle=False) as saved:
            data = {key: np.asarray(saved[key]) for key in saved.files}
        return data, json.loads(audit_path.read_text())

    source_path = SOURCE / model / "features.pkl"
    summary_path = SOURCE / model / "summary.json"
    if not source_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"{model}: extraction has not been merged")
    rows = load_pkl(source_path)
    source_summary = json.loads(summary_path.read_text())
    if len(rows) != int(source_summary["samples"]):
        raise ValueError(f"{model}: source row count differs from summary")
    keys = [_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate target keys")
    first = rows[0]
    layers = int(np.asarray(first["risks"][RISK_NAMES[0]]).shape[0])
    count = len(rows)
    risks = {name: np.empty((count, layers), dtype=np.float32) for name in RISK_NAMES}
    features = {name: np.empty((count, layers), dtype=np.float32) for name in F_NAMES}
    labels = np.empty(count, dtype=np.int8)
    image_ids = np.empty(count, dtype=np.int64)
    response_indices = np.empty(count, dtype=np.int32)
    target_ids = np.empty(count, dtype=np.int32)
    train_mask = np.empty(count, dtype=bool)
    val_mask = np.empty(count, dtype=bool)
    test_only_mask = np.empty(count, dtype=bool)
    vocabulary_sizes = set()

    for index, row in enumerate(rows):
        matrices = row["matrices"]
        vocabulary_size = int(row["vocabulary_size"])
        vocabulary_sizes.add(vocabulary_size)
        current_f = _compute_four_f(
            matrices["object_probability_norm_hpre"],
            matrices["am_norm"],
            vocabulary_size,
        )
        for name in RISK_NAMES:
            value = np.asarray(row["risks"][name], dtype=np.float32)
            if value.shape != (layers,) or not np.isfinite(value).all():
                raise ValueError(f"{model}/{row['sample_id']}: invalid {name}")
            risks[name][index] = value
        for name in F_NAMES:
            if current_f[name].shape != (layers,):
                raise ValueError(f"{model}/{row['sample_id']}: invalid {name}")
            features[name][index] = current_f[name]
        split = str(row["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"{model}: unknown split {split!r}")
        labels[index] = int(row["label"])
        image_ids[index] = int(row["image_id"])
        response_indices[index] = int(row["response_token_idx"])
        target_ids[index] = int(row["target_token_id"])
        train_mask[index] = split == "train"
        val_mask[index] = split == "val"
        test_only_mask[index] = split == "test"
    if len(vocabulary_sizes) != 1:
        raise ValueError(f"{model}: inconsistent vocabulary sizes {vocabulary_sizes}")
    strict_test_mask = val_mask | test_only_mask
    train_images = set(image_ids[train_mask].tolist())
    test_images = set(image_ids[strict_test_mask].tolist())
    if train_images & test_images:
        raise ValueError(f"{model}: train/test image leakage")
    data = {
        **risks,
        **features,
        "label": labels,
        "image_id": image_ids,
        "response_token_idx": response_indices,
        "target_token_id": target_ids,
        "train_mask": train_mask,
        "strict_test_mask": strict_test_mask,
        "val_mask": val_mask,
        "test_only_mask": test_only_mask,
    }
    atomic_npz(cache, **data)
    audit = {
        "model": model,
        "source": str(source_path),
        "schema": source_summary["schema"],
        "targets": count,
        "layers": layers,
        "vocabulary_size": next(iter(vocabulary_sizes)),
        "split_targets": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_only_mask.sum()),
            "strict_test": int(strict_test_mask.sum()),
        },
        "split_images": {"train": len(train_images), "strict_test": len(test_images)},
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "risk_names": list(RISK_NAMES),
        "f_names": list(F_NAMES),
        "probability_field": "matrices.object_probability_norm_hpre",
        "am_norm_field": "matrices.am_norm",
        "strict_protocol": "train versus val+test (3200/800 images)",
    }
    atomic_json(audit_path, audit)
    del rows
    gc.collect()
    return data, audit


def _matrices(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    matrices = {f"risk__{name}": data[name] for name in RISK_NAMES}
    matrices.update({f"f__{name}": data[name] for name in F_NAMES})
    matrices.update({
        f"{risk}__{feature}": np.concatenate((data[risk], data[feature]), axis=1)
        for risk in RISK_NAMES
        for feature in F_NAMES
    })
    if tuple(matrices) != GROUPS:
        raise AssertionError("feature-group order changed")
    return matrices


def summarize(model: str, model_out: Path) -> None:
    rows = []
    for group in GROUPS:
        cases = [
            json.loads((model_out / group / f"seed{seed}" / "metrics.json").read_text())
            for seed in SEEDS
        ]
        for threshold in ("fixed_0.5", "train_f1"):
            reports = [report(case, threshold) for case in cases]
            row = {"model": model, "threshold": threshold, "feature_group": group}
            for metric, (side, key) in {
                "real_f1": ("real_positive", "f1"),
                "real_auc": ("real_positive", "auc"),
                "hall_aupr": ("hallucination_positive", "aupr"),
            }.items():
                values = np.asarray([item[side][key] for item in reports], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std())
            rows.append(row)
    by_key = {(row["threshold"], row["feature_group"]): row for row in rows}
    for row in rows:
        group = row["feature_group"]
        row["delta_real_auc_vs_risk"] = ""
        row["delta_hall_aupr_vs_risk"] = ""
        row["delta_real_auc_vs_f"] = ""
        row["delta_hall_aupr_vs_f"] = ""
        if group.startswith("risk__") or group.startswith("f__"):
            continue
        risk, feature = group.split("__", 1)
        risk_base = by_key[(row["threshold"], f"risk__{risk}")]
        f_base = by_key[(row["threshold"], f"f__{feature}")]
        row["delta_real_auc_vs_risk"] = row["real_auc_mean"] - risk_base["real_auc_mean"]
        row["delta_hall_aupr_vs_risk"] = row["hall_aupr_mean"] - risk_base["hall_aupr_mean"]
        row["delta_real_auc_vs_f"] = row["real_auc_mean"] - f_base["real_auc_mean"]
        row["delta_hall_aupr_vs_f"] = row["hall_aupr_mean"] - f_base["hall_aupr_mean"]
    with (model_out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: linear-cost risks × four F features", "",
        "Strict 3200/800 image split; seeds 43–45; one hidden layer 128; "
        "training-only column z-score; dropout 0.3; no BatchNorm.", "",
        "The table uses the train-F1 threshold. Deltas are against the matching "
        "risk-only and F-only heads.", "",
        "| Feature | Real F1 | Real AUC | Hall AUPR | ΔAUC vs risk | ΔAUPR vs risk | ΔAUC vs F | ΔAUPR vs F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["threshold"] != "train_f1":
            continue
        def delta(name: str) -> str:
            return "—" if row[name] == "" else f"{float(row[name]):+.4f}"
        lines.append(
            f"| `{row['feature_group']}` | {row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{delta('delta_real_auc_vs_risk')} | {delta('delta_hall_aupr_vs_risk')} | "
            f"{delta('delta_real_auc_vs_f')} | {delta('delta_hall_aupr_vs_f')} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")


def train(model: str, device_name: str, output_root: Path, prepare_only: bool) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    data, audit = prepare(model, model_out)
    if prepare_only:
        print(f"PREPARED {model}", flush=True)
        return
    matrices = _matrices(data)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    example = replace(
        _probe_config(root_config, SEEDS[0]),
        hidden_sizes=(128,),
        batch_norm=False,
        standardize_input=True,
    )
    atomic_json(model_out / "protocol.json", {
        **audit,
        "probe": _json_ready(asdict(example)),
        "seeds": list(SEEDS),
        "feature_groups": {name: list(value.shape) for name, value in matrices.items()},
        "total_heads": EXPECTED,
    })
    completed = sum(1 for _ in model_out.glob("*/seed*/metrics.json"))
    labels = data["label"]
    train_mask = data["train_mask"]
    test_mask = data["strict_test_mask"]
    for group, matrix in matrices.items():
        for seed in SEEDS:
            case = model_out / group / f"seed{seed}"
            metrics_path = case / "metrics.json"
            if metrics_path.exists():
                continue
            config = replace(
                _probe_config(root_config, seed),
                hidden_sizes=(128,),
                batch_norm=False,
                standardize_input=True,
            )
            metrics = train_and_evaluate_probe(
                X_train=matrix[train_mask],
                y_train=labels[train_mask],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=matrix[test_mask],
                y_test=labels[test_mask],
                config=config,
                device=torch.device(device_name),
                output_dir=str(case),
            )
            atomic_json(metrics_path, metrics)
            completed += 1
            atomic_json(model_out / "progress.json", {
                "model": model,
                "stage": "training",
                "completed": completed,
                "total": EXPECTED,
                "feature_group": group,
                "seed": seed,
            })
            print("DONE", model, group, seed, flush=True)
    summarize(model, model_out)
    atomic_json(model_out / "progress.json", {
        "model": model,
        "stage": "complete",
        "completed": EXPECTED,
        "total": EXPECTED,
    })


def summarize_all(output_root: Path) -> None:
    rows = []
    for model in MODELS:
        path = output_root / model / "summary.csv"
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError("No completed model summaries")
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fusion_rows = [
        row for row in rows
        if row["threshold"] == "train_f1"
        and not row["feature_group"].startswith(("risk__", "f__"))
    ]
    ranked_by_model = {}
    for model in MODELS:
        ranked_by_model[model] = sorted(
            (row for row in fusion_rows if row["model"] == model),
            key=lambda row: float(row["real_auc_mean"]),
            reverse=True,
        )
    groups = sorted({row["feature_group"] for row in fusion_rows})
    common_rows = []
    for group in groups:
        group_rows = [row for row in fusion_rows if row["feature_group"] == group]
        ranks = [
            next(
                rank for rank, candidate in enumerate(ranked_by_model[row["model"]], 1)
                if candidate["feature_group"] == group
            )
            for row in group_rows
        ]
        common_rows.append({
            "feature_group": group,
            "mean_auc": float(np.mean([float(row["real_auc_mean"]) for row in group_rows])),
            "mean_rank": float(np.mean(ranks)),
        })
    common_rows.sort(key=lambda row: (row["mean_rank"], -row["mean_auc"]))
    auc_vs_risk = [float(row["delta_real_auc_vs_risk"]) for row in fusion_rows]
    auc_vs_f = [float(row["delta_real_auc_vs_f"]) for row in fusion_rows]
    aupr_vs_risk = [float(row["delta_hall_aupr_vs_risk"]) for row in fusion_rows]
    aupr_vs_f = [float(row["delta_hall_aupr_vs_f"]) for row in fusion_rows]
    lines = [
        "# Linear-cost risks × four final-norm F features", "",
        "Train-F1 threshold; AUROC and HALL-AUPR are means across seeds 43–45.", "",
        "## Overview", "",
        "| Model | Best risk + F | Real F1 | Real AUC | Hall AUPR | ΔAUC vs risk | ΔAUC vs F |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        if not ranked_by_model[model]:
            continue
        row = ranked_by_model[model][0]
        lines.append(
            f"| {model} | `{row['feature_group']}` | "
            f"{float(row['real_f1_mean']):.4f}±{float(row['real_f1_std']):.4f} | "
            f"{float(row['real_auc_mean']):.4f}±{float(row['real_auc_std']):.4f} | "
            f"{float(row['hall_aupr_mean']):.4f}±{float(row['hall_aupr_std']):.4f} | "
            f"{float(row['delta_real_auc_vs_risk']):+.4f} | "
            f"{float(row['delta_real_auc_vs_f']):+.4f} |"
        )
    lines += [
        "",
        f"Across the {len(fusion_rows)} model/combination cases, fusion improves AUROC "
        f"over its matching risk in {sum(value > 0 for value in auc_vs_risk)}/{len(auc_vs_risk)} "
        f"cases and over its matching F in {sum(value > 0 for value in auc_vs_f)}/{len(auc_vs_f)} cases. "
        f"It improves HALL-AUPR in {sum(value > 0 for value in aupr_vs_risk)}/{len(aupr_vs_risk)} "
        f"and {sum(value > 0 for value in aupr_vs_f)}/{len(aupr_vs_f)} cases, respectively.",
        "",
        "### Best common combinations", "",
        "Mean AUC and rank are descriptive averages across six separately evaluated models.", "",
        "| Rank | Risk + F | Mean model rank | Mean Real AUC |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(common_rows[:5], 1):
        lines.append(
            f"| {rank} | `{row['feature_group']}` | {row['mean_rank']:.2f} | "
            f"{row['mean_auc']:.4f} |"
        )
    lines.append("")
    for model in MODELS:
        model_rows = ranked_by_model[model]
        if not model_rows:
            continue
        lines += [
            f"## {model}", "",
            "| Rank | Risk + F | Real AUC | Hall AUPR | ΔAUC vs risk | ΔAUC vs F |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(model_rows, 1):
            lines.append(
                f"| {rank} | `{row['feature_group']}` | {float(row['real_auc_mean']):.4f} | "
                f"{float(row['hall_aupr_mean']):.4f} | "
                f"{float(row['delta_real_auc_vs_risk']):+.4f} | "
                f"{float(row['delta_real_auc_vs_f']):+.4f} |"
            )
        lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-all", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.summarize_all:
        summarize_all(output_root)
        return
    if args.model is None:
        parser.error("--model is required unless --summarize-all is used")
    train(args.model, args.device, output_root, args.prepare_only)


if __name__ == "__main__":
    main()
