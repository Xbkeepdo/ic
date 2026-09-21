#!/usr/bin/env python3
"""Train mass-preserving/asinh semantic features alone and with fixed weighted-cost risk."""

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
    RUN,
    atomic_json,
    fixed_risk_spec,
    report,
    target_key,
)
from scripts.train_iclr_endac_attention_i_fusion import MODELS, _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe  # noqa: E402
from scripts.train_weighted_cost_attention_i_fusion import SEEDS, _baseline_path  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


FEATURE_ROOT = ROOT / "outputs/mass_preserving_asinh_semantic_curves_by_label"
OUT = ROOT / "outputs/fixed_wcost_mass_preserving_asinh_semantic_fusion"
GF_PROBABILITY_ROOT = GF_ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
GROUPS = ("mass_preserving_fusion", "risk_mass_preserving_fusion", "asinh_sqrt_fusion", "risk_asinh_sqrt_fusion")
EXPECTED = len(GROUPS) * len(SEEDS)


def prepare(model: str, model_out: Path) -> tuple[dict[str, np.ndarray], dict]:
    aligned_path = model_out / "aligned_features.npz"
    audit_path = model_out / "alignment.json"
    if aligned_path.exists() and audit_path.exists():
        with np.load(aligned_path, allow_pickle=False) as saved:
            return {key: np.asarray(saved[key]) for key in saved.files}, json.loads(audit_path.read_text())

    with np.load(FEATURE_ROOT / model / "compact_features.npz", allow_pickle=False) as saved:
        compact = {key: np.asarray(saved[key]) for key in saved.files}
    semantic_rows = load_pkl(GF_PROBABILITY_ROOT / model / "semantic_attention.pkl")
    risk_rows = load_pkl(ROOT / "outputs" / model / RUN / "features.pkl")
    spec = fixed_risk_spec()
    risk_by_key = {target_key(row): row for row in risk_rows}
    semantic_keys = [target_key(row) for row in semantic_rows]
    if len(risk_by_key) != len(risk_rows) or set(semantic_keys) != set(risk_by_key):
        raise ValueError(f"{model}: risk and semantic target keys differ")
    aligned = [risk_by_key[key] for key in semantic_keys]
    risk = np.stack([np.asarray(row[spec["field"]], dtype=np.float32) for row in aligned])
    labels = np.asarray([int(row["label"]) for row in semantic_rows], dtype=np.int8)
    image_ids = np.asarray([int(row["image_id"]) for row in semantic_rows], dtype=np.int64)
    if not (np.array_equal(labels, compact["label"])
            and np.array_equal(image_ids, compact["image_id"])):
        raise ValueError(f"{model}: compact feature order differs from semantic source")
    if any(int(left["label"]) != int(right["label"]) for left, right in zip(semantic_rows, aligned)):
        raise ValueError(f"{model}: aligned risk label mismatch")
    mass_preserving = np.asarray(compact["mass_preserving_fusion"], dtype=np.float32)
    asinh_sqrt = np.asarray(compact["asinh_sqrt_fusion"], dtype=np.float32)
    if (risk.shape != mass_preserving.shape or risk.shape != asinh_sqrt.shape
            or not np.isfinite(risk).all() or not np.isfinite(mass_preserving).all()
            or not np.isfinite(asinh_sqrt).all()):
        raise ValueError(
            f"{model}: incompatible risk/feature shapes "
            f"{risk.shape}/{mass_preserving.shape}/{asinh_sqrt.shape}"
        )
    data = {
        "risk": risk,
        "mass_preserving_fusion": mass_preserving,
        "asinh_sqrt_fusion": asinh_sqrt,
        "label": labels,
        "image_id": image_ids,
        "train_mask": np.asarray(compact["train_mask"], dtype=bool),
        "test_mask": np.asarray(compact["test_mask"], dtype=bool),
    }
    np.savez_compressed(aligned_path, **data)
    audit = {
        "model": model,
        "targets": len(labels),
        "train_targets": int(data["train_mask"].sum()),
        "test_targets": int(data["test_mask"].sum()),
        "layers": risk.shape[1],
        "risk": spec,
        "risk_source": str(ROOT / "outputs" / model / RUN / "features.pkl"),
        "feature_source": str(FEATURE_ROOT / model / "compact_features.npz"),
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "matched_targets": len(aligned),
    }
    atomic_json(audit_path, audit)
    del semantic_rows, risk_rows, risk_by_key, aligned
    gc.collect()
    return data, audit


def summarize(model: str, model_out: Path) -> None:
    spec = fixed_risk_spec()
    sources = {
        "risk": [_baseline_path(model, spec, seed) for seed in SEEDS],
        **{group: [model_out / group / f"seed{seed}" / "metrics.json" for seed in SEEDS]
           for group in GROUPS},
    }
    rows = []
    for group, paths in sources.items():
        cases = [json.loads(path.read_text()) for path in paths]
        for threshold in ("fixed_0.5", "train_f1"):
            reports = [report(case, threshold) for case in cases]
            row = {"model": model, "threshold": threshold, "feature_group": group}
            for metric, (side, key) in {
                "real_f1": ("real_positive", "f1"),
                "real_auc": ("real_positive", "auc"),
                "hall_aupr": ("hallucination_positive", "aupr"),
            }.items():
                values = np.asarray([value[side][key] for value in reports], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std())
            rows.append(row)
    baselines = {row["threshold"]: row for row in rows if row["feature_group"] == "risk"}
    for row in rows:
        base = baselines[row["threshold"]]
        row["delta_real_auc_vs_risk"] = row["real_auc_mean"] - base["real_auc_mean"]
        row["delta_hall_aupr_vs_risk"] = row["hall_aupr_mean"] - base["hall_aupr_mean"]
    with (model_out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = [
        f"# {model}: mass-preserving and asinh-sqrt semantic fusion", "",
        "`F_MP=I+C*sum_m I_m*p_tilde_m`; `F_AS=sum_m I_m*(1+asinh(sqrt(C*p_m)))`; fixed risk is WCOST-ALL / softmax / hpre_raw_logit_gauss / w=0.2.",
        "Probe: one hidden layer 128, training-only column z-score, dropout 0.3, no BatchNorm, seeds 43–45.", "",
        "| Threshold | Feature | Real F1 | Real AUC | Δ AUC vs risk | Hall AUPR | Δ Hall AUPR vs risk |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['feature_group']} | "
            f"{row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | {row['delta_real_auc_vs_risk']:+.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | {row['delta_hall_aupr_vs_risk']:+.4f} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")


def train(model: str, device_name: str, output_root: Path) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    data, audit = prepare(model, model_out)
    matrices = {
        "mass_preserving_fusion": data["mass_preserving_fusion"],
        "risk_mass_preserving_fusion": np.concatenate(
            (data["risk"], data["mass_preserving_fusion"]), axis=1
        ),
        "asinh_sqrt_fusion": data["asinh_sqrt_fusion"],
        "risk_asinh_sqrt_fusion": np.concatenate(
            (data["risk"], data["asinh_sqrt_fusion"]), axis=1
        ),
    }
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    example = replace(_probe_config(root_config, SEEDS[0]), hidden_sizes=(128,),
                      batch_norm=False, standardize_input=True)
    atomic_json(model_out / "protocol.json", {
        **audit, "probe": _json_ready(asdict(example)), "seeds": list(SEEDS),
        "feature_groups": {name: list(value.shape) for name, value in matrices.items()},
        "risk_only": "reused existing weighted_cost_probe metrics",
    })
    completed = sum(1 for _ in model_out.glob("*/seed*/metrics.json"))
    labels = data["label"]
    for group, matrix in matrices.items():
        for seed in SEEDS:
            case = model_out / group / f"seed{seed}"
            path = case / "metrics.json"
            if path.exists():
                continue
            config = replace(_probe_config(root_config, seed), hidden_sizes=(128,),
                             batch_norm=False, standardize_input=True)
            metrics = train_and_evaluate_probe(
                X_train=matrix[data["train_mask"]], y_train=labels[data["train_mask"]],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=matrix[data["test_mask"]], y_test=labels[data["test_mask"]],
                config=config, device=torch.device(device_name), output_dir=str(case),
            )
            path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
            completed += 1
            atomic_json(model_out / "progress.json", {
                "model": model, "stage": "training", "completed": completed,
                "total": EXPECTED, "feature_group": group, "seed": seed,
            })
            print("DONE", model, group, seed, flush=True)
    summarize(model, model_out)
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "complete", "completed": EXPECTED, "total": EXPECTED,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    train(args.model, args.device, args.output_root.resolve())


if __name__ == "__main__":
    main()
