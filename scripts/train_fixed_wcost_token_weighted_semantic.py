#!/usr/bin/env python3
"""Train sum_v p_raw(v,y)||a_v|| alone and with one fixed weighted-cost risk."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import MODELS, _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from scripts.train_weighted_cost_attention_i_fusion import (  # noqa: E402
    GF_RUN,
    GF_SOURCE,
    SEEDS,
    _baseline_path,
    _load_model,
    _risk_specs,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


OUT = ROOT / "outputs/fixed_wcost_token_weighted_semantic"
SEMANTIC_SOURCE = GF_ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
RISK_NAME = "wcost_all__softmax__hpre_raw_logit_gauss__w02"
FEATURE_GROUPS = ("raw_semantic_am_norm_sum", "risk_raw_semantic_am_norm_sum")
EXPECTED_CASES = len(FEATURE_GROUPS) * len(SEEDS)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fixed_risk_spec() -> dict:
    matches = [spec for spec in _risk_specs() if spec["name"] == RISK_NAME]
    if len(matches) != 1:
        raise AssertionError(f"Expected one fixed risk spec, found {len(matches)}")
    return matches[0]


def prepare_feature(model: str, model_out: Path) -> tuple[np.ndarray, dict]:
    cache = model_out / "raw_semantic_am_norm_sum.npz"
    audit_path = model_out / "feature_audit.json"
    if cache.exists() and audit_path.exists():
        saved = np.load(cache, allow_pickle=False)
        feature = np.asarray(saved["feature"], dtype=np.float32)
        audit = json.loads(audit_path.read_text())
        if feature.shape != tuple(audit["feature_shape"]) or not np.isfinite(feature).all():
            raise ValueError(f"{model}: invalid cached token-weighted feature")
        return feature, audit

    semantic_path = SEMANTIC_SOURCE / model / "semantic_attention.pkl"
    am_path = GF_SOURCE / model / "am_norms/features.pkl"
    reference_path = GF_RUN / model / "features.pt"
    print(f"LOAD semantic matrices: {semantic_path}", flush=True)
    semantic_rows = load_pkl(semantic_path)
    print(f"LOAD a_m norms: {am_path}", flush=True)
    am_rows = load_pkl(am_path)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False, mmap=True)
    reference_ids = list(reference["sample_id"])
    if len(semantic_rows) != len(reference_ids) or len(am_rows) != len(reference_ids):
        raise ValueError(
            f"{model}: row count differs: semantic={len(semantic_rows)}, "
            f"am={len(am_rows)}, reference={len(reference_ids)}"
        )
    semantic_ids = [row["sample_id"] for row in semantic_rows]
    if semantic_ids != reference_ids:
        raise ValueError(f"{model}: semantic matrix order differs from reference")
    am_by_id = {row["sample_id"]: row for row in am_rows}
    if len(am_by_id) != len(reference_ids) or set(am_by_id) != set(reference_ids):
        raise ValueError(f"{model}: a_m IDs differ from reference")

    layers = int(np.asarray(reference["I"]).shape[1])
    feature = np.empty((len(reference_ids), layers), dtype=np.float32)
    max_semantic_mean_error = 0.0
    max_i_error = 0.0
    reference_semantic = np.asarray(reference["all_visual"]["raw"], dtype=np.float32)[:, :, 0]
    reference_i = np.asarray(reference["I"], dtype=np.float32)
    for index, semantic_row in enumerate(semantic_rows):
        am_row = am_by_id[semantic_row["sample_id"]]
        probability = np.asarray(
            semantic_row["matrices"]["object_probability_raw"], dtype=np.float32
        )
        am_norm = np.asarray(am_row["am_norm"], dtype=np.float32)
        if probability.shape != am_norm.shape or probability.shape[0] != layers:
            raise ValueError(
                f"{model}/{semantic_row['sample_id']}: shape mismatch "
                f"{probability.shape} vs {am_norm.shape}"
            )
        if not (np.isfinite(probability).all() and np.isfinite(am_norm).all()):
            raise ValueError(f"{model}/{semantic_row['sample_id']}: nonfinite source")
        feature[index] = np.sum(
            probability.astype(np.float64) * am_norm.astype(np.float64), axis=1
        ).astype(np.float32)
        semantic_error = np.max(np.abs(probability.mean(axis=1) - reference_semantic[index]))
        i_error = np.max(np.abs(am_norm.sum(axis=1, dtype=np.float64) - reference_i[index]))
        max_semantic_mean_error = max(max_semantic_mean_error, float(semantic_error))
        max_i_error = max(max_i_error, float(i_error))

    if not np.isfinite(feature).all() or (feature < 0).any():
        raise ValueError(f"{model}: invalid token-weighted feature")
    if max_semantic_mean_error > 1e-6 or max_i_error > 1e-4:
        raise ValueError(
            f"{model}: source reconstruction differs: semantic={max_semantic_mean_error}, "
            f"I={max_i_error}"
        )
    audit = {
        "schema": "raw-semantic-am-norm-sum-v1",
        "model": model,
        "definition": "sum_visual_token object_probability_raw * ||a_m||_2, per layer",
        "semantic_source": str(semantic_path),
        "am_norm_source": str(am_path),
        "reference": str(reference_path),
        "matched_rows": len(reference_ids),
        "feature_shape": list(feature.shape),
        "min": float(feature.min()),
        "max": float(feature.max()),
        "mean": float(feature.mean()),
        "max_raw_semantic_mean_reconstruction_error": max_semantic_mean_error,
        "max_I_reconstruction_error": max_i_error,
    }
    atomic_npz(cache, feature=feature)
    atomic_json(audit_path, audit)
    del semantic_rows, am_rows, am_by_id, reference
    gc.collect()
    return feature, audit


def report(metrics: dict, threshold: str) -> dict:
    return metrics["threshold_reports"][threshold]["test_metrics"]


def summarize(model: str, model_out: Path) -> None:
    spec = fixed_risk_spec()
    sources = {
        "risk": [_baseline_path(model, spec, seed) for seed in SEEDS],
        **{
            name: [model_out / name / f"seed{seed}" / "metrics.json" for seed in SEEDS]
            for name in FEATURE_GROUPS
        },
    }
    rows = []
    for group, paths in sources.items():
        cases = [json.loads(path.read_text()) for path in paths]
        for threshold in ("fixed_0.5", "train_f1"):
            reports = [report(case, threshold) for case in cases]
            row = {
                "model": model,
                "threshold": threshold,
                "feature_group": group,
                "run": spec["run"],
                "source": spec["source"],
                "method": spec["method"],
                "weight": spec["weight"],
            }
            for metric_name, (side, key) in {
                "real_f1": ("real_positive", "f1"),
                "real_auc": ("real_positive", "auc"),
                "hall_aupr": ("hallucination_positive", "aupr"),
            }.items():
                values = np.asarray([value[side][key] for value in reports], dtype=np.float64)
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(values.std())
            rows.append(row)
    baselines = {
        row["threshold"]: row for row in rows if row["feature_group"] == "risk"
    }
    for row in rows:
        baseline = baselines[row["threshold"]]
        row["delta_real_auc_vs_risk"] = row["real_auc_mean"] - baseline["real_auc_mean"]
        row["delta_hall_aupr_vs_risk"] = row["hall_aupr_mean"] - baseline["hall_aupr_mean"]
    with (model_out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: token-weighted raw semantic feature",
        "",
        "New feature: sum over visual tokens of raw semantic probability × ||a_m||, per layer.",
        "Fixed risk: WCOST-ALL / softmax / hpre_raw_logit_gauss / w=0.2. "
        "Probe: one hidden layer 128, training-only z-score, dropout 0.3, no BatchNorm, seeds 43–45.",
        "",
        "| Threshold | Feature group | Real F1 | Real AUC | Δ AUC vs risk | Hall AUPR | Δ Hall AUPR vs risk |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['feature_group']} | "
            f"{row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | "
            f"{row['delta_real_auc_vs_risk']:+.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{row['delta_hall_aupr_vs_risk']:+.4f} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")


def train(model: str, device: str, output_root: Path, prepare_only: bool) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    feature, feature_audit = prepare_feature(model, model_out)
    if prepare_only:
        print(json.dumps(feature_audit, indent=2), flush=True)
        return
    data, source_audit = _load_model(model)
    risk = data["risks"][RISK_NAME]
    if feature.shape != risk.shape:
        raise ValueError(f"{model}: new feature {feature.shape} differs from risk {risk.shape}")
    matrices = {
        "raw_semantic_am_norm_sum": feature,
        "risk_raw_semantic_am_norm_sum": np.concatenate((risk, feature), axis=1).astype(
            np.float32, copy=False
        ),
    }
    audit = {
        **source_audit,
        "schema": "fixed-wcost-token-weighted-semantic-v1",
        "fixed_risk": fixed_risk_spec(),
        "new_feature": feature_audit,
        "feature_groups": {name: list(value.shape) for name, value in matrices.items()},
        "training_case_count": EXPECTED_CASES,
    }
    atomic_json(model_out / "alignment.json", audit)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    example = replace(
        _probe_config(root_config, SEEDS[0]),
        hidden_sizes=(128,), batch_norm=False, standardize_input=True,
    )
    atomic_json(model_out / "protocol.json", {
        **audit,
        "probe": _json_ready(asdict(example)),
        "seeds": list(SEEDS),
        "risk_only": "reused existing weighted_cost_probe metrics",
    })
    completed = sum(1 for _ in model_out.glob("*/seed*/metrics.json"))
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "training", "completed": completed, "total": EXPECTED_CASES,
    })
    labels = data["label"]
    for name, matrix in matrices.items():
        for seed in SEEDS:
            case = model_out / name / f"seed{seed}"
            metrics_path = case / "metrics.json"
            if metrics_path.exists():
                print("SKIP", model, name, seed, flush=True)
                continue
            config = replace(
                _probe_config(root_config, seed),
                hidden_sizes=(128,), batch_norm=False, standardize_input=True,
            )
            metrics = train_and_evaluate_probe(
                X_train=matrix[data["train_mask"]],
                y_train=labels[data["train_mask"]],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=matrix[data["test_mask"]],
                y_test=labels[data["test_mask"]],
                config=config,
                device=torch.device(device),
                output_dir=str(case),
            )
            metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
            completed += 1
            atomic_json(model_out / "progress.json", {
                "model": model, "stage": "training", "completed": completed,
                "total": EXPECTED_CASES, "feature_group": name, "seed": seed,
            })
            print("DONE", model, name, seed, flush=True)
    summarize(model, model_out)
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "complete", "completed": EXPECTED_CASES, "total": EXPECTED_CASES,
    })
    print("SUMMARY", model_out / "summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    train(args.model, args.device, args.output_root.resolve(), args.prepare_only)


if __name__ == "__main__":
    main()
