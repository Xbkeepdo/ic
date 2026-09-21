#!/usr/bin/env python3
"""Train sum_m log1p(softmax gate_m) log1p(||a_m||), alone and with risk."""

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
    SEEDS,
    _baseline_path,
    _risk_specs,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


OUT_BY_MODE = {
    "log": ROOT / "outputs/fixed_wcost_log_gate_log_am",
    "raw": ROOT / "outputs/fixed_wcost_raw_gate_am",
}
RUN = "COCO4000-512-WCOST-ALL"
GATE_FIELD = "dgst_t_hpre_softmax_prob_gauss_gate_per_layer"
RISK_NAME = "wcost_all__softmax__hpre_raw_logit_gauss__w02"
FEATURE_GROUPS_BY_MODE = {
    "log": ("log_gate_log_am_sum", "risk_log_gate_log_am_sum"),
    "raw": ("raw_gate_am_sum", "risk_raw_gate_am_sum"),
}
EXPECTED_CASES = 2 * len(SEEDS)


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


def target_key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def prepare_compact(
    model: str, model_out: Path, feature_mode: str = "log"
) -> tuple[dict[str, np.ndarray], dict]:
    cache = model_out / "compact_features.npz"
    audit_path = model_out / "feature_audit.json"
    if cache.exists() and audit_path.exists():
        saved = np.load(cache, allow_pickle=False)
        data = {key: np.asarray(saved[key]) for key in saved.files}
        audit = json.loads(audit_path.read_text())
        if audit.get("feature_mode", "log") != feature_mode:
            raise ValueError(f"{model}: cached feature mode differs from {feature_mode}")
        if list(data["feature"].shape) != audit["feature_shape"]:
            raise ValueError(f"{model}: cached feature shape differs from audit")
        if any(not np.isfinite(data[key]).all() for key in ("feature", "risk")):
            raise ValueError(f"{model}: cached feature is nonfinite")
        return data, audit

    spec = fixed_risk_spec()
    run = ROOT / "outputs" / model / RUN
    risk_path = run / "features.pkl"
    am_path = GF_ROOT / "outputs/coco4000_512_endac_811" / model / "am_norms/features.pkl"
    print(f"LOAD gate/risk: {risk_path}", flush=True)
    rows = load_pkl(risk_path)
    print(f"LOAD a_m norms: {am_path}", flush=True)
    am_rows = load_pkl(am_path)
    if not rows or GATE_FIELD not in rows[0] or spec["field"] not in rows[0]:
        raise KeyError(f"{model}: missing gate or risk field")
    am_by_id = {target_key(row): row for row in am_rows}
    row_ids = [target_key(row) for row in rows]
    if len(am_by_id) != len(rows) or set(am_by_id) != set(row_ids):
        raise ValueError(f"{model}: gate/risk and a_m target IDs differ")

    first_gate = np.asarray(rows[0][GATE_FIELD], dtype=np.float32)
    layers = int(first_gate.shape[0])
    feature = np.empty((len(rows), layers), dtype=np.float32)
    risk = np.empty((len(rows), layers), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int8)
    image_ids = np.empty(len(rows), dtype=np.int64)
    gate_min, gate_max = math.inf, -math.inf
    am_min, am_max = math.inf, -math.inf
    for index, row in enumerate(rows):
        key = target_key(row)
        am_row = am_by_id[key]
        gate = np.asarray(row[GATE_FIELD], dtype=np.float32)
        am_norm = np.asarray(am_row["am_norm"], dtype=np.float32)
        layer_risk = np.asarray(row[spec["field"]], dtype=np.float32)
        if gate.shape != am_norm.shape or gate.shape[0] != layers:
            raise ValueError(
                f"{model}/{key}: gate {gate.shape} differs from a_m {am_norm.shape}"
            )
        if layer_risk.shape != (layers,):
            raise ValueError(f"{model}/{key}: wrong risk shape {layer_risk.shape}")
        if not (np.isfinite(gate).all() and np.isfinite(am_norm).all()
                and np.isfinite(layer_risk).all()):
            raise ValueError(f"{model}/{key}: nonfinite source")
        if (gate < 0).any() or (am_norm < 0).any():
            raise ValueError(f"{model}/{key}: log1p source must be nonnegative")
        if int(row["label"]) != int(am_row["label"]):
            raise ValueError(f"{model}/{key}: label mismatch")
        if int(row["image_id"]) != int(am_row["image_id"]):
            raise ValueError(f"{model}/{key}: image mismatch")
        gate64 = gate.astype(np.float64)
        am64 = am_norm.astype(np.float64)
        if feature_mode == "log":
            product = np.log1p(gate64) * np.log1p(am64)
        elif feature_mode == "raw":
            product = gate64 * am64
        else:
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        feature[index] = np.sum(product, axis=1).astype(np.float32)
        risk[index] = layer_risk
        labels[index] = int(row["label"])
        image_ids[index] = int(row["image_id"])
        gate_min, gate_max = min(gate_min, float(gate.min())), max(gate_max, float(gate.max()))
        am_min, am_max = min(am_min, float(am_norm.min())), max(am_max, float(am_norm.max()))

    if not np.isfinite(feature).all() or (feature < 0).any():
        raise ValueError(f"{model}: invalid constructed feature")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{model}: expected both labels")
    split = json.loads((run / "image_splits.json").read_text())
    train_ids = {int(value) for value in split["train"]}
    test_ids = {int(value) for value in split["test"]}
    if train_ids & test_ids or len(train_ids) != 3200 or len(test_ids) != 800:
        raise ValueError(f"{model}: expected strict 3200/800 image split")
    train_mask = np.isin(image_ids, list(train_ids))
    test_mask = np.isin(image_ids, list(test_ids))
    if not np.all(train_mask | test_mask):
        raise ValueError(f"{model}: rows outside split")
    data = {
        "feature": feature,
        "risk": risk,
        "label": labels,
        "image_id": image_ids,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }
    audit = {
        "schema": f"{feature_mode}-gate-am-sum-v1",
        "model": model,
        "feature_mode": feature_mode,
        "definition": (
            "sum_visual_token log1p(hpre_softmax_prob_gauss_gate) * log1p(||a_m||_2), per layer"
            if feature_mode == "log"
            else "sum_visual_token hpre_softmax_prob_gauss_gate * ||a_m||_2, per layer"
        ),
        "gate_field": GATE_FIELD,
        "risk": spec,
        "risk_source": str(risk_path),
        "am_norm_source": str(am_path),
        "matched_rows": len(rows),
        "matched_train_rows": int(train_mask.sum()),
        "matched_test_rows": int(test_mask.sum()),
        "feature_shape": list(feature.shape),
        "feature_min": float(feature.min()),
        "feature_max": float(feature.max()),
        "feature_mean": float(feature.mean()),
        "gate_source_range": [gate_min, gate_max],
        "am_norm_source_range": [am_min, am_max],
    }
    atomic_npz(cache, **data)
    atomic_json(audit_path, audit)
    del rows, am_rows, am_by_id
    gc.collect()
    return data, audit


def report(metrics: dict, threshold: str) -> dict:
    return metrics["threshold_reports"][threshold]["test_metrics"]


def summarize(model: str, model_out: Path, feature_mode: str = "log") -> None:
    spec = fixed_risk_spec()
    sources = {
        "risk": [_baseline_path(model, spec, seed) for seed in SEEDS],
        **{
            name: [model_out / name / f"seed{seed}" / "metrics.json" for seed in SEEDS]
            for name in FEATURE_GROUPS_BY_MODE[feature_mode]
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
    baselines = {row["threshold"]: row for row in rows if row["feature_group"] == "risk"}
    for row in rows:
        baseline = baselines[row["threshold"]]
        row["delta_real_auc_vs_risk"] = row["real_auc_mean"] - baseline["real_auc_mean"]
        row["delta_hall_aupr_vs_risk"] = row["hall_aupr_mean"] - baseline["hall_aupr_mean"]
    with (model_out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: {feature_mode}-gate × write-norm feature", "",
        (
            "New feature: sum over visual tokens of log1p(softmax-prob Gaussian gate) × log1p(||a_m||), per layer."
            if feature_mode == "log"
            else "New feature: sum over visual tokens of softmax-prob Gaussian gate × ||a_m||, per layer."
        ),
        "Fixed risk: WCOST-ALL / softmax / hpre_raw_logit_gauss / w=0.2. Probe: one hidden layer 128, training-only z-score, dropout 0.3, no BatchNorm, seeds 43–45.",
        "", "| Threshold | Feature group | Real F1 | Real AUC | Δ AUC vs risk | Hall AUPR | Δ Hall AUPR vs risk |",
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


def train(
    model: str, device: str, output_root: Path, prepare_only: bool,
    feature_mode: str = "log",
) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    data, audit = prepare_compact(model, model_out, feature_mode)
    if prepare_only:
        print(json.dumps(audit, indent=2), flush=True)
        return
    standalone_name, fused_name = FEATURE_GROUPS_BY_MODE[feature_mode]
    matrices = {
        standalone_name: data["feature"].astype(np.float32, copy=False),
        fused_name: np.concatenate(
            (data["risk"], data["feature"]), axis=1
        ).astype(np.float32, copy=False),
    }
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    example = replace(
        _probe_config(root_config, SEEDS[0]),
        hidden_sizes=(128,), batch_norm=False, standardize_input=True,
    )
    atomic_json(model_out / "protocol.json", {
        **audit,
        "probe": _json_ready(asdict(example)),
        "seeds": list(SEEDS),
        "feature_groups": {name: list(value.shape) for name, value in matrices.items()},
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
                X_train=matrix[data["train_mask"]], y_train=labels[data["train_mask"]],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=matrix[data["test_mask"]], y_test=labels[data["test_mask"]],
                config=config, device=torch.device(device), output_dir=str(case),
            )
            metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
            completed += 1
            atomic_json(model_out / "progress.json", {
                "model": model, "stage": "training", "completed": completed,
                "total": EXPECTED_CASES, "feature_group": name, "seed": seed,
            })
            print("DONE", model, name, seed, flush=True)
    summarize(model, model_out, feature_mode)
    atomic_json(model_out / "progress.json", {
        "model": model, "stage": "complete", "completed": EXPECTED_CASES, "total": EXPECTED_CASES,
    })
    print("SUMMARY", model_out / "summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-mode", choices=tuple(OUT_BY_MODE), default="log")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root or OUT_BY_MODE[args.feature_mode]
    train(
        args.model, args.device, output_root.resolve(), args.prepare_only,
        args.feature_mode,
    )


if __name__ == "__main__":
    main()
