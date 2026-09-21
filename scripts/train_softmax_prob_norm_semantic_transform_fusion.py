#!/usr/bin/env python3
"""Fuse the fixed softmax-probability weighted-cost risk with four norm semantic Fs."""

from __future__ import annotations

import argparse
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
    target_key,
)
from scripts.train_iclr_endac_attention_i_fusion import MODELS, _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe  # noqa: E402
from scripts.train_weighted_cost_attention_i_fusion import SEEDS, _risk_specs  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


FEATURE_ROOT = ROOT / "outputs/norm_semantic_transform_curves_by_label"
OUT = ROOT / "outputs/fixed_wcost_softmax_prob_norm_semantic_transform_fusion"
GF_PROBABILITY_ROOT = GF_ROOT / "outputs/coco4000_512_endac_semantic_attention_topk"
F_NAMES = (
    "sqrt_residual_fusion",
    "log_residual_fusion",
    "mass_preserving_fusion",
    "asinh_sqrt_fusion",
)
GROUPS = tuple(f"risk_{name}" for name in F_NAMES)
EXPECTED = len(GROUPS) * len(SEEDS)
RISK_NAME = "wcost_all__softmax__hpre_softmax_prob_gauss__w02"
RISK_FIELD = (
    "dgst_t_hpre_softmax_prob_gauss_risk_linear_matched_state_target_"
    "w02_hpre_source_softmax_per_layer"
)


def fixed_risk_spec() -> dict:
    matches = [spec for spec in _risk_specs() if spec["name"] == RISK_NAME]
    if len(matches) != 1 or matches[0]["field"] != RISK_FIELD:
        raise AssertionError(f"Expected exact fixed risk spec, got {matches}")
    return matches[0]


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
    if not (np.array_equal(labels, compact["label"]) and np.array_equal(image_ids, compact["image_id"])):
        raise ValueError(f"{model}: compact feature order differs from semantic source")
    if any(int(left["label"]) != int(right["label"]) for left, right in zip(semantic_rows, aligned)):
        raise ValueError(f"{model}: aligned risk label mismatch")
    features = {name: np.asarray(compact[name], dtype=np.float32) for name in F_NAMES}
    if (any(risk.shape != value.shape for value in features.values())
            or not np.isfinite(risk).all()
            or any(not np.isfinite(value).all() for value in features.values())):
        raise ValueError(f"{model}: incompatible or non-finite feature arrays")
    data = {
        "risk": risk,
        **features,
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


def train(model: str, device_name: str, output_root: Path) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    data, audit = prepare(model, model_out)
    matrices = {
        f"risk_{name}": np.concatenate((data["risk"], data[name]), axis=1)
        for name in F_NAMES
    }
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    example = replace(_probe_config(root_config, SEEDS[0]), hidden_sizes=(128,),
                      batch_norm=False, standardize_input=True)
    atomic_json(model_out / "protocol.json", {
        **audit,
        "probe": _json_ready(asdict(example)),
        "seeds": list(SEEDS),
        "feature_groups": {name: list(value.shape) for name, value in matrices.items()},
        "risk_only": "reused existing weighted_cost_probe metrics",
        "standalone_f": "reused fixed_wcost_norm_semantic_transform_fusion metrics",
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
