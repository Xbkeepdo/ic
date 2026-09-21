#!/usr/bin/env python3
"""Train one- and three-hidden-layer ICLR fusion probes with input z-scores."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_iclr_endac_attention_i_fusion import (
    FUSIONS, MODELS, RISK_KEYS, ROOT, SEEDS, _matrix, _probe_config, _summary,
    load_aligned,
)
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe
from utils.config_utils import load_config

OUT = ROOT / "outputs/iclr_512_endac_attention_i_fusion_standardized_no_bn"
ARCHITECTURES = {
    "one_hidden_128": (128,),
    "three_hidden_128_64_32": (128, 64, 32),
}


def train(model: str, device: str, out: Path) -> None:
    torch.set_num_threads(1)
    audit_source = ROOT / "outputs/iclr_512_endac_attention_i_fusion" / model / "alignment.json"
    expected = json.loads(audit_source.read_text())
    data, audit = load_aligned(model, expected["risk_run_name"], True)
    if audit != expected:
        raise ValueError(f"{model}: aligned cohort changed since the original fusion run")
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    output_device = torch.device(device)
    for arch, hidden_sizes in ARCHITECTURES.items():
        model_out = out / arch / model
        model_out.mkdir(parents=True, exist_ok=True)
        audit_path = model_out / "alignment.json"
        if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
            raise ValueError(f"{model}: existing alignment differs at {audit_path}")
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        for risk in RISK_KEYS:
            for fusion in FUSIONS:
                features = _matrix(data, risk, fusion)
                for seed in SEEDS:
                    case_out = model_out / risk / fusion / f"seed{seed}"
                    metrics_path = case_out / "metrics.json"
                    if metrics_path.exists():
                        print("SKIP", model, arch, risk, fusion, seed, flush=True)
                        continue
                    cfg = replace(
                        _probe_config(config, seed),
                        hidden_sizes=hidden_sizes,
                        batch_norm=False,
                        standardize_input=True,
                    )
                    metrics = train_and_evaluate_probe(
                        X_train=features[data["train_mask"]],
                        y_train=data["label"][data["train_mask"]],
                        X_val=np.empty((0, features.shape[1]), dtype=np.float32),
                        y_val=np.empty((0,), dtype=np.int32),
                        X_test=features[data["test_mask"]],
                        y_test=data["label"][data["test_mask"]],
                        config=cfg,
                        device=output_device,
                        output_dir=str(case_out),
                    )
                    metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
                    print("DONE", model, arch, risk, fusion, seed, flush=True)
        _summary(model, out / arch, audit)
        summary_path = model_out / "summary.md"
        contents = summary_path.read_text()
        contents = contents.replace(
            "\n\nExact matched mentions:",
            f"\n\nArchitecture: {arch} = {list(hidden_sizes)} hidden units; no BatchNorm. "
            "Each input column is z-scored using training-only mean and population standard deviation; "
            "constant columns use scale 1. Each seed saves input_scaler.npz.\n\nExact matched mentions:",
            1,
        )
        summary_path.write_text(contents)
        print("SUMMARY", summary_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    train(args.model, args.device, args.output_root.resolve())


if __name__ == "__main__":
    main()
