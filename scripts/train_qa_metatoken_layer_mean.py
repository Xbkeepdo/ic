#!/usr/bin/env python3
"""Train QA MetaToken after averaging visual-attention heads over all layers.

The formal QA artifacts retain per-layer/per-head visual attention mass in the
SVAR payload.  For a sample with P visual tokens, MetaToken's attention feature
for layer l and head h is SVAR[l, h] / P.  This script replaces only those H
entries in the existing MetaToken vector with their mean over decoder layers;
the other ten published MetaToken features remain unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.baseline import baseline_config  # noqa: E402
from features.qa_baseline import (  # noqa: E402
    baseline_record_for_position,
    split_qa_baseline_records,
)
from scripts.train_baselines import _run_training_protocol  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.qa_paths import resolve_qa_paths  # noqa: E402


DEFAULT_MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
DATASET = "clevr_exist_9k"
LABEL_PROTOCOL = "answer_correctness_all"
POSITION = "prompt_last_token"
NAMESPACE = "metatoken_layer_mean_prompt_last_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--config",
        default="configs/model_configs_qa_logs_k32_tau003_train_nodhcp.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/qa_benchmarks",
    )
    parser.add_argument("--output", default="LOGS-K32-TAU003-NATIVE6-FULL")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    return parser.parse_args()


def _visual_token_count(record: Mapping[str, Any]) -> int:
    metadata = record.get("metadata") or {}
    grid = metadata.get("visual_grid")
    if grid is None:
        dhcp = (record.get("baselines") or {}).get("dhcp") or {}
        grid = dhcp.get("source_grid")
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or len(grid) != 2:
        raise ValueError(f"Missing two-dimensional visual grid for {record.get('key')!r}")
    count = int(grid[0]) * int(grid[1])
    if count <= 0:
        raise ValueError(f"Invalid visual grid {grid!r} for {record.get('key')!r}")
    return count


def use_all_layer_mean_attention(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return records whose MetaToken head slots use mean_l mean_p A[l,h,p]."""

    converted: list[dict[str, Any]] = []
    layer_counts: set[int] = set()
    head_counts: set[int] = set()
    max_last_layer_error = 0.0
    for raw_record in records:
        record = deepcopy(dict(raw_record))
        baselines = record.get("baselines") or {}
        meta = deepcopy(dict(baselines.get("metatoken") or {}))
        svar = baselines.get("svar") or {}
        vector = np.asarray(meta.get("vector"), dtype=np.float32).reshape(-1).copy()
        ratios = np.asarray(
            svar.get("visual_attention_ratio"), dtype=np.float32
        )
        heads = int(meta.get("num_attention_heads", -1))
        if ratios.ndim != 2 or heads <= 0 or ratios.shape[1] != heads:
            raise ValueError(
                f"Invalid MetaToken/SVAR shapes for {record.get('key')!r}: "
                f"vector={vector.shape}, ratios={ratios.shape}, heads={heads}"
            )
        if vector.size != 10 + heads:
            raise ValueError(
                f"MetaToken vector for {record.get('key')!r} has "
                f"{vector.size} values; expected {10 + heads}"
            )

        patches = _visual_token_count(record)
        per_layer_head_means = ratios / float(patches)
        original_last_layer = vector[2 : 2 + heads]
        error = float(
            np.max(np.abs(original_last_layer - per_layer_head_means[-1]))
        )
        max_last_layer_error = max(max_last_layer_error, error)
        if not np.allclose(
            original_last_layer,
            per_layer_head_means[-1],
            rtol=2e-4,
            atol=2e-6,
        ):
            raise ValueError(
                "SVAR cannot reconstruct the saved last-layer MetaToken "
                f"attention for {record.get('key')!r}; max error={error:.3e}"
            )

        vector[2 : 2 + heads] = per_layer_head_means.mean(axis=0)
        meta["vector"] = vector
        meta["attention_layer_aggregation"] = "mean_over_all_decoder_layers"
        meta["attention_layer_count"] = int(ratios.shape[0])
        meta["attention_reconstruction_source"] = (
            "svar.visual_attention_ratio / num_visual_tokens"
        )
        baselines = dict(baselines)
        baselines["metatoken"] = meta
        record["baselines"] = baselines
        converted.append(record)
        layer_counts.add(int(ratios.shape[0]))
        head_counts.add(heads)

    audit = {
        "feature_variant": "metatoken_all_decoder_layers_mean",
        "source": "saved_svar_visual_attention_ratio_float32",
        "formula": "mean_l(sum_p(attention[l,h,p])/P)",
        "unchanged_metatoken_features": 10,
        "num_records": len(converted),
        "layer_counts": sorted(layer_counts),
        "head_counts": sorted(head_counts),
        "max_last_layer_reconstruction_abs_error": max_last_layer_error,
    }
    return converted, audit


def train_model(
    *,
    model: str,
    config: Mapping[str, Any],
    output_root: Path,
    output_name: str,
    seeds: Sequence[int],
) -> None:
    run_dir = resolve_qa_paths(
        str(output_root), model, output_name, DATASET
    ).benchmark_dir
    baseline_dir = run_dir / "baseline" / LABEL_PROTOCOL
    feature_path = baseline_dir / "features.pkl"
    split_path = baseline_dir / f"probe_splits.{POSITION}.json"
    with feature_path.open("rb") as handle:
        records = pickle.load(handle)
    positioned = [
        value
        for record in records
        if (value := baseline_record_for_position(record, POSITION)) is not None
    ]
    converted, audit = use_all_layer_mean_attention(positioned)
    split_records = split_qa_baseline_records(
        converted, label_protocol=LABEL_PROTOCOL
    )
    with split_path.open(encoding="utf-8") as handle:
        split_manifest = json.load(handle)

    result_stem = (
        f"{model}_{DATASET}_{LABEL_PROTOCOL}_qa_baselines"
    )
    _run_training_protocol(
        model=model,
        seeds=seeds,
        methods=("metatoken",),
        split_records=split_records,
        image_split_counts=split_manifest["image_counts"],
        feature_path=feature_path,
        split_path=split_path,
        result_root=baseline_dir,
        baseline_cfg=baseline_config(config),
        device="cpu",
        run_name=None,
        result_stem=result_stem,
        label_protocol=(
            "qa_answer_correctness_all_0hall_1real_question_probe_split_"
            "physical_image_disjoint_metatoken_all_decoder_layers_mean"
        ),
        write_summary=True,
        positive_class="real",
        sample_audit=audit,
        trainer_namespace=NAMESPACE,
    )
    print(
        f"[MetaTokenLayerMean] complete model={model} "
        f"records={len(converted)} audit={audit}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(args.output_root).resolve()
    for model in args.models:
        train_model(
            model=str(model),
            config=config,
            output_root=output_root,
            output_name=str(args.output),
            seeds=tuple(dict.fromkeys(int(seed) for seed in args.seeds)),
        )


if __name__ == "__main__":
    main()
