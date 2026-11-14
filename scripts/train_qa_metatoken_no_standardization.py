#!/usr/bin/env python3
"""Train last-layer QA MetaToken without input-feature standardization."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
NAMESPACE = "metatoken_last_layer_no_standardization_prompt_last_token"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--config",
        default="configs/model_configs_qa_logs_k32_tau003_train_nodhcp.yaml",
    )
    parser.add_argument("--output-root", default="outputs/qa_benchmarks")
    parser.add_argument("--output", default="LOGS-K32-TAU003-NATIVE6-FULL")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    return parser.parse_args()


def _no_standardization_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(baseline_config(config))
    metatoken = dict(value.get("metatoken") or {})
    metatoken["standardize"] = False
    value["metatoken"] = metatoken
    return value


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
    split_records = split_qa_baseline_records(
        positioned, label_protocol=LABEL_PROTOCOL
    )
    with split_path.open(encoding="utf-8") as handle:
        split_manifest = json.load(handle)

    result_stem = f"{model}_{DATASET}_{LABEL_PROTOCOL}_qa_baselines"
    audit = {
        "feature_variant": "metatoken_last_decoder_layer_no_standardization",
        "attention_layer": "last_decoder_layer",
        "standardize": False,
        "unchanged_classifier_hyperparameters": True,
        "num_records": len(positioned),
    }
    _run_training_protocol(
        model=model,
        seeds=seeds,
        methods=("metatoken",),
        split_records=split_records,
        image_split_counts=split_manifest["image_counts"],
        feature_path=feature_path,
        split_path=split_path,
        result_root=baseline_dir,
        baseline_cfg=_no_standardization_config(config),
        device="cpu",
        run_name=None,
        result_stem=result_stem,
        label_protocol=(
            "qa_answer_correctness_all_0hall_1real_question_probe_split_"
            "physical_image_disjoint_metatoken_last_layer_no_standardization"
        ),
        write_summary=True,
        positive_class="real",
        sample_audit=audit,
        trainer_namespace=NAMESPACE,
    )
    print(
        f"[MetaTokenNoStandardization] complete model={model} "
        f"records={len(positioned)}"
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
