#!/usr/bin/env python3
"""Materialize aligned Yes-only SAM and object-position SVAR feature matrices."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.qa_benchmark import atomic_write_json, load_jsonl, sha256_file
from features.baseline.svar import svar_training_vector


ANSWER = "prompt_last_token"
OBJECT = "question_object_pre_token"
LABEL = "object_hallucination_yes_only_label"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    with (args.run_dir / "features.pkl").open("rb") as handle:
        method_rows = pickle.load(handle)
    baseline_path = args.run_dir / "baseline" / "object_hallucination_yes_only" / "features.pkl"
    with baseline_path.open("rb") as handle:
        baseline_rows = pickle.load(handle)
    method = {row["key"]: row for row in method_rows}
    baseline = {row["key"]: row for row in baseline_rows}
    keys = [row["key"] for row in questions]
    if len(set(keys)) != len(keys) or set(method) != set(keys) or set(baseline) != set(keys):
        raise ValueError("Question, SAM, and SVAR keys are not exactly aligned")

    sam_vectors, svar_vectors, labels, splits = [], [], [], []
    for question in questions:
        key = question["key"]
        row = method[key]
        base = baseline[key]
        if (row.get("question_object_position") or {}).get("status") != "extracted":
            raise ValueError(f"Object position not extracted: {key}")
        risk = np.asarray(row["positions"][OBJECT]["dgst"]["logs_risk"], dtype=np.float32)
        f_log = np.asarray(row["positions"][ANSWER]["dgst"]["f_log"], dtype=np.float32)
        if risk.ndim != 1 or f_log.shape != risk.shape:
            raise ValueError(f"SAM layer shape mismatch: {key}")
        svar = np.asarray(
            svar_training_vector(base["positions"][OBJECT]["baselines"]["svar"]),
            dtype=np.float32,
        )
        label = row[LABEL]
        if label not in (0, 1) or int(base["label"]) != label:
            raise ValueError(f"Yes-only label mismatch: {key}")
        if row["probe_split"] != question["probe_split"] or base["probe_split"] != question["probe_split"]:
            raise ValueError(f"Image split mismatch: {key}")
        sam_vectors.append(np.concatenate((risk, f_log)))
        svar_vectors.append(svar)
        labels.append(label)
        splits.append(question["probe_split"])

    sam = np.stack(sam_vectors).astype(np.float32, copy=False)
    svar = np.stack(svar_vectors).astype(np.float32, copy=False)
    if not np.isfinite(sam).all() or not np.isfinite(svar).all():
        raise ValueError("Non-finite SAM or SVAR features")
    output = args.run_dir / "sam_object_risk_answer_f_log_vs_svar_object_yesonly.npz"
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        keys=np.asarray(keys),
        sam=sam,
        svar=svar,
        labels=np.asarray(labels, dtype=np.int8),
        splits=np.asarray(splits),
    )
    os.replace(temporary, output)
    atomic_write_json(
        args.run_dir / "sam_svar_cross_position_summary.json",
        {
            "protocol": "yes_only_sam_object_risk_answer_f_log_vs_svar_object_v1",
            "sam_definition": "concat(logs_risk@question_object_pre_token,f_log@prompt_last_token)",
            "svar_definition": "controlled_svar@question_object_pre_token, layers [5,19)",
            "label_semantics": {"0": "hallucination (GT No, model Yes)", "1": "real (GT Yes, model Yes)"},
            "num_questions": len(keys),
            "counts_by_split": dict(Counter(splits)),
            "counts_by_label": {str(value): labels.count(value) for value in (0, 1)},
            "sam_shape": list(sam.shape),
            "svar_shape": list(svar.shape),
            "questions_sha256": sha256_file(args.questions),
            "method_features_sha256": sha256_file(args.run_dir / "features.pkl"),
            "svar_features_sha256": sha256_file(baseline_path),
            "npz_sha256": sha256_file(output),
        },
    )
    print(f"[export] {output}: SAM {sam.shape}, SVAR {svar.shape}")


if __name__ == "__main__":
    main()
