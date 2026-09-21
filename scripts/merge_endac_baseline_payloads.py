#!/usr/bin/env python3
"""Merge reused ENDAC baseline payloads into the V2 extraction artifact."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.baseline import validate_baseline_record
from utils.io_utils import load_pkl, save_json, save_pkl


REQUIRED = (
    "metatoken",
    "svar",
    "dhcp",
    "projectaway",
    "token_grounding",
    "instruct_lens",
)

SOURCES = {
    "qwen2_5_vl_7b": (
        "outputs/qwen2_5_vl_7b/COCO4000-512-ICLR-RISK/baseline/features.pkl",
        ("metatoken", "svar"),
    ),
    "qwen3_vl_8b": (
        "outputs/qwen3_vl_8b/COCO4000-512-ENDAC/baseline/features.pkl",
        ("svar", "projectaway"),
    ),
    "internvl_2_5_8b": (
        "outputs/internvl_2_5_8b/COCO4000-512-ICLR-RISK/baseline/features.pkl",
        ("metatoken", "svar"),
    ),
}

EXTRA_SOURCES = {
    # The earlier ENDAC artifact has a legacy MetaToken probability-difference
    # adaptation.  This exact-key ICLR artifact contains the current original-
    # paper Eq.1--12 / Eq.11 payload for the same generation and labels.
    "qwen3_vl_8b": (
        (
            "outputs/qwen3_vl_8b/COCO4000-512-ICLR-BASELINES/baseline/features.pkl",
            ("metatoken",),
        ),
    ),
}


def _key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    target = record.get("target_token_id")
    if target is None:
        raise ValueError("ENDAC baseline record is missing target_token_id")
    return (
        int(record["image_id"]),
        int(record["response_token_idx"]),
        int(target),
    )


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        return array.dtype.kind not in "biufc" or bool(np.isfinite(array).all())
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted((*SOURCES, "llava_1_5_7b")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--instruct-lens-correction",
        default=None,
        help=(
            "Optional InsLen-only features.pkl whose exact-key payload replaces "
            "the preliminary InsLen payload in the merged target."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_path = Path(args.output_dir) / "baseline" / "features.pkl"
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    rows = load_pkl(str(target_path))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a list in {target_path}")

    reused: tuple[str, ...] = ()
    source_path: Path | None = None
    extra_source_audit: list[dict[str, Any]] = []
    if args.model in SOURCES:
        raw_source, reused = SOURCES[args.model]
        source_path = Path(raw_source)
        source_rows = load_pkl(str(source_path))
        source_by_key = {_key(record): record for record in source_rows}
        if len(source_by_key) != len(source_rows):
            raise RuntimeError(f"Duplicate source ENDAC key in {source_path}")
        target_keys = {_key(record) for record in rows}
        if target_keys != set(source_by_key):
            missing = sorted(set(source_by_key) - target_keys)[:5]
            extra = sorted(target_keys - set(source_by_key))[:5]
            raise RuntimeError(
                "Extracted/reused ENDAC key sets differ: "
                f"target={len(target_keys)}, source={len(source_by_key)}, "
                f"missing={missing}, extra={extra}"
            )
        for record in rows:
            source = source_by_key[_key(record)]
            if int(record["label"]) != int(source["label"]):
                raise RuntimeError(f"Label mismatch for key {_key(record)}")
            source_payloads = source.get("baselines") or {}
            for method in reused:
                if method not in source_payloads:
                    raise KeyError(f"Source key {_key(record)} lacks {method}")
                payload = deepcopy(source_payloads[method])
                # These reusable artifacts predate the explicit protocol
                # metadata, but their exact object-token keys and labels are
                # the controlled ENDAC records audited above.  Preserve the
                # values and add the metadata required by the current trainer.
                if method == "svar" and not str(payload.get("protocol") or ""):
                    payload["protocol"] = "controlled"
                record.setdefault("baselines", {})[method] = payload
            if "svar" in reused:
                metadata = record.setdefault("metadata", {})
                if not isinstance(metadata, dict):
                    raise TypeError(f"Record metadata is not mutable for key {_key(record)}")
                metadata["svar_protocol"] = "controlled"

    for raw_extra_path, extra_methods in EXTRA_SOURCES.get(args.model, ()):
        extra_path = Path(raw_extra_path)
        extra_rows = load_pkl(str(extra_path))
        extra_by_key = {_key(record): record for record in extra_rows}
        if len(extra_by_key) != len(extra_rows):
            raise RuntimeError(f"Duplicate extra-source ENDAC key in {extra_path}")
        target_keys = {_key(record) for record in rows}
        if target_keys != set(extra_by_key):
            missing = sorted(set(extra_by_key) - target_keys)[:5]
            extra = sorted(target_keys - set(extra_by_key))[:5]
            raise RuntimeError(
                "Extracted/extra-source ENDAC key sets differ: "
                f"target={len(target_keys)}, source={len(extra_by_key)}, "
                f"missing={missing}, extra={extra}"
            )
        for record in rows:
            source = extra_by_key[_key(record)]
            if int(record["label"]) != int(source["label"]):
                raise RuntimeError(f"Extra-source label mismatch for key {_key(record)}")
            source_payloads = source.get("baselines") or {}
            for method in extra_methods:
                if method not in source_payloads:
                    raise KeyError(f"Extra source key {_key(record)} lacks {method}")
                record.setdefault("baselines", {})[method] = deepcopy(
                    source_payloads[method]
                )
        extra_source_audit.append({
            "path": str(extra_path),
            "methods": list(extra_methods),
        })

    correction_path: Path | None = None
    corrected_object_state: str | None = None
    if args.instruct_lens_correction is not None:
        correction_path = Path(args.instruct_lens_correction)
        correction_rows = load_pkl(str(correction_path))
        correction_by_key = {_key(record): record for record in correction_rows}
        if len(correction_by_key) != len(correction_rows):
            raise RuntimeError(f"Duplicate correction ENDAC key in {correction_path}")
        target_keys = {_key(record) for record in rows}
        if target_keys != set(correction_by_key):
            missing = sorted(target_keys - set(correction_by_key))[:5]
            extra = sorted(set(correction_by_key) - target_keys)[:5]
            raise RuntimeError(
                "InsLen correction/target ENDAC key sets differ: "
                f"target={len(target_keys)}, correction={len(correction_by_key)}, "
                f"missing={missing}, extra={extra}"
            )
        state_definitions: set[str] = set()
        for record in rows:
            correction = correction_by_key[_key(record)]
            if int(record["label"]) != int(correction["label"]):
                raise RuntimeError(f"InsLen correction label mismatch for key {_key(record)}")
            payload = (correction.get("baselines") or {}).get("instruct_lens")
            if payload is None:
                raise KeyError(f"Correction key {_key(record)} lacks instruct_lens")
            state_definitions.add(str(payload.get("object_state_definition")))
            record.setdefault("baselines", {})["instruct_lens"] = deepcopy(payload)
        if state_definitions != {"final_decoder_norm"}:
            raise RuntimeError(
                "InsLen correction must use final_decoder_norm object states, got "
                f"{sorted(state_definitions)}"
            )
        corrected_object_state = "final_decoder_norm"

    keys = []
    observed_object_states: set[str] = set()
    labels = {0: 0, 1: 0}
    for record in rows:
        validate_baseline_record(record, required=REQUIRED)
        if not _finite(record["baselines"]):
            raise ValueError(f"Non-finite baseline payload for key {_key(record)}")
        keys.append(_key(record))
        labels[int(record["label"])] += 1
        observed_object_states.add(str(
            record["baselines"]["instruct_lens"].get(
                "object_state_definition", "raw_post_block"
            )
        ))
    if len(set(keys)) != len(keys):
        raise RuntimeError("Merged output contains duplicate ENDAC keys")
    if corrected_object_state is None and len(observed_object_states) == 1:
        corrected_object_state = next(iter(observed_object_states))
    rows.sort(key=_key)
    save_pkl(rows, str(target_path))
    audit = {
        "model": args.model,
        "target": str(target_path),
        "source": None if source_path is None else str(source_path),
        "reused_methods": list(reused),
        "extra_sources": extra_source_audit,
        "reused_svar_protocol_metadata_added": "svar" in reused,
        "instruct_lens_correction": (
            None if correction_path is None else str(correction_path)
        ),
        "instruct_lens_object_state": corrected_object_state,
        "required_methods": list(REQUIRED),
        "records": len(rows),
        "unique_keys": len(keys),
        "labels": {"hallucination": labels[0], "real": labels[1]},
        "status": "complete",
    }
    save_json(audit, str(Path(args.output_dir) / "baseline" / "merge_audit.json"))
    print(audit)


if __name__ == "__main__":
    main()
