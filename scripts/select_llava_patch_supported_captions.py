#!/usr/bin/env python3
"""Compare actual patch evidence and later FFN preference on cached captions."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_jsonl, load_jsonl
from scripts.evaluate_generation_time_evidence_drift_llava import load_variant

OUTPUT_ROOT = ROOT / "outputs/generation_time_evidence_drift_llava"
POLICIES = ("vqa", "write", "drift", "joint")


def load_scores(folder: Path, variants: list[str], ids: list[int]) -> dict:
    expected = set(ids)
    rows = {}
    protocol = None
    pattern = f"grounding_scores.{'-'.join(variants)}.verify_nouns.patch_drift.shard*.jsonl"
    for path in sorted(folder.glob(pattern)):
        for row in load_jsonl(path):
            image_id = int(row["image_id"])
            if image_id not in expected:
                continue
            if image_id in rows or set(row["captions"]) != set(variants):
                raise ValueError(f"Duplicate or incompatible patch scoring for image {image_id}")
            if protocol is None:
                protocol = row["protocol"]
            if row["protocol"] != protocol:
                raise ValueError(f"Mixed patch scoring protocols in {folder}")
            rows[image_id] = row
    if expected != rows.keys():
        raise ValueError(f"{len(expected - rows.keys())} images missing patch scoring in {folder}")
    return rows


def fit_scale(records: dict, variants: list[str]) -> tuple[float, float]:
    values = [
        word["patch_drift"]["write_semantic_support"]
        for row in records.values()
        for name in variants
        for word in row["captions"][name]["objects"]
        if word["patch_drift"]["write_semantic_support"] is not None
    ]
    if not values or not np.isfinite(values).all():
        raise ValueError("Expected finite write-semantic scores on the calibration cohort")
    lower, center, upper = np.quantile(values, (0.25, 0.50, 0.75))
    return float(center), max(float(upper - lower), 1e-6)


def summarize_caption(caption: dict, center: float, scale: float) -> tuple[dict, float]:
    scores = {policy: [] for policy in POLICIES}
    coverage = 0.0
    for word in caption["objects"]:
        yes_no = float(word["presence_verification"]["image_yes_minus_no"])
        evidence = word["patch_drift"]
        support = evidence["write_semantic_support"]
        distance = evidence["ffn_hellinger2"]
        if not math.isfinite(yes_no):
            raise ValueError(f"Invalid VQA margin for {word['lemma']!r}")
        base = float(np.logaddexp(0.0, -yes_no / 2.0))
        coverage += math.exp(-base)
        weak, drift = 0.0, 0.0
        if support is not None and distance is not None:
            if not math.isfinite(support) or not 0 <= distance <= 1:
                raise ValueError(f"Invalid direct patch evidence for {word['lemma']!r}")
            weak = math.exp(-float(np.logaddexp(0.0, (support - center) / scale)))
            drift = float(distance)
        scores["vqa"].append(base)
        scores["write"].append(base * (1 + weak))
        scores["drift"].append(base * (1 + drift))
        scores["joint"].append(base * (1 + weak * drift))
    return {policy: max(values, default=None) for policy, values in scores.items()}, coverage


def run(args) -> None:
    manifest = json.loads((args.output_root / "cohort.json").read_text(encoding="utf-8"))
    ids = list(map(int, manifest[args.cohort]))
    calib_ids = list(map(int, manifest[args.calibration_cohort]))
    selected = ids[:args.limit]
    calib_ids = calib_ids[:args.calibration_limit]
    if not selected or not calib_ids or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty, distinct evaluation and calibration cohorts")
    folder = args.output_root / args.cohort
    records = load_scores(folder, args.scored_variants, selected)
    calibration = load_scores(args.output_root / args.calibration_cohort, args.scored_variants, calib_ids)
    if next(iter(records.values()))["protocol"] != next(iter(calibration.values()))["protocol"]:
        raise ValueError("Calibration and evaluation patch scoring setups differ")
    center, scale = fit_scale(calibration, args.candidates)
    generated = {name: load_variant(folder, name, ids, selected) for name in args.candidates}
    base = args.candidates[0]
    outputs = {policy: [] for policy in POLICIES}
    for image_id in selected:
        row = records[image_id]
        risk, coverage = {}, {}
        for name in args.candidates:
            caption = row["captions"][name]
            if caption["response_token_ids"] != generated[name][image_id]["response_token_ids"]:
                raise ValueError(f"Cached patch features and caption differ for {name}, image {image_id}")
            risk[name], coverage[name] = summarize_caption(caption, center, scale)
        candidates = [
            name for name in args.candidates
            if (not args.preserve_coverage or coverage[name] >= coverage[base])
            and risk[name]["vqa"] is not None
        ]
        for policy in POLICIES:
            chosen = base
            if risk[base][policy] is not None:
                chosen = min(candidates, key=lambda name: risk[name][policy])
            chosen_row = generated[chosen][image_id]
            outputs[policy].append({
                "image_id": image_id,
                "variant": f"{args.name}_{policy}",
                "selected_from": chosen,
                "generated_text": chosen_row["generated_text"],
                "response_token_ids": chosen_row["response_token_ids"],
                "seconds": sum(generated[name][image_id]["seconds"] for name in args.candidates) + row["seconds"],
                "coverage_proxy": coverage,
                "preserve_coverage": args.preserve_coverage,
                "caption_risks": {name: value[policy] for name, value in risk.items()},
                "support_center": center,
                "support_scale": scale,
            })
    for policy, rows in outputs.items():
        path = folder / f"{args.name}_{policy}.shard0.jsonl"
        atomic_write_jsonl(path, rows)
        print(f"{path.name}: {sum(row['selected_from'] != base for row in rows)}/{len(rows)} switches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--calibration-cohort", default="pilot")
    parser.add_argument("--calibration-limit", type=int, default=100)
    parser.add_argument("--scored-variants", nargs="+", default=["svar", "sadi", "air_decode"])
    parser.add_argument("--candidates", nargs="+", required=True, help="First variant is the default caption")
    parser.add_argument("--name", required=True)
    parser.add_argument("--preserve-coverage", action="store_true")
    args = parser.parse_args()
    names = [args.cohort, args.calibration_cohort, args.name, *args.scored_variants, *args.candidates]
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        parser.error("Invalid cohort or variant name")
    if (len(set(args.candidates)) != len(args.candidates) or len(args.candidates) < 2
            or not set(args.candidates).issubset(args.scored_variants)
            or args.limit <= 0 or args.calibration_limit <= 0):
        parser.error("Expected at least two distinct scored candidates and positive limits")
    run(args)


if __name__ == "__main__":
    main()
