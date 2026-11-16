#!/usr/bin/env python3
"""Select between cached captions using visual verification and FFN promotion."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_jsonl, load_jsonl
from scripts.evaluate_generation_time_evidence_drift_llava import load_variant

OUTPUT_ROOT = ROOT / "outputs/generation_time_evidence_drift_llava"


def caption_risk(caption, ffn_weight, repeat_weight):
    first_mentions = {}
    counts = Counter()
    for word in caption["objects"]:
        first_mentions.setdefault(word["lemma"], word)
        counts[word["lemma"]] += 1
    risks = []
    for word in first_mentions.values():
        margin = float(word["presence_verification"]["image_yes_minus_no"])
        support = float(word["image_support"])
        promotion = float(word["image_positive_ffn_gain"])
        if not all(math.isfinite(value) for value in (margin, support, promotion)):
            raise ValueError(f"Non-finite verification inputs for {word['lemma']}")
        verification_risk = np.logaddexp(0.0, -margin / 2.0)
        ground_deficit = math.exp(-float(np.logaddexp(0.0, support)))
        ffn_excess = math.tanh(promotion / 4.0)
        recurrence = 1 + repeat_weight * math.log(counts[word["lemma"]])
        risks.append(float(verification_risk * (1 + ffn_weight * ground_deficit * ffn_excess) * recurrence))
    return max(risks, default=None)


def unsupported_count(caption, *, require_joint_evidence):
    first_mentions = {}
    for word in caption["objects"]:
        first_mentions.setdefault(word["lemma"], word)
    if not first_mentions:
        return None
    count = 0
    for word in first_mentions.values():
        margin = float(word["presence_verification"]["image_yes_minus_no"])
        if margin >= 0:
            continue
        if require_joint_evidence and (word["image_support"] >= 0 or word["image_positive_ffn_gain"] <= 0):
            continue
        count += 1
    return count


def read_scores(folder, variants, ids):
    scores = {}
    expected = set(ids)
    pattern = f"grounding_scores.{'-'.join(variants)}.verify_nouns.shard*.jsonl"
    for path in sorted(folder.glob(pattern)):
        for row in load_jsonl(path):
            image_id = int(row["image_id"])
            if image_id not in expected:
                continue
            if image_id in scores:
                raise ValueError(f"Repeated verification for image {image_id}")
            if set(row["captions"]) != set(variants):
                raise ValueError(f"Incorrect caption variants for image {image_id}")
            scores[image_id] = row
    if scores.keys() != expected:
        raise ValueError(f"{len(expected - scores.keys())} images missing verified caption scores")
    return scores


def select(args):
    manifest = json.loads((args.output_root / "cohort.json").read_text(encoding="utf-8"))
    cohort_ids = list(map(int, manifest[args.cohort]))
    ids = cohort_ids[:args.limit]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected a nonempty image cohort with distinct IDs")
    folder = args.output_root / args.cohort
    variants = args.variants
    generations = {name: load_variant(folder, name, cohort_ids, ids) for name in variants}
    scores = read_scores(folder, variants, ids)
    base, alternative = variants
    selected = []
    for image_id in ids:
        record = scores[image_id]
        risk = {}
        for name in variants:
            scored = record["captions"][name]
            if scored["response_token_ids"] != generations[name][image_id]["response_token_ids"]:
                raise ValueError(f"Scored caption and generated caption differ for {name}, image {image_id}")
            if args.selection == "rank":
                risk[name] = caption_risk(scored, args.ffn_weight, args.repeat_weight)
            else:
                risk[name] = unsupported_count(scored, require_joint_evidence=args.selection == "joint-veto")
        chosen = base
        if risk[base] is not None and risk[alternative] is not None and risk[alternative] < risk[base]:
            chosen = alternative
        original = generations[chosen][image_id]
        seconds = sum(generations[name][image_id]["seconds"] for name in variants) + record["seconds"]
        prefill_seconds = sum(generations[name][image_id].get("prefill_seconds", 0) for name in variants)
        selected.append({
            "image_id": image_id,
            "generated_text": original["generated_text"],
            "response_token_ids": original["response_token_ids"],
            "seconds": seconds,
            "prefill_seconds": prefill_seconds,
            "verification_seconds": record["seconds"],
            "variant": args.name,
            "selected_from": chosen,
            "caption_risks": risk,
            "selection": args.selection,
            "ffn_weight": args.ffn_weight,
            "repeat_weight": args.repeat_weight,
        })
    output = folder / f"{args.name}.shard0.jsonl"
    atomic_write_jsonl(output, selected)
    switched = sum(row["selected_from"] == alternative for row in selected)
    print(f"{output} ({len(selected)} images, selected {alternative} on {switched})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--variants", nargs=2, default=["sadi", "air_decode"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--name", default="sadi_air_grounding_ffn")
    parser.add_argument("--ffn-weight", type=float, default=1.0)
    parser.add_argument("--repeat-weight", type=float, default=0.0)
    parser.add_argument("--selection", choices=("rank", "joint-veto", "vqa-veto"), default="rank")
    args = parser.parse_args()
    names = [args.cohort, args.name, *args.variants]
    if (not all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)
            or len(set([args.name, *args.variants])) != 3):
        parser.error("Expected separate valid names for the caption variants and output")
    if (not all(math.isfinite(w) and w >= 0 for w in (args.ffn_weight, args.repeat_weight))
            or (args.limit is not None and args.limit <= 0)):
        parser.error("Expected nonnegative finite weights and a positive image limit")
    select(args)


if __name__ == "__main__":
    main()
