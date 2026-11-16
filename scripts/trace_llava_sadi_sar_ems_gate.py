#!/usr/bin/env python3
"""Replay cached SADI captions and record causal SAR/EMS at CHAIR object tokens."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.llava_decoding_evidence import (  # noqa: E402
    CANDIDATE_COUNT, EVIDENCE_LAYERS, VARIANTS, WriteEvidence, lexical_nouns,
)
from models import build_model  # noqa: E402
from scripts.evaluate_generation_time_evidence_drift_llava import load_chair, load_variant  # noqa: E402
from scripts.generation_time_evidence_drift_llava import (  # noqa: E402
    COCO_ROOT, CONFIG, MODEL, MODEL_ROOT, OUT, _decode, _prefill, _read_rows, _response,
)
from scripts.generation_time_sar_ems_seed43 import _append_jsonl, _inputs, _read_json  # noqa: E402
from utils.config_utils import get_extraction_model_cfg, load_config  # noqa: E402
from utils.token_alignment import build_response_token_offsets, token_indices_for_char_span  # noqa: E402


def align_mentions(chair, tokenizer, image_id, caption, response_ids):
    offsets = build_response_token_offsets(tokenizer, response_ids, caption)
    by_step = {}
    unmatched = []
    for mention in chair.compute_chair_token(image_id, caption)["object_mentions"]:
        indices = token_indices_for_char_span(offsets, mention["char_start"], mention["char_end"])
        label = {
            "canonical": mention["canonical_name"],
            "surface": mention["surface"],
            "label": mention["label"],
            "char_start": mention["char_start"],
            "char_end": mention["char_end"],
            "token_indices": indices,
        }
        if indices:
            by_step.setdefault(indices[0], []).append(label)
        else:
            unmatched.append(label)
    return by_step, unmatched


def trace_caption(wrapper, decoder, chair, image_file, row):
    image_id = int(row["image_id"])
    response_ids = row["response_token_ids"]
    caption = row["generated_text"]
    if not response_ids or _response(wrapper, response_ids) != caption:
        raise ValueError(f"Image {image_id}: saved SADI token IDs do not decode to its caption")
    by_step, unmatched = align_mentions(chair, wrapper.tokenizer, image_id, caption, response_ids)
    with Image.open(image_file) as image:
        inputs = _inputs(wrapper, image.convert("RGB"), MODEL)
    span = wrapper._find_visual_token_range(inputs, wrapper._image_token_id())
    if span[1] >= inputs["input_ids"].shape[1] or not inputs["attention_mask"].all().item():
        raise ValueError(f"Image {image_id}: expected an unpadded prompt ending in text")
    decoder.reset_image(span)

    # Capture the same tensors as the online reranker. Replay only the checked SADI winner.
    capture_variant = replace(VARIANTS["sadi"], sar_ems_selection=True)
    with decoder.capture(prefill=True, variant=VARIANTS["greedy"], need_prefill=True,
                         need_sar_ems_selection=True):
        prefix = _prefill(wrapper, inputs)
    records = []

    def select(logits, step, _response_so_far):
        scores = logits[0].float()
        winner = int(scores.argmax().item())
        if step >= len(response_ids) or winner != response_ids[step]:
            expected = response_ids[step] if step < len(response_ids) else None
            raise RuntimeError(f"Image {image_id}, step {step}: saved SADI token {expected}, replay {winner}")
        if step not in by_step:
            return winner, None

        top_logits, top_tokens = scores.topk(CANDIDATE_COUNT)
        alternatives = [(token, float(gap)) for token, gap in zip(
            top_tokens.tolist(), (scores[winner] - top_logits).tolist(), strict=True,
        ) if token != winner]
        relative_sar, f_log, evidence = decoder._candidate_sar(top_tokens.new_tensor([winner]))
        risk = decoder._candidate_risks(evidence, [0])[0]
        records.append({
            "step": step,
            "target_token_id": winner,
            "winner_logit": float(scores[winner]),
            "runner_up_logit_gap": alternatives[0][1],
            "winner_in_noun_gate": winner in decoder.nouns,
            "alternatives": [{"token_id": token, "logit_gap": gap, "in_noun_gate": token in decoder.nouns}
                             for token, gap in alternatives],
            "F_log": f_log[:, 0].tolist(),
            "relative_SAR": relative_sar[:, 0].tolist(),
            "risk": risk,
            "mentions": by_step[step],
        })
        return winner, None

    with decoder.capture(prefill=False, variant=capture_variant):
        replay, _ = _decode(wrapper, inputs["input_ids"][:, -1:], prefix, select)
    if replay != response_ids or len(records) != len(by_step):
        raise RuntimeError(f"Image {image_id}: incomplete SADI replay or object trace")
    return records, unmatched


@torch.inference_mode()
def run(args):
    manifest = _read_json(args.output_root / "cohort.json")
    cohorts = {"pilot": manifest["pilot"][:100], "confirmation": manifest["confirmation"][:40]}
    if len(cohorts["pilot"]) != 100 or len(cohorts["confirmation"]) != 40:
        raise ValueError("Expected 100 pilot and 40 fixed SADI caption IDs")
    selections = [(cohort, int(image_id)) for cohort, ids in cohorts.items() for image_id in ids]
    if len(dict((image_id, cohort) for cohort, image_id in selections)) != len(selections):
        raise ValueError("The calibration images must be distinct")
    selections = selections[:args.limit][args.shard::args.shards]
    destination = args.trace_dir or args.output_root / "sadi_sar_ems_trace"
    path = destination / f"sadi_sar_ems_trace.shard{args.shard}.jsonl"
    existing = _read_rows(path)
    for image_id, record in existing.items():
        if (record["cohort"], image_id) not in selections or tuple(record["layers"]) != EVIDENCE_LAYERS:
            raise ValueError(f"Incompatible existing trace in {path}: image {image_id}")
    remaining = [(cohort, image_id) for cohort, image_id in selections if image_id not in existing]
    if not remaining:
        return

    generations = {
        cohort: load_variant(args.output_root / cohort, "sadi", manifest[cohort], ids)
        for cohort, ids in cohorts.items()
    }
    ground_truth_ids = [int(image_id) for cohort in cohorts for image_id in manifest[cohort]]
    chair, _ = load_chair(args.coco_root, args.output_root, ground_truth_ids, args.chair_cache)
    torch.set_num_threads(1)
    config = load_config(str(CONFIG))
    model_config = dict(get_extraction_model_cfg(config, MODEL))
    model_config["hf_name"] = str(args.model_root)
    wrapper = build_model(MODEL, model_config, device=args.device)
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer)
    if not nouns:
        raise RuntimeError("Tokenizer did not yield eligible WordNet nouns")
    decoder = WriteEvidence(wrapper.model, nouns)

    for ordinal, (cohort, image_id) in enumerate(remaining, 1):
        started = time.perf_counter()
        image_file = args.coco_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        row = generations[cohort][image_id]
        trace, unmatched = trace_caption(wrapper, decoder, chair, image_file, row)
        seconds = time.perf_counter() - started
        _append_jsonl(path, {
            "image_id": image_id,
            "cohort": cohort,
            "layers": EVIDENCE_LAYERS,
            "response_tokens": len(row["response_token_ids"]),
            "object_positions": trace,
            "unaligned_mentions": unmatched,
            "seconds": seconds,
        })
        print(f"[{args.shard}:{ordinal}/{len(remaining)}] {cohort} image={image_id} "
              f"positions={len(trace)} unaligned={len(unmatched)} seconds={seconds:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT, help="Existing generation output root")
    parser.add_argument("--trace-dir", type=Path, help="Destination for trace JSONL shards")
    parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--chair-cache", type=Path, help="Optional full CHAIR evaluator cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=140)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or not 1 <= args.limit <= 140:
        parser.error("Expected a valid shard and an image limit within [1, 140]")
    run(args)


if __name__ == "__main__":
    main()
