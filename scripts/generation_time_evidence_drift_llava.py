#!/usr/bin/env python3
"""LLaVA paired decoding experiments with a shared image and prompt KV cache."""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_model
from features.llava_decoding_evidence import (
    EVIDENCE_LAYERS, VARIANTS, WriteEvidence, has_repeat_attention_span,
    has_repeated_noun_phrase, lexical_nouns,
)
from scripts.generation_time_sar_ems_seed43 import (
    _append_jsonl, _generation_kwargs, _inputs, _read_json,
)
from scripts.generation_time_svar_official_seed43 import heads_guided_attention
from utils.config_utils import get_extraction_model_cfg, load_config

MODEL = "llava_1_5_7b"
COCO_ROOT = Path("/root/rivermind-data/dataset/coco")
MODEL_ROOT = Path("/root/rivermind-data/models/llava_1_5_7b")
OUT = ROOT / "outputs/generation_time_evidence_drift_llava"
CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
SVAR_LAYERS = tuple(range(5, 19))
REPEAT_BASELINE = "sadi_sar_ems_attention"
VGA_VARIANTS = {"vga", "vga_write"}


def _response(wrapper, ids):
    return wrapper.tokenizer.decode(ids, skip_special_tokens=True)


def _clone_cache(cache):
    return DynamicCache.from_legacy_cache(
        tuple((keys.clone(), values.clone()) for keys, values in cache.to_legacy_cache())
    )


def _prefill(wrapper, inputs):
    """Cache every prompt state except the last query, which predicts token 0."""
    kwargs = dict(inputs)
    kwargs["input_ids"] = kwargs["input_ids"][:, :-1]
    kwargs["attention_mask"] = kwargs["attention_mask"][:, :-1]
    return wrapper.model(**kwargs, use_cache=True, logits_to_keep=1).past_key_values


def _decode(wrapper, prompt_tail, prefix_cache, selector=None, prepare=None, replay=None):
    cache = _clone_cache(prefix_cache)
    response = []
    changes = []
    token = prompt_tail
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for step in range(wrapper.generation_max_new_tokens):
        if prepare is not None:
            prepare(response)
        if replay is None:
            result = wrapper.model(
                input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1,
            )
        else:
            result, token_id, change = replay.step(wrapper.model, token, cache, response)
            if change is not None:
                changes.append(change)
        cache = result.past_key_values
        logits = result.logits[:, -1, :]
        if replay is not None:
            token = torch.full_like(token, token_id)
        elif selector is None:
            token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(token.item())
        else:
            token_id, change = selector(logits, step, response)
            if change is not None:
                changes.append(change)
            token = torch.full_like(token, token_id)
        response.append(token_id)
        if token_id in eos:
            break
    return response, changes


def _native(wrapper, inputs, span, variant):
    context = (
        heads_guided_attention(wrapper.model, layers=SVAR_LAYERS,
                               image_start=span[0], image_end=span[1], alpha=0.5)
        if variant == "svar" else nullcontext()
    )
    with context:
        result = wrapper.model.generate(**inputs, **_generation_kwargs(wrapper, MODEL))
    return result[0, inputs["input_ids"].shape[1]:].tolist()


def _read_rows(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id in rows:
                raise ValueError(f"Duplicate image IDs in {path}")
            rows[image_id] = row
    return rows


def _repeat_baseline_rows(folder: Path) -> dict[int, dict]:
    rows = {}
    for path in sorted(folder.glob(f"{REPEAT_BASELINE}.shard*.jsonl")):
        shard = _read_rows(path)
        if rows.keys() & shard.keys():
            raise ValueError(f"Duplicate repeat baseline images in {path}")
        rows.update(shard)
    return rows


@torch.inference_mode()
def run(args) -> None:
    torch.set_num_threads(1)
    config = load_config(str(CONFIG))
    model_config = dict(get_extraction_model_cfg(config, MODEL))
    model_config["hf_name"] = str(args.model_root)
    wrapper = build_model(MODEL, model_config, device=args.device)
    wrapper.model.eval().requires_grad_(False)
    needs_evidence = any(VARIANTS[name].needs_visual_cache for name in args.variants)
    needs_sar_ems = any(VARIANTS[name].sar_ems_attention_strength > 0 or VARIANTS[name].replay
                       for name in args.variants)
    needs_patch_mixture = any(VARIANTS[name].sar_ems_patch_strength > 0 for name in args.variants)
    needs_sar_ems_allocation = any(VARIANTS[name].sar_ems_allocation > 0 for name in args.variants)
    needs_write_prior = any(VARIANTS[name].write_prior_strength > 0 for name in args.variants)
    needs_sar_ems_selection = any(VARIANTS[name].compares_sar_ems for name in args.variants)
    needs_transfer_evidence = any(VARIANTS[name].transfer_uses_writes for name in args.variants)
    uses_calibration = any(VARIANTS[name].calibrated_penalty > 0.0 for name in args.variants)
    calibration = None
    if uses_calibration:
        path = args.calibration or args.output_root / "sar_ems_decoding_gate_seed43.json"
        calibration = _read_json(path)
        if tuple(calibration["layers"]) != EVIDENCE_LAYERS or not 0 <= calibration["threshold"] < 1:
            raise ValueError(f"Invalid layer selection or threshold in {path}")
    nouns = lexical_nouns(wrapper.tokenizer) if needs_evidence else ()
    decoder = WriteEvidence(wrapper.model, nouns, calibration)
    if needs_evidence and not nouns:
        raise RuntimeError("Tokenizer did not yield eligible WordNet nouns")
    cohort_ids = _read_json(args.output_root / "cohort.json")[args.cohort][:args.limit]
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("Duplicate cohort images")
    images = cohort_ids[args.shard::args.shards]
    if args.image_ids is not None:
        selected = set(map(int, _read_json(args.image_ids)))
        if not selected or not selected.issubset(cohort_ids):
            raise ValueError("The supplied image IDs must form a nonempty subset of the selected cohort")
        images = [image_id for image_id in images if image_id in selected]
    folder = args.output_root / args.cohort
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{name}.shard{args.shard}.jsonl" for name in args.variants}
    existing = {name: _read_rows(path) for name, path in paths.items()}
    repeat_variants = {name for name in args.variants if VARIANTS[name].repeat_mode is not None
                       or VARIANTS[name].repeat_attention is not None}
    repeat_baseline = _repeat_baseline_rows(folder) if repeat_variants and not args.smoke else {}
    for index, image_id in enumerate(images, 1):
        variants = [name for name in args.variants if image_id not in existing[name] or args.smoke]
        base_row = repeat_baseline.get(image_id)
        candidates = repeat_variants.intersection(variants)
        if base_row is not None and candidates:
            ids = base_row["response_token_ids"]
            trigger = {}
            for name in sorted(candidates):
                attention_mode = VARIANTS[name].repeat_attention
                if attention_mode not in trigger:
                    trigger[attention_mode] = (has_repeat_attention_span(ids, nouns, sustained=attention_mode == "early_write")
                                               if attention_mode else has_repeated_noun_phrase(ids, nouns))
                if not trigger[attention_mode]:
                    _append_jsonl(paths[name], dict(base_row, variant=name, reused_from=REPEAT_BASELINE))
                    variants.remove(name)
                    print(f"[{args.shard}:{index}/{len(images)}] {name} image={image_id} reused", flush=True)
        if not variants:
            continue
        start = time.perf_counter()
        image_file = args.coco_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        with Image.open(image_file) as opened:
            inputs = _inputs(wrapper, opened.convert("RGB"), MODEL)
        span = wrapper._find_visual_token_range(inputs, wrapper._image_token_id())
        if span[1] >= inputs["input_ids"].shape[1] or not inputs["attention_mask"].all().item():
            raise ValueError("Decoding requires an unpadded prompt ending in text")
        decoder.reset_image(span)
        vga = None
        if VGA_VARIANTS.intersection(variants):
            from features.llava_vga import VgaAttention
            vga = VgaAttention(wrapper.model, wrapper.tokenizer, *span)
        vga_prefill = vga.capture_prefill() if vga is not None else nullcontext()
        with (
            vga_prefill,
            decoder.capture(prefill=True, variant=VARIANTS["greedy"], need_prefill=needs_evidence,
                            need_sar_ems=needs_sar_ems, need_patch_mixture=needs_patch_mixture,
                            need_write_prior=needs_write_prior, need_sar_ems_selection=needs_sar_ems_selection,
                            need_transfer_evidence=needs_transfer_evidence,
                            need_sar_ems_allocation=needs_sar_ems_allocation),
        ):
            prefix = _prefill(wrapper, inputs)
        torch.cuda.synchronize(wrapper.device)
        prefill_seconds = time.perf_counter() - start
        prompt_tail = inputs["input_ids"][:, -1:]
        for name in variants:
            start = time.perf_counter()
            variant = VARIANTS[name]
            decoder.candidate_steps = 0
            decoder.ffn_steps.clear()
            decoder.reset_sar_ems_steps()
            prepare = None
            replay = None
            if variant.replay:
                from features.llava_sar_ems_replay import SadiSarEmsReplay
                replay = SadiSarEmsReplay(decoder, mode=variant.replay, tokenizer=wrapper.tokenizer)
            if variant.repeat_attention is not None:
                def prepare(tokens):
                    decoder.prepare_repeat_attention(tokens, sustained=variant.repeat_attention == "early_write")
            if name in VGA_VARIANTS:
                prepare = vga.prepare
                capture = vga.capture(projected_write=name == "vga_write")
            else:
                capture = decoder.capture(prefill=False, variant=variant)
            with capture:
                if replay is not None:
                    with replay.capture():
                        ids, changes = _decode(wrapper, prompt_tail, prefix, replay=replay)
                elif variant.rerank:
                    def select(logits, step, response):
                        return decoder.select(logits, step, response, variant)
                    ids, changes = _decode(wrapper, prompt_tail, prefix, select, prepare=prepare)
                else:
                    ids, changes = _decode(wrapper, prompt_tail, prefix, prepare=prepare)
            seconds = time.perf_counter() - start
            ffn_statistics = decoder.ffn_statistics()
            sar_ems_statistics = decoder.sar_ems_statistics(ids) if variant.uses_sar_ems else {}
            transfer_statistics = decoder.attention_transfer_statistics()
            if args.smoke and name in ("greedy", "svar"):
                expected = _native(wrapper, inputs, span, name)
                if ids != expected:
                    first_diff = next((j for j, (a, b) in enumerate(zip(ids, expected)) if a != b),
                                      min(len(ids), len(expected)))
                    raise RuntimeError(f"{name}: split prefill vs native diverged at token {first_diff}, "
                                       f"lengths {len(ids)}/{len(expected)}")
                print(f"[smoke] {name} image={image_id} {len(ids)} identical tokens", flush=True)
                continue
            if args.smoke:
                print(f"[smoke] {name} image={image_id} tokens={len(ids)} "
                      f"candidate_steps={decoder.candidate_steps} changes={len(changes)} "
                      f"ffn={ffn_statistics} sar_ems={sar_ems_statistics}", flush=True)
                continue
            _append_jsonl(paths[name], {
                "image_id": image_id,
                "generated_text": _response(wrapper, ids),
                "response_token_ids": ids,
                "seconds": seconds,
                "prefill_seconds": prefill_seconds,
                "changes": changes,
                "candidate_steps": decoder.candidate_steps,
                "ems_comparison_steps": decoder.ems_comparison_steps,
                "ffn_statistics": ffn_statistics,
                "sar_ems_statistics": sar_ems_statistics,
                "attention_transfer_statistics": transfer_statistics,
                **({"replay_statistics": replay.statistics()} if replay else {}),
                **({"sadi_head_selection": decoder.sadi_head_selection()} if variant.sadi_head_rank else {}),
                "variant": name,
                "decoder": "shared_prompt_prefix_kv",
            })
            print(f"[{args.shard}:{index}/{len(images)}] {name} image={image_id} "
                  f"tokens={len(ids)} changes={len(changes)}/{decoder.candidate_steps} "
                  f"ffn={ffn_statistics.get('accepted_updates', 0)} "
                  f"guided={sar_ems_statistics.get('guided_steps', 0)} decode={seconds:.3f}s "
                  f"prefill={prefill_seconds:.3f}s",
                  flush=True)
        del prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--cohort", choices=("pilot", "confirmation", "rebuttal_holdout",
                                            "rebuttal_replication", "mitigation_external",
                                            "mitigation_reserve"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--image-ids", type=Path, help="JSON list restricting the cohort without changing shard assignments")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["greedy", "svar"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.shard < args.shards) or args.limit <= 0:
        parser.error("Invalid shard index or limit")
    run(args)


if __name__ == "__main__":
    main()
