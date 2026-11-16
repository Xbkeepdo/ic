#!/usr/bin/env python3
"""Run the fixed Qwen3 caption interventions in same-span multimodal batches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
from pathlib import Path
import sys
import time

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.llava_decoding_evidence import lexical_nouns
from features.qwen3_sar_ems_attention import LATE_VARIANTS, Qwen3LateAttention, VARIANTS
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from models import build_model
from scripts.generation_time_evidence_drift_llava import _read_rows
from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
from scripts.generation_time_sar_ems_seed43 import _append_jsonl, _generation_kwargs, _read_json
from utils.config_utils import get_extraction_model_cfg, load_config

DEFAULT_VARIANTS = ("svar", "sadi", "sadi_sar_ems_attention")
OUT = ROOT / "outputs/generation_time_evidence_drift_qwen3_coco500"


def same_span_batches(wrapper, image_ids, coco_root, batch_size):
    pending = defaultdict(list)
    for image_id in image_ids:
        path = coco_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            inputs = wrapper._prompt_inputs(image, "Describe this image.")
        input_ids = inputs["input_ids"]
        span = wrapper._find_vision_token_range(input_ids[0])
        if not inputs["attention_mask"].all().item() or span[1] >= input_ids.shape[1]:
            raise RuntimeError(f"Invalid unpadded caption prompt for image {image_id}")
        group = pending[(span, input_ids.shape[1])]
        group.append((image_id, inputs))
        if len(group) == batch_size:
            del pending[(span, input_ids.shape[1])]
            yield span, group
    for (span, _), group in pending.items():
        yield span, group


def trim_eos(tokens, eos):
    for position, token in enumerate(tokens):
        if token in eos:
            return tokens[:position+1]
    return tokens


def generate_group(wrapper, span, group, nouns, name, max_tokens, *, native_single=False):
    controller_type = Qwen3LateAttention if native_single else Qwen3BatchLateAttention
    controller = controller_type(wrapper.model, span, nouns, name)
    if any(set(inputs) != set(group[0][1]) for _, inputs in group):
        raise RuntimeError("Incompatible processor output fields in a caption batch")
    inputs = {field: torch.cat([item[field] for _, item in group], dim=0).to(wrapper.device)
              for field in group[0][1]}
    args = _generation_kwargs(wrapper, MODEL)
    args["max_new_tokens"] = max_tokens
    eos_ids = wrapper.model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    torch.cuda.synchronize(wrapper.device)
    started = time.perf_counter()
    with controller.capture():
        output = wrapper.model.generate(**inputs, **args)
    torch.cuda.synchronize(wrapper.device)
    seconds = time.perf_counter() - started
    raw = output[:, inputs["input_ids"].shape[1]:].tolist()
    tokens = [trim_eos(ids, eos) for ids in raw]
    if any(not ids for ids in tokens):
        raise RuntimeError(f"Empty caption on batch {[i for i, _ in group]} ({name})")
    if native_single:
        stats = [controller.statistics(tokens[0])] if name in LATE_VARIANTS else [{}]
    else:
        stats = controller.statistics(tokens)
    return tokens, stats, seconds


def write_group(wrapper, span, group, nouns, existing, paths, max_tokens, shard):
    image_ids = [i for i, _ in group]
    outputs = {}
    try:
        for name in paths:
            if any(i not in existing[name] for i in image_ids):
                outputs[name] = generate_group(wrapper, span, group, nouns, name, max_tokens)
    except torch.OutOfMemoryError:
        outputs.clear()
        if len(group) == 1:
            raise
        retry_smaller = True
    else:
        retry_smaller = False
    if retry_smaller:
        print(f"[oom:{shard}] splitting {len(group)} caption images on the same GPU", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        mid = len(group) // 2
        write_group(wrapper, span, group[:mid], nouns, existing, paths, max_tokens, shard)
        write_group(wrapper, span, group[mid:], nouns, existing, paths, max_tokens, shard)
        return

    for name, (tokens, statistics, seconds) in outputs.items():
        texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        for i, ids, text, stats in zip(image_ids, tokens, texts, statistics, strict=True):
            if i in existing[name]:
                continue
            _append_jsonl(paths[name], {
                "image_id": i, "model": MODEL, "variant": name,
                "generated_text": text.strip(), "response_token_ids": ids,
                "seconds": seconds / len(group),
                "batch_size": len(group), "batch_wall_seconds": seconds,
                "visual_token_count": span[1] - span[0],
                "sar_ems_statistics": stats, "decoder": "native_qwen3_generate_same_span_batch",
            })
            existing[name].add(i)
        print(f"[{shard}] {name}: {len(existing[name])} images, batch={len(group)} "
              f"tokens={sum(map(len,tokens))}/{max(map(len,tokens))}, seconds={seconds:.3f}",
              flush=True)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config["hf_name"] = str(args.model_root)
    wrapper = build_model(MODEL, config, device=args.device)
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start="Ġ")
    if not nouns:
        raise RuntimeError("Expected WordNet entity tokens for Qwen3")
    if args.smoke_manifest is not None:
        manifest = _read_json(args.smoke_manifest)
        ids = next(iter(manifest.values()))
        groups = same_span_batches(wrapper, ids[args.shard::args.shards], args.coco_root, args.batch_size)
        span, group = max(groups, key=lambda pair: len(pair[1]))
        if len(group) < 2:
            raise RuntimeError("Smoke cohort has no multi-image same-span batch")
        for name in args.variants:
            batched, bs, elapsed = generate_group(wrapper, span, group, nouns, name, 24)
            native_seconds = 0.0
            mismatches = []
            for index, item in enumerate(group):
                native, ns, duration = generate_group(wrapper, span, [item], nouns, name, 24,
                                                      native_single=True)
                if index == 0:
                    one, os, _ = generate_group(wrapper, span, [item], nouns, name, 24)
                    if one != native or os != ns:
                        raise RuntimeError(f"Single-row batched implementation failed native parity: {name}")
                if batched[index] != native[0] or bs[index] != ns[0]:
                    mismatches.append(item[0])
                native_seconds += duration
            print(f"[smoke:{args.shard}] {name} images={[i for i, _ in group]} "
                  f"batch={elapsed:.3f}s native={native_seconds:.3f}s mismatches={mismatches}", flush=True)
        return

    ids = list(map(int, _read_json(args.output_root / "cohort.json")[args.cohort]))
    if not ids or len(set(ids)) != len(ids):
        raise RuntimeError("Expected a nonempty cohort with unique COCO image IDs")
    images = ids[args.shard::args.shards]
    folder = args.output_root / args.cohort
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{name}.shard{args.shard}.jsonl" for name in args.variants}
    existing = {name: set(_read_rows(path)) for name, path in paths.items()}
    if any(not rows.issubset(images) for rows in existing.values()):
        raise RuntimeError("Saved generation does not belong to this frozen shard")
    for span, group in same_span_batches(wrapper, images, args.coco_root, args.batch_size):
        if not all(all(i in existing[name] for name in args.variants) for i, _ in group):
            write_group(wrapper, span, group, nouns, existing, paths, wrapper.generation_max_new_tokens, args.shard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    parser.add_argument("--cohort", default="coco500")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-manifest", type=Path)
    args = parser.parse_args()
    if (not 0 <= args.shard < args.shards or args.batch_size < 1
            or len(set(args.variants)) != len(args.variants)):
        parser.error("Specify a valid shard, a positive batch size, and distinct variants")
    run(args)


if __name__ == "__main__":
    main()
