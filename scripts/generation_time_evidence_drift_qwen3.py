#!/usr/bin/env python3
"""Evaluate the unchanged late SAR/EMS recipe on native Qwen3-VL decoding."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.llava_decoding_evidence import lexical_nouns
from features.qwen3_sar_ems_attention import LATE_VARIANTS, Qwen3LateAttention, VARIANTS
from models import build_model
from scripts.generation_time_evidence_drift_llava import _read_rows
from scripts.generation_time_sar_ems_seed43 import _append_jsonl, _generation_kwargs, _inputs, _read_json
from utils.config_utils import get_extraction_model_cfg, load_config

MODEL = "qwen3_vl_8b"
MODEL_ROOT = Path("/root/rivermind-data/models/qwen3_vl_8b")
COCO_ROOT = Path("/root/rivermind-data/dataset/coco")
CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
OUT = ROOT / "outputs/generation_time_evidence_drift_qwen3"


@torch.inference_mode()
def run(args) -> None:
    torch.set_num_threads(1)
    manifest = _read_json(args.output_root / "cohort.json")
    cohort = list(map(int, manifest[args.cohort]))
    if len(set(cohort)) != len(cohort):
        raise ValueError("Qwen3 cohort contains duplicate image IDs")
    if args.limit is not None and not 1 <= args.limit <= len(cohort):
        raise ValueError(f"Qwen3 limit must be within [1, {len(cohort)}]")
    image_ids = cohort[:args.limit][args.shard::args.shards]
    folder = args.output_root / args.cohort
    folder.mkdir(parents=True, exist_ok=True)
    paths = {name: folder / f"{name}.shard{args.shard}.jsonl" for name in args.variants}
    done = {name: _read_rows(path) for name, path in paths.items()}

    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config["hf_name"] = str(args.model_root)
    wrapper = build_model(MODEL, config, device=args.device)
    wrapper.model.eval().requires_grad_(False)
    nouns = frozenset()
    if any(name in LATE_VARIANTS for name in args.variants):
        nouns = lexical_nouns(wrapper.tokenizer, word_start="Ġ")
        if len(nouns) < 10 or not max(nouns) < wrapper.model.get_output_embeddings().weight.shape[0]:
            raise RuntimeError("Qwen3 tokenizer did not yield a valid WordNet entity vocabulary")
    generation_kwargs = _generation_kwargs(wrapper, MODEL)
    print(f"[qwen3] {len(image_ids)} images, {len(nouns)} entity tokens", flush=True)

    for index, image_id in enumerate(image_ids, 1):
        pending = [name for name in args.variants if args.smoke or image_id not in done[name]]
        if not pending:
            continue
        image_path = args.coco_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        with Image.open(image_path) as image:
            inputs = _inputs(wrapper, image.convert("RGB"), MODEL)
        prompt = inputs["input_ids"]
        span = wrapper._find_vision_token_range(prompt[0])
        if span[1] >= prompt.shape[1] or not inputs["attention_mask"].all().item():
            raise ValueError("Expected a single unpadded Qwen3 image prompt ending in text")

        for name in pending:
            intervention = Qwen3LateAttention(wrapper.model, span, nouns, name)
            torch.cuda.synchronize(wrapper.device)
            before = time.perf_counter()
            with intervention.capture():
                output = wrapper.model.generate(**inputs, **generation_kwargs)
            torch.cuda.synchronize(wrapper.device)
            seconds = time.perf_counter() - before
            tokens = output[0, prompt.shape[1]:].tolist()
            if not tokens:
                raise RuntimeError(f"Empty Qwen3 generation for image {image_id}, {name}")
            statistics = intervention.statistics(tokens) if name in LATE_VARIANTS else {}
            if not args.smoke:
                _append_jsonl(paths[name], {
                    "image_id": image_id,
                    "model": MODEL,
                    "variant": name,
                    "generated_text": wrapper.processor.decode(
                        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    ).strip(),
                    "response_token_ids": tokens,
                    "seconds": seconds,
                    "visual_token_count": span[1] - span[0],
                    "sar_ems_statistics": statistics,
                    "decoder": "native_qwen3_generate",
                })
            print(f"[qwen3:{args.shard}:{index}/{len(image_ids)}] {name} image={image_id} "
                  f"visual={span[1]-span[0]} tokens={len(tokens)} "
                  f"guided={statistics.get('guided_steps', 0)} seconds={seconds:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--coco-root", type=Path, default=COCO_ROOT)
    parser.add_argument("--cohort", default="rebuttal_holdout")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["svar", "sadi", "sadi_sar_ems_attention"])
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run the same protocol without writing captions")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or len(set(args.variants)) != len(args.variants):
        parser.error("Expected a valid shard and distinct variants")
    run(args)


if __name__ == "__main__":
    main()
