#!/usr/bin/env python3
"""Paired SADI baseline / next-layer / same-layer feedback on frozen COCO images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_json
from features.llava_decoding_evidence import lexical_nouns
from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback
from models import build_model
from scripts import generation_time_evidence_drift_qwen3_batched as shared
from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from utils.config_utils import get_extraction_model_cfg, load_config

NAMES = ('sadi', 'sadi_sar_ems_next_19_20', 'sadi_sar_ems_same_19_20')
BASE_GENERATE = shared.generate_group


def generate_group(wrapper, span, group, nouns, name, max_tokens):
    if name == 'sadi':
        return BASE_GENERATE(wrapper, span, group, nouns, name, max_tokens)
    mode = {'sadi_sar_ems_next_19_20': 'next', 'sadi_sar_ems_same_19_20': 'same'}[name]
    controller = Qwen3BatchLayerFeedback(wrapper.model, span, nouns, mode=mode, strength=5.0)
    if any(set(item) != set(group[0][1]) for _, item in group):
        raise RuntimeError('Incompatible processor fields')
    inputs = {field: torch.cat([item[field] for _, item in group], dim=0).to(wrapper.device)
              for field in group[0][1]}
    args = _generation_kwargs(wrapper, MODEL)
    args['max_new_tokens'] = max_tokens
    eos_ids = wrapper.model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    torch.cuda.synchronize(wrapper.device)
    started = time.perf_counter()
    with controller.capture():
        output = wrapper.model.generate(**inputs, **args)
    torch.cuda.synchronize(wrapper.device)
    seconds = time.perf_counter() - started
    raw = output[:, inputs['input_ids'].shape[1]:].tolist()
    tokens = [shared.trim_eos(ids, eos) for ids in raw]
    if any(not ids for ids in tokens):
        raise RuntimeError('Empty caption')
    return tokens, controller.statistics(tokens), seconds


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    folder = args.output_root / 'coco100'
    folder.mkdir(parents=True, exist_ok=True)
    ids = json.loads((args.output_root / 'cohort.json').read_text())['coco100']
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError('Expected exactly 100 frozen unique images')
    paths = {name: folder / f'{name}.shard{args.shard}.jsonl' for name in NAMES}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Existing rows: inspect the paired batch plan before any resume')
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config['hf_name'] = str(MODEL_ROOT)
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    if not nouns:
        raise RuntimeError('Empty WordNet entity vocabulary')
    if args.verify:
        from scripts.verify_qwen3_layer_feedback import verify
        result = verify(wrapper, nouns)
        atomic_write_json(args.output_root / 'verification.json', result)
        print('PREFLIGHT_PASSED', flush=True)
    existing = {name: set() for name in NAMES}
    with patch.object(shared, 'generate_group', generate_group):
        groups = shared.same_span_batches(wrapper, ids[args.shard::2], COCO_ROOT, 16)
        for span, group in groups:
            shared.write_group(wrapper, span, group, nouns, existing, paths, 512, args.shard)
    if any(len(rows) != 50 for rows in existing.values()):
        raise RuntimeError('Incomplete shard')
    print('LAYER_FEEDBACK_SHARD_COMPLETE', args.shard, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--verify', action='store_true')
    run(parser.parse_args())
