"""Replay saved positions and decompose candidate margins with visual writes removed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.trace_qwen3_object_writes import OUT, SOURCE, groups, load, write

LABELS = OUT / 'qwen3_sadi_first_object_trace'
RUN = OUT / 'qwen3_sadi_visual_removal_trace'
LAYERS = (24, 28, 31)


def worker(shard):
    import torch
    from transformers import LogitsProcessor, StoppingCriteria

    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback
    from features.qwen3_visual_write_contrast import readout_contrast
    from models import build_model
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    from utils.config_utils import get_extraction_model_cfg, load_config

    planned = load(RUN / 'protocol.json')['batches'][shard]
    label_rows = [json.loads(line) for line in (LABELS / f'trace.shard{shard}.jsonl').read_bytes().splitlines()]
    by_batch = {}
    for row in label_rows:
        assert row['replay_matches_target'] and row['shard'] == shard
        by_batch.setdefault(row['batch'], []).append(row)

    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config['hf_name'] = str(MODEL_ROOT)
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    eos_id = wrapper.model.generation_config.eos_token_id
    eos = eos_id[0] if isinstance(eos_id, list) else eos_id

    class Observer(Qwen3BatchLayerFeedback):
        def __init__(self, span, targets):
            super().__init__(wrapper.model, span, nouns)
            self.sources, self.targets = LAYERS, tuple(i + 1 for i in LAYERS)
            self.positions = targets
            self.projections = {}

        def _budget(self, layer):
            selected = self.positions.get(self.step, {})
            if not selected:
                return [], None, []
            rows = list(selected)
            state = self.scratch[layer]
            # Keep the same batch dimension for the layer candidate ranking.
            post = self.decoder.norm(state['ffn_residual'] + state['update'])
            native = self.head(post)[rows].float()
            native_scores, native_ids = native.topk(8, dim=-1)
            words = [selected[row]['target_token_id'] for row in rows]
            top = native_ids.tolist()
            rivals = [next((other for other in ids if other in nouns and other != token),
                           next(other for other in ids if other != token)) for token, ids in zip(words, top, strict=True)]
            tokens = native_ids.new_tensor(list(zip(words, rivals, strict=True)))
            vis, gain, vis_fixed, after = readout_contrast(self, layer, rows, tokens)
            stats = torch.cat((vis, gain, vis_fixed, after), dim=-1).tolist()
            for row, rival, ids, value, peak in zip(rows, rivals, top, stats, native_scores[:, 0].tolist(), strict=True):
                selected[row]['contrast'][str(layer)] = {
                    'rival': rival, 'rival_is_entity': rival in nouns, 'target_in_top8': selected[row]['target_token_id'] in ids,
                    'target_gap': peak - float(native[rows.index(row), selected[row]['target_token_id']]),
                    'visual_target': value[0], 'visual_margin': value[0] - value[1],
                    'ffn_target': value[2], 'ffn_margin': value[2] - value[3],
                    'visual_fixed_ffn_target': value[4], 'visual_fixed_ffn_margin': value[4] - value[5],
                    'post_margin': value[6] - value[7],
                }
            return [], None, []

    class Replay(LogitsProcessor):
        def __init__(self, saved, positions, prompt_length):
            self.saved, self.positions, self.prompt_length = saved, positions, prompt_length

        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - self.prompt_length
            for row, position in self.positions.get(step, {}).items():
                position['counterfactual_replay_matches'] = int(scores[row].argmax()) == position['target_token_id']
            chosen = [r['response_token_ids'][step] if step < len(r['response_token_ids']) else eos for r in self.saved]
            return torch.full_like(scores, -float('inf')).scatter_(1, input_ids.new_tensor(chosen)[:, None], 0)

    class Stop(StoppingCriteria):
        def __init__(self, length):
            self.length = length

        def __call__(self, input_ids, scores, **_):
            return input_ids.new_full((len(input_ids),), input_ids.shape[1] >= self.length, dtype=torch.bool)

    with torch.inference_mode(), (RUN / f'trace.shard{shard}.jsonl').open('x', encoding='utf-8') as stream:
        for i, (batch, expected) in enumerate(zip(groups(shard), planned, strict=True)):
            ids = [r['image_id'] for r in batch]
            assert ids == expected['ids']
            targets = {}
            for label in by_batch[i]:
                row = ids.index(label['image_id'])
                assert batch[row]['response_token_ids'][label['step']] == label['target_token_id']
                targets.setdefault(label['step'], {})[row] = dict(label, contrast={})
            (span, inputs), = same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [id for id, _ in inputs] == ids and span[1] - span[0] == batch[0]['visual_token_count']
            model_inputs = {k: torch.cat([v[k] for _, v in inputs], dim=0).to(wrapper.device) for k in inputs[0][1]}
            prompt_length = model_inputs['input_ids'].shape[1]
            kwargs = _generation_kwargs(wrapper, MODEL)
            kwargs.update(max_new_tokens=256, logits_processor=[Replay(batch, targets, prompt_length)],
                          stopping_criteria=[Stop(prompt_length + max(targets) + 1)])
            observer = Observer(span, targets)
            with observer.capture():
                wrapper.model.generate(**model_inputs, **kwargs)
            for target in targets.values():
                for row in target.values():
                    assert len(row['contrast']) == len(LAYERS)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print('BATCH', shard, i, len(by_batch[i]), flush=True)


def run():
    assert load(LABELS / 'status.json')['status'] == 'complete'
    plan = load(LABELS / 'protocol.json')
    sources = [LABELS / name for name in ('protocol.json', 'trace.shard0.jsonl', 'trace.shard1.jsonl')]
    sources += [Path(__file__), REPO/'features/qwen3_visual_write_contrast.py']
    plan.update(contrast_layers=LAYERS, contrast_sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                interpretation='Remove the realized visual additive write; keep textual writes and the observed FFN update fixed.')
    RUN.mkdir()
    write(RUN / 'protocol.json', plan)
    children = []
    try:
        for shard in (0, 1):
            with (RUN / f'gpu{shard}.log').open('w') as log:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                           NLTK_DATA=str(OUT / 'generation_time_evidence_drift_llava/nltk_data'))
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', str(shard)], cwd=REPO, env=env,
                                                  stdout=log, stderr=subprocess.STDOUT))
        write(RUN / 'status.json', {'status': 'running', 'driver': os.getpid(), 'workers': [p.pid for p in children]})
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Trace worker failed')
            time.sleep(2)
        if any(p.returncode for p in children):
            raise RuntimeError('Trace worker failed')
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in plan['contrast_sources'].items())
        write(RUN / 'status.json', {'status': 'complete'})
    except BaseException as exc:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        for proc in children:
            proc.wait()
        write(RUN / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
