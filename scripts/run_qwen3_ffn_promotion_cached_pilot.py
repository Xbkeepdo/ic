"""Replay only batches where the positive-FFN predicate differs from a cached gate."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_ffn_promotion_pilot as full
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-ffn-promotion-cached-pilot')
CACHE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-first-class-conflict-pilot')
NAME = full.NAME


def promoted_cached_row(row, source):
    result = copy.deepcopy(row)
    result['variant'] = NAME
    result['generation_provenance'] = dict(kind='gate_equivalent_cached_trajectory', source=str(source),
        original_variant=row['variant'], fresh_generation=False)
    stats = result['sar_ems_statistics']
    decisions = stats.pop('conflict_decisions')
    assert all(d['permitted'] == (d['gain'] > 0) for d in decisions)
    stats.pop('conflict_sign_gate')
    stats['blocked_by_ffn_promotion'] = stats.pop('blocked_by_conflict_sign')
    stats.update(ffn_promotion_gate=True, require_negative_visual_support=False,
                 promotion_decisions=decisions)
    return result


def prepare():
    with patch.object(full, 'OUT', OUT):
        plan = full.prepare()
    assert shared.read(CACHE / 'run_status.json')['status'] == 'complete'
    assert shared.read(CACHE / 'first_class_conflict_audit.json')['passed']
    cached_plan = shared.read(CACHE / 'protocol.json')
    for filename, expected in cached_plan['source_sha256'].items():
        assert shared.sha(Path(filename)) == expected, filename
    cached_name, = cached_plan['candidates']
    batches = []
    for shard in (0, 1):
        path = CACHE / 'coco100' / f'{cached_name}.shard{shard}.jsonl'
        plan['source_sha256'][str(path)] = shared.sha(path)
        source_groups = shared.source_groups(path)
        controls = shared.source_groups(OUT / 'coco100' / f'sadi.shard{shard}.jsonl')
        assert shared.group_keys(source_groups) == shared.group_keys(controls)
        assert (CACHE / 'coco100' / f'sadi.shard{shard}.jsonl').read_bytes() == (
            OUT / 'coco100' / f'sadi.shard{shard}.jsonl').read_bytes()
        for index, group in enumerate(source_groups):
            decisions = [d for row in group for d in row['sar_ems_statistics']['conflict_decisions']]
            assert all(d['permitted'] == (d['visual_support'] < 0 and d['gain'] > 0) for d in decisions)
            different = [dict(image_id=row['image_id'], **d) for row in group
                         for d in row['sar_ems_statistics']['conflict_decisions']
                         if d['permitted'] != (d['gain'] > 0)]
            batches.append(dict(shard=shard, index=index, ids=[r['image_id'] for r in group],
                source=str(path), reuse=not different, differing_decisions=different,
                checked_risk_positive_decisions=len(decisions)))
    for filename in ('scripts/run_qwen3_ffn_promotion_cached_pilot.py',
                     'scripts/audit_qwen3_ffn_promotion_cached.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    for path in (CACHE / 'protocol.json', CACHE / 'first_class_conflict_audit.json'):
        plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-ffn-promotion-cached-pilot', gate_cache_source=str(CACHE),
        cache_replay_batches=batches,
        reused_images=sum(len(b['ids']) for b in batches if b['reuse']),
        fresh_images=sum(len(b['ids']) for b in batches if not b['reuse']),
        reuse_argument='Same inputs, ordered batch, base, numerical action and acceptance. For reused batches, every risk-positive predicate along the entire saved trajectory is unchanged. By prefix induction the trajectories are identical. Regenerate an entire original batch at any differing decision.',
        reuse_limitation='99 cached trajectories are not99 fresh generations. Their historical timings are retained for provenance and are not this ablation runtime.')
    assert plan['reused_images'] + plan['fresh_images'] == 100
    assert plan['fresh_images'] > 0
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker():
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_ffn_promotion import Qwen3BatchFFNPromotion
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as batching
    from scripts import generation_time_qwen3_layer_feedback as generation
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config

    plan = shared.read(OUT / 'protocol.json')
    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL), hf_name=str(MODEL_ROOT))
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    checked = []
    with torch.inference_mode():
        for shard in (0, 1):
            path = OUT / 'coco100' / f'{NAME}.shard{shard}.jsonl'
            with path.open('x', encoding='utf-8') as stream:
                for batch in (b for b in plan['cache_replay_batches'] if b['shard'] == shard):
                    group = shared.source_groups(Path(batch['source']))[batch['index']]
                    assert [r['image_id'] for r in group] == batch['ids']
                    if batch['reuse']:
                        rows = [promoted_cached_row(row, batch['source']) for row in group]
                    else:
                        (span, inputs), = batching.same_span_batches(wrapper, batch['ids'], COCO_ROOT, len(group))
                        assert [i for i, _ in inputs] == batch['ids']
                        assert span[1] - span[0] == group[0]['visual_token_count']
                        controls = shared.source_groups(OUT / 'coco100' / f'sadi.shard{shard}.jsonl')[batch['index']]
                        reference, _, _ = generation.generate_group(wrapper, span, inputs, allowed, 'sadi', 64)
                        assert reference == [r['response_token_ids'][:64] for r in controls]
                        checked.append(dict(shard=shard, batch=batch['index'], ids=batch['ids'], native_sadi_prefix_exact=True))
                        cache_path = Path(plan['source_trace']) / f'semantics/shard{shard}.batch{batch["index"]}.npz'
                        with np.load(cache_path, allow_pickle=False) as saved:
                            cache = {k: saved[k] for k in saved.files}
                        assert cache['image_ids'].tolist() == batch['ids'] and cache['span'].tolist() == list(span)

                        def factory(model, current_span, nouns, mode, strength):
                            assert mode == 'next' and current_span == span
                            return Qwen3BatchFFNPromotion(model, span, allowed, probe=probe,
                                semantic_tokens=cache['token_ids'].tolist(), cached_semantics=torch.from_numpy(cache['E_l']),
                                guided=True, tokenizer=wrapper.tokenizer, history_parser=parser)

                        with patch.object(generation, 'Qwen3BatchLayerFeedback', factory):
                            tokens, stats, seconds = generation.generate_group(
                                wrapper, span, inputs, allowed, shared.POSITIVE, plan['max_new_tokens'])
                        texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        rows = [dict(image_id=image_id, model=MODEL, variant=NAME, generated_text=text.strip(),
                            response_token_ids=ids, seconds=seconds / len(group), batch_size=len(group),
                            batch_wall_seconds=seconds, visual_token_count=span[1] - span[0], sar_ems_statistics=info,
                            decoder='native_qwen3_generate_same_span_batch', generation_provenance=dict(
                                kind='fresh_affected_original_batch', fresh_generation=True))
                            for (image_id, _), ids, text, info in zip(inputs, tokens, texts, stats, strict=True)]
                        print('FRESH_BATCH', batch['ids'], seconds, flush=True)
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    shared.write(OUT / 'sparse_native_replay.json', dict(passed=True, batches=checked))


def run():
    plan = prepare()
    children = []
    queued = time.time()
    try:
        shared.write(OUT / 'run_status.json', dict(status='waiting_for_predecessor', pid=os.getpid(), queued=queued))
        while True:
            state = shared.read(full.PREDECESSOR / 'run_status.json')
            if state['status'] == 'complete':
                break
            assert state['status'] != 'failed', state
            os.kill(state['pid'], 0)
            time.sleep(5)
        while max(map(int, subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines())) >= 100:
            time.sleep(5)
        started = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                   NLTK_DATA=str(shared.OUT / 'generation_time_evidence_drift_llava/nltk_data'))
        with (OUT / 'logs/gpu1.log').open('w') as log:
            child = subprocess.Popen([sys.executable, __file__, '--worker'], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        shared.write(OUT / 'run_status.json', dict(status='generating', pid=os.getpid(), workers=[child.pid],
            queued=queued, started=started, fresh_images=plan['fresh_images'], cached_images=plan['reused_images']))
        assert child.wait() == 0, 'Sparse worker failed; preserve outputs and inspect log.'
        shared.write(OUT / 'run_status.json', dict(status='evaluating', pid=os.getpid(), queued=queued, started=started))
        shared.evaluate(OUT, plan)
        shared.write(OUT / 'run_status.json', dict(status='complete', queued=queued, started=started, finished=time.time(),
            fresh_images=plan['fresh_images'], cached_images=plan['reused_images']))
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait()
        shared.write(OUT / 'run_status.json', dict(status='failed', error=str(exc), queued=queued))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    worker() if parser.parse_args().worker else run()
