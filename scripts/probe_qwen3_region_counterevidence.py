"""Boost previously verified regions using cached consistency measurements.

Same fixed six source prefixes and geometry; no crop encoding is repeated.
Current native target and log probability must exactly match the cached run.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read, write, sha


def signed_attention(module, query, key, value, attention_mask, scaling, span, row, rect, grid, strength):
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    key = qwen.repeat_kv(key, module.num_key_value_groups); value = qwen.repeat_kv(value, module.num_key_value_groups)
    scores = (query @ key.transpose(2, 3))*scaling
    if attention_mask is not None: scores = scores+attention_mask[..., :key.shape[-2]]
    start, end = span; height, width = grid; top, left, bottom, right = rect
    assert end-start == height*width and query.shape[-2] == 1 and strength < 0
    old = scores[row:row+1, :, :, start:end].float()
    penalty = torch.zeros((height, width), device=query.device, dtype=torch.float32)
    penalty[top:bottom, left:right] = float(strength)
    changed = old-penalty.flatten()
    changed += old.logsumexp(-1, keepdim=True)-changed.logsumexp(-1, keepdim=True)
    full = scores.float().clone(); full[row:row+1, :, :, start:end] = changed
    original_p = scores.float().softmax(-1); probability = full.softmax(-1)
    before, after = original_p[row, :, :, start:end].sum(-1), probability[row, :, :, start:end].sum(-1)
    error = float(((before-after).abs()/before.clamp_min(1e-12)).max()); assert error < 1e-5
    probability = probability.to(query.dtype)
    return (probability @ value).transpose(1, 2).contiguous(), probability, error


def prepare(root):
    assert not root.exists()
    previous = REPO / 'outputs/qwen3_region_consistency_probe_20261120'
    assert read(previous / 'pipeline_complete.json')['passed']
    plan = read(previous / 'protocol.json')
    for name, expected in plan['source_sha256'].items(): assert sha(REPO / name) == expected, name
    files = [Path(__file__), previous / 'protocol.json', previous / 'pipeline_complete.json',
             *[previous / f'{stem}{shard}.json' for stem in ['worker', 'results'] for shard in [0, 1]]]
    hashes = dict(plan['source_sha256']); hashes.update({str(p): sha(p) for p in files})
    root.mkdir(); (root / 'source').mkdir()
    (root / 'source' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    plan.update(fixed_at=time.time(), source_sha256=hashes, previous=str(previous),
        policies=['verified_boost', 'unverified_boost', 'shifted_boost'],
        action='Change only sign of the previous mass-preserving visual ROI operator: add5*D*r within the ROI; control adds5*D, shifted uses the same5*D*r outside. Native visual KV/values retained.',
        cache_reuse='Reuse exact previous rectangles, drift, alias consistency probabilities and cached detector scores; assert current native argmax/logp match before using them. No crop or fine-prefill recomputation.',
        reason='Earlier attenuation can increase the false target probability: the region then has a net counterevidence effect. This test measures the opposite attention direction without selecting a sign from GT.',
        no_new_captions=True, no_new_crop_encoding=True)
    write(root / 'protocol.json', plan); print(dict(prepared=True, cases=6), flush=True)


def worker(root, shard):
    import torch
    from features import qwen3_region_consistency as region
    from features.qwen3_region_memory import RegionEvidenceReader
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_signed_feedback import source_groups
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    plan = read(root / 'protocol.json')
    for name, expected in plan['source_sha256'].items(): assert sha(REPO / name) == expected, name
    group = plan['groups'][shard]
    cached = {(r['image_id'], r['step']): r for r in read(Path(plan['previous']) / f'results{shard}.json')}
    archived = {r['image_id']: r for s in [0, 1] for g in source_groups(Path(plan['source']) / f'coco100/sadi.shard{s}.jsonl') for r in g}
    wrapper = build_wrapper(); model = wrapper.model; core = model.model
    (span, items), = same_span_batches(wrapper, group['ids'], COCO_ROOT, len(group['ids']))
    assert [i for i, _ in items] == group['ids'] and span[1]-span[0] == group['visual_tokens']
    inputs = {key: torch.cat([item[key] for _, item in items], dim=0).to(wrapper.device) for key in items[0][1]}
    cases = {}
    for item in group['cases']: cases.setdefault(item['step'], []).append(item)
    reader = RegionEvidenceReader(model, span, sorted({r['token_id'] for r in group['cases']}))
    vision = []; encoder = [0]
    hook = core.visual.blocks[0].register_forward_hook(lambda *_: encoder.__setitem__(0, encoder[0]+1))
    capture = core.visual.register_forward_hook(lambda _m, _a, output: vision.append(output))
    responses = [archived[image]['response_token_ids'] for image in group['ids']]
    prompt = inputs['input_ids'].shape[1]; core.rope_deltas = None; past = None
    results = []; zero_checked = False; started = time.time(); replay_error = 0.
    def digest_cache(cache):
        h = hashlib.sha256()
        for layer in cache.layers:
            for t in [layer.keys, layer.values]: h.update(t.contiguous().view(torch.int16).cpu().numpy().tobytes())
        return h.hexdigest()
    def top(logits):
        ids = logits.topk(5).indices.tolist(); return dict(ids=ids, text=[wrapper.tokenizer.decode([t]) for t in ids])
    with torch.inference_mode(), Qwen3BatchLateAttention(model, span, frozenset(), 'sadi').capture(), reader.capture():
        for step in range(max(cases)+1):
            if step == 0: current = dict(inputs, cache_position=torch.arange(prompt, device=wrapper.device))
            else:
                previous = [r[step-1] if step <= len(r) else wrapper.tokenizer.eos_token_id for r in responses]
                current = dict(input_ids=torch.tensor(previous, device=wrapper.device)[:, None],
                    attention_mask=torch.ones((len(responses), prompt+step), device=wrapper.device, dtype=inputs['attention_mask'].dtype),
                    past_key_values=past, cache_position=torch.tensor([prompt+step-1], device=wrapper.device))
            output = core(**current, use_cache=True, return_dict=True); past = output.past_key_values
            assert reader.step == step
            if step == 0:
                assert len(vision) == 1; capture.remove()
                memory = region.RegionConsistency(wrapper, inputs, span, vision.pop(), output.last_hidden_state[:, -1:])
            if step not in cases: continue
            reference = output.last_hidden_state[:, -1:].clone(); logits = model.get_output_embeddings()(reference).float()[:, 0]
            for case in cases[step]:
                row, token = case['row'], case['token_id']; old = cached[case['image_id'], step]
                logp = float(logits[row].log_softmax(0)[token])
                replay_error = max(replay_error, abs(logp-old['native_logp']))
                assert int(logits[row].argmax()) == token and abs(logp-old['native_logp']) < 1e-7
                before_digest = digest_cache(past) if not zero_checked else None
                if not zero_checked:
                    hidden, _ = memory.attenuate(row, past, reader, (0, 0, 3, 3), 0., reference, zero=True)
                    assert torch.equal(model.get_output_embeddings()(hidden).float()[:, 0], logits)
                record = dict(case, native=top(logits[row]), native_logp=logp, trials=[])
                for locator in plan['locators']:
                    evidence, = [t for t in old['trials'] if t['locator'] == locator and t['policy'] == 'verified']
                    shifted, = [t for t in old['trials'] if t['locator'] == locator and t['policy'] == 'shifted']
                    for policy in plan['policies']:
                        strength = 5*evidence['drift']*(1. if policy == 'unverified_boost' else evidence['inspection']['reliability_deficit'])
                        rect = shifted['applied_rect'] if policy == 'shifted_boost' else evidence['rect']
                        with patch.object(region, 'mask_attention', signed_attention):
                            hidden, error = memory.attenuate(row, past, reader, rect, -strength, reference)
                        proposed = model.get_output_embeddings()(hidden).float()[row, 0]
                        after = float(proposed.log_softmax(0)[token]); changed = int(proposed.argmax()) != token
                        record['trials'].append(dict(locator=locator, policy=policy, strength=strength,
                            rect=rect, drift=evidence['drift'], inspection=evidence['inspection'], top5=top(proposed),
                            original_logp=after, original_logp_delta=after-logp, token_changed=changed,
                            would_accept=changed and after < logp, visual_mass_relative_error_fp32=error))
                if not zero_checked: assert digest_cache(past) == before_digest; zero_checked = True
                results.append(record)
                print(dict(image=case['image_id'], step=step, native=record['native']['text'][0],
                    proposals=[(t['locator'], t['policy'], t['top5']['text'][0], t['original_logp_delta']) for t in record['trials']]), flush=True)
    hook.remove(); assert encoder[0] == 1 and len(results) == len(group['cases'])
    for name, expected in plan['source_sha256'].items(): assert sha(REPO / name) == expected, name
    write(root / f'results{shard}.json', results)
    write(root / f'worker{shard}.json', dict(passed=True, shard=shard, cases=len(results), original_encoder_calls=1,
        crop_encoder_calls=0, fine_prefill_calls=0, cached_native_score_replay_max_error=replay_error,
        zero_and_full36_cache_rollback_checked=zero_checked, seconds=time.time()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), all_trials_rolled_back=True, no_new_captions=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        elif args.worker is not None: worker(args.root, args.worker)
        else: parser.error('Choose prepare or worker.')
    except Exception:
        if args.root.exists():
            target = args.root / (f'failure.worker{args.worker}.json' if args.worker is not None else 'failure.json')
            if not target.exists(): write(target, dict(failed_at=time.time(), traceback=traceback.format_exc()))
        raise
