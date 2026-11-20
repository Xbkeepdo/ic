"""Test four critical-patch erasure policies on six frozen native prefixes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha

OLD = REPO/'outputs/qwen3_region_consistency_probe_20261120'
CACHE = REPO/'outputs/qwen3_visual_semantics_cache_20261120'
POLICIES = ['attention', 'sar', 'sar_ems', 'shifted']


def check(plan):
    for name, wanted in plan['source_sha256'].items():
        assert sha(REPO/name) == wanted, name


def prepare(root):
    assert not root.exists(), 'Inspect prior state; never overwrite.'
    old, cached = read(OLD/'protocol.json'), read(CACHE/'protocol.json')
    assert read(OLD/'pipeline_complete.json')['passed'] and read(CACHE/'pipeline_complete.json')['passed']
    # Cache already proves layer19 selected probabilities and 540 original E_l cells.
    hashes = {**old['source_sha256'], **cached['source_sha256']}
    files = [Path(__file__), REPO/'features/qwen3_signal_erasure_contrast.py',
             REPO/'features/qwen3_region_memory.py', REPO/'features/qwen3_online_risk_f.py',
             REPO/'features/qwen3_bounded_visual_tilt.py', REPO/'scripts/run_qwen3_semantic_final_pilot.py',
             OLD/'protocol.json', OLD/'pipeline_complete.json', CACHE/'protocol.json', CACHE/'pipeline_complete.json']
    files += [CACHE/f'image{image}.npz' for image in [243882, 353411]]
    for shard in [0, 1]:
        proof = read(CACHE/f'worker{shard}.json'); assert proof['passed']
        files.append(CACHE/f'worker{shard}.json')
        for record in proof['records']:
            assert sha(CACHE/f'image{record["image_id"]}.npz') == record['sha256']
    hashes.update({str(p): sha(p) for p in files})
    plan = dict(fixed_at=time.time(), source=old['source'], source_sha256=hashes, groups=old['groups'],
        policies=POLICIES, base_layers=list(range(5, 19)), source_layer=19,
        action_layers=list(range(20, 36)), fraction=.1, alpha=1., beta=.1,
        selection='Top ceil(.1*M) positions by attention19 sum, Q_SAR19, relu(Q19-P_FFN19), or half-span shifted deficit.',
        action='Set selected visual logits to -inf in negative tail20--35; full softmax renormalizes. No visual-mass correction.',
        decoding='z=(1+alpha)*native-alpha*negative; eligibility native>=max(native)+log(beta). Alpha1/beta.1 fixed before results.',
        evidence='Original native BF16 per-patch write norms and final-norm hpre log_s; P=cosine(visual,actualFFN19)/.03 softmax.',
        cache='Reuse exact existing layer19 selected309 log_s, same original B/P; no new visual head projections or crop encoders.',
        rollback='Negative branch read-only: restore current20--35 KV, visual slots, and assert full36 cache digest unchanged.',
        scope='Six previously inspected development prefixes. No new captions, no quality or generalization claim.',
        relation='ASCD-inspired critical-visual erasure; uses ordinary full SADI as positive path, fixed source19 selection, tail20--35 only. Not author baseline replication.',
        reference='https://arxiv.org/html/2506.14766v1',
        advance='Inspect physical false/real candidate effects; only a defensible mechanism advances to complete100. Never tune on reserved500.')
    check(plan); root.mkdir(); (root/'source').mkdir()
    for p in [Path(__file__), REPO/'features/qwen3_signal_erasure_contrast.py']:
        (root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json', plan)


def verify(root):
    import torch
    from features.qwen3_signal_erasure_contrast import select_positions, contrast_logits
    generator = torch.Generator().manual_seed(43); checks = 0
    for length in [11, 32, 192, 300]:
        attention = torch.rand((4, length), generator=generator)
        q = torch.rand(length, generator=generator).softmax(0)
        p = torch.rand(length, generator=generator).softmax(0)
        for policy in POLICIES:
            got = select_positions(attention, q, p, policy)
            assert len(got) == __import__('math').ceil(.1*length) and len(set(got.tolist())) == len(got)
            checks += 1
        original = select_positions(attention, q, p, 'sar_ems')
        shifted = select_positions(attention, q, p, 'shifted')
        assert torch.equal(shifted, (original+length//2) % length)
    for _ in range(20):
        native = torch.randn((3, 29), generator=generator); negative = torch.randn((3, 29), generator=generator)
        result = contrast_logits(native, negative)
        independent = 2*native-negative
        independent[native < native.max(-1, keepdim=True).values+__import__('math').log(.1)] = -torch.inf
        assert torch.equal(result, independent)
        assert torch.equal(contrast_logits(native, negative, alpha=0), native)
        assert torch.equal(contrast_logits(native, native).argmax(-1), native.argmax(-1))
        checks += 3
    write(root/'math_proof.json', dict(passed=True, checks=checks, cuda_initialized=torch.cuda.is_initialized()))


def worker(root, shard):
    import numpy as np
    import torch
    from features.qwen3_region_memory import RegionMemory, RegionEvidenceReader
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from features.qwen3_signal_erasure_contrast import select_positions, contrast_logits, erase_visual
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch, digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan = read(root/'protocol.json'); check(plan); assert read(root/'math_proof.json')['passed']
    group = plan['groups'][shard]; wrapper = build_wrapper(); model = wrapper.model; core = model.model
    span, inputs, _ = batch(wrapper, group['ids'])
    archived = {r['image_id']: r for s in [0, 1] for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl') for r in g}
    responses = [archived[i]['response_token_ids'] for i in group['ids']]
    image = group['cases'][0]['image_id']; row = group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz', allow_pickle=False) as saved:
        assert saved['batch_ids'].tolist() == group['ids'] and saved['span'].tolist() == list(span)
        assert int(saved['prompt_length']) == inputs['input_ids'].shape[1]
        tokens = saved['token_ids'].tolist(); log_s = torch.from_numpy(saved['log_s'][19].copy()).to(wrapper.device)
    reader = RegionEvidenceReader(model, span, tokens); reader.static[19, row] = dict(log_s=log_s)
    vision = []; encoder = []
    handle = core.visual.blocks[0].register_forward_hook(lambda *_: encoder.append(1))
    capture = core.visual.register_forward_hook(lambda _m, _a, out: vision.append(out))
    prompt = inputs['input_ids'].shape[1]; past = None; core.rope_deltas = None
    cases = {c['step']: c for c in group['cases']}; results = []; zero_checked = False; started = time.time()
    def cache_digest(): return digest([t for layer in past.layers for t in [layer.keys, layer.values]])
    def top(scores):
        ids = scores.topk(5).indices.tolist()
        return dict(ids=ids, text=[wrapper.tokenizer.decode([i]) for i in ids])
    with torch.inference_mode(), Qwen3BatchLateAttention(model, span, frozenset(), 'sadi').capture(), reader.capture():
        for step in range(max(cases)+1):
            if step == 0:
                current = dict(inputs, cache_position=torch.arange(prompt, device=wrapper.device))
            else:
                previous = [r[step-1] if step <= len(r) else wrapper.tokenizer.eos_token_id for r in responses]
                current = dict(input_ids=torch.tensor(previous, device=wrapper.device)[:, None],
                    attention_mask=torch.ones((len(responses), prompt+step), device=wrapper.device, dtype=inputs['attention_mask'].dtype),
                    past_key_values=past, cache_position=torch.tensor([prompt+step-1], device=wrapper.device))
            output = core(**current, use_cache=True, return_dict=True); past = output.past_key_values
            assert reader.step == step and all(past.get_seq_length(i) == prompt+step for i in range(36))
            if step == 0:
                assert len(vision) == len(encoder) == 1; capture.remove()
                memory = RegionMemory(wrapper, inputs, span, vision.pop(), output.last_hidden_state[:, -1:])
            if step not in cases: continue
            case = cases[step]; token = case['token_id']; row = case['row']
            reference = output.last_hidden_state[:, -1:].clone(); logits = model.get_output_embeddings()(reference).float()[:, 0]
            assert int(logits[row].argmax()) == token == responses[row][step]
            fingerprint = cache_digest()
            if not zero_checked:
                with erase_visual(span, row, torch.empty(0, device=wrapper.device, dtype=torch.long)):
                    zero = memory.replay(row, past, reader, {}, reference, zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[:, 0], logits)
                assert cache_digest() == fingerprint; zero_checked = True
            q, p, drift = reader.distributions(row, token)
            record = dict(case, native=top(logits[row]), native_logp=float(logits[row].log_softmax(0)[token]), drift=drift, trials=[])
            for policy in POLICIES:
                selected = select_positions(reader.current[19]['attention'][row], q, p, policy, plan['fraction'])
                with erase_visual(span, row, selected) as calls:
                    hidden = memory.replay(row, past, reader, {}, reference)
                negative = model.get_output_embeddings()(hidden).float()[row, 0]
                contrasted = contrast_logits(logits[row], negative, plan['alpha'], plan['beta'])
                assert cache_digest() == fingerprint and calls == plan['action_layers']
                chosen = int(contrasted.argmax())
                record['trials'].append(dict(policy=policy, selected_positions=selected.tolist(),
                    negative=top(negative), contrast=top(contrasted), new_token=chosen,
                    native_target_erasure_logp_change=float(negative.log_softmax(0)[token])-record['native_logp'],
                    target_contrast_margin=float(contrasted[token]-contrasted.max()),
                    changed=chosen != token, all36_cache_unchanged=True, action_layers=calls))
            results.append(record); write(root/f'results{shard}.json', results)
            print(dict(image=image, step=step, native=record['native']['text'][0],
                contrast=[(t['policy'], t['contrast']['text'][0]) for t in record['trials']]), flush=True)
    handle.remove(); assert len(encoder) == 1 and reader.head_calls == 0 and len(results) == len(cases)
    check(plan)
    write(root/f'worker{shard}.json', dict(passed=True, seconds=time.time()-started, encoder_calls=len(encoder),
        cached_semantic_head_calls=reader.head_calls, cases=len(results), zero_exact=zero_checked,
        all_trials_rolled_back=True, no_generation=True, peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--verify', action='store_true'); parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        elif args.verify: verify(args.root)
        elif args.worker is not None: worker(args.root, args.worker)
        else: parser.error('Select prepare, verify or worker.')
    except BaseException:
        if args.root.exists():
            write(args.root/f'failure{args.worker}.json', dict(traceback=traceback.format_exc(), failed_at=time.time()))
        raise
