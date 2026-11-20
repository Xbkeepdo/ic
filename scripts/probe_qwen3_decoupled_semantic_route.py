"""Test source19 -> next20 routing after full SADI, with matched controls."""

import argparse
import math
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.probe_qwen3_signal_erasure_contrast import CACHE, check

PREVIOUS = REPO/'outputs/qwen3_relative_evidence_probe_20261120'
OLD = REPO/'outputs/qwen3_post_sadi_ffn_attenuation_probe_20261120'
DUAL = REPO/'outputs/qwen3_dual_attention_contrast_probe_20261120'
PRIORS = REPO/'outputs/qwen3_early_semantics_analysis32_20261120/background_priors.npz'
POLICIES = ['early3', 'weighted3', 'late19', 'shifted3', 'uniform_donor']
SCOPES = dict(next20=[20], fixed_tail20_35=list(range(20, 36)))


def prepare(root):
    assert not root.exists()
    old = read(PREVIOUS/'protocol.json'); check(old)
    assert read(PREVIOUS/'pipeline_complete.json')['passed']
    files = [Path(__file__), REPO/'features/qwen3_decoupled_semantic_route.py', PRIORS,
        PREVIOUS/'protocol.json', PREVIOUS/'head_class_neighbors.json',
        REPO/'features/qwen3_region_memory.py', REPO/'features/qwen3_post_sadi_ffn_attenuation.py']
    for shard in [0, 1]: files += [OLD/f'results{shard}.json', DUAL/f'results{shard}.json']
    plan = dict(old)
    plan['source_sha256'] = {**old['source_sha256'], **{str(p): sha(p) for p in files}}
    plan.update(fixed_at=time.time(), policies=POLICIES, scopes=SCOPES,
        action='Keep complete SADI5--18. Read source19 Q/P/D and positive target FFN gain. Destination uses early3 semantic rival difference without current I; reference19 and I-weighted3 controls.',
        semantic='Nearest single class prototype from fixed output-head graph. rho_r > max(1,rho_y), G19_y>0. Priors from disjoint27 equal-image cache; no GT in action.',
        destination='Normalize(relu(log1p(m_r/mu_r)-log1p(m_y/mu_y))). Main layer3; fixed controls weighted3, layer19, half-span roll.',
        movement='eta=min(.5,5*D*tanh(G/2)), donor rate=P19/(P19+Qdest3). Remove eta*rate*A_h, redistribute removed per-head mass over destination. Controls keep the same donor reference; uniform donor keeps removed mass per-head.',
        matching='Next20 controls see identical native A, so removed masses match across policies exactly. In full-tail controls A changes after earlier edits; matching is conditional on each own input A, not identical across trajectories.',
        scope='Primary19->20 only. Secondary same fixed19 budget20--35, not refreshed. Zero cases reuse native without tail. No FFN/logit editing.',
        limitations='Layer choice and utensil case known in advance. Mechanism probe only, not new caption/CHAIR/500 evidence.',
        checks='CPU mass/conservation/outside-span/row/matched-donor tests; native Q/P/G/logp match; full36 cache before/after each case, current36 each proposal; native zero hidden/logits/KV exact.')
    check(plan); root.mkdir(); (root/'source').mkdir()
    for p in [Path(__file__), REPO/'features/qwen3_decoupled_semantic_route.py']:
        (root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json', plan)


def verify(root):
    import torch
    from features.qwen3_decoupled_semantic_route import (
        movement_fraction, semantic_destination, move_visual_probability)
    rng = torch.Generator().manual_seed(43); checks = 0
    for length in [11, 32, 300]:
        mass = torch.rand((length, 3), generator=rng)
        prior = torch.rand(3, generator=rng)
        dest = semantic_destination(mass, prior, 0, 1)
        expected = (torch.log1p(mass[:, 1]/prior[1])-torch.log1p(mass[:, 0]/prior[0])).clamp_min(0)
        expected /= expected.sum()
        assert torch.allclose(dest, expected); checks += 1
        pref = torch.rand(length, generator=rng); pref /= pref.sum()
        native = torch.randn((3, 4, 1, length+8), generator=rng).softmax(-1)
        span = (3, 3+length)
        changed, moved = move_visual_probability(native, span, 1, dest, pref, dest, .4)
        assert torch.isfinite(changed).all() and (changed >= 0).all(); checks += 1
        assert torch.allclose(changed.sum(-1), native.sum(-1), rtol=1e-6, atol=1e-7); checks += 1
        assert torch.equal(changed[[0, 2]], native[[0, 2]]); checks += 1
        assert torch.equal(changed[..., :3], native[..., :3]) and torch.equal(changed[..., 3+length:], native[..., 3+length:]); checks += 1
        for alternative in [dest.roll(length//2), torch.ones_like(dest)/length]:
            trial, new_moved = move_visual_probability(native, span, 1, alternative, pref, dest, .4)
            assert new_moved == moved and torch.allclose(trial.sum(-1), native.sum(-1), atol=1e-7); checks += 1
        uniform, new_moved = move_visual_probability(native, span, 1, dest, pref, dest, .4, True)
        assert new_moved == moved and torch.allclose(uniform.sum(-1), native.sum(-1), atol=1e-7); checks += 1
        zero, empty = move_visual_probability(native, span, 1, dest, pref, dest, 0.)
        assert zero is native and not empty; checks += 1
    assert movement_fraction(.4, -1) == movement_fraction(0., 1) == 0.
    assert movement_fraction(1, 100) == .5
    checks += 2
    write(root/'math_proof.json', dict(passed=True, checks=checks, cuda_initialized=torch.cuda.is_initialized()))


def worker(root, shard):
    import numpy as np
    import torch
    from features.qwen3_decoupled_semantic_route import movement_fraction, semantic_destination, semantic_route
    from features.qwen3_post_sadi_ffn_attenuation import FFNResidualReader
    from features.qwen3_region_memory import RegionMemory
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch, digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan = read(root/'protocol.json'); check(plan); assert read(root/'math_proof.json')['passed']
    group = plan['groups'][shard]; wrapper = build_wrapper(); model = wrapper.model; core = model.model
    span, inputs, _ = batch(wrapper, group['ids'])
    archived = {r['image_id']: r for s in [0, 1] for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl') for r in g}
    responses = [archived[i]['response_token_ids'] for i in group['ids']]
    image = group['cases'][0]['image_id']; row = group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz', allow_pickle=False) as d:
        assert d['batch_ids'].tolist() == group['ids'] and d['span'].tolist() == list(span)
        tokens = d['token_ids'].tolist(); mapping = d['token_classes'].tolist(); names = d['class_names'].tolist()
        logs = {l: d['log_s'][l].copy() for l in [3, 19]}
    with np.load(PRIORS, allow_pickle=False) as d:
        assert d['token_ids'].tolist() == tokens and d['class_names'].tolist() == names
        priors = {l: torch.from_numpy(d['class_prior'][j].copy()).to(wrapper.device) for j, l in enumerate(d['layers'].tolist())}
    masses = {}
    for l, values in logs.items():
        scaled = np.expm1(values.astype(np.float64))
        classes = np.stack([scaled[:, np.array(mapping) == c].sum(1) for c in names], axis=1)
        masses[l] = torch.from_numpy(classes).to(wrapper.device)
    graph = read(PREVIOUS/'head_class_neighbors.json')
    assert graph['classes'] == names and graph['tokens'] == tokens
    reader = FFNResidualReader(model, span, tokens)
    reader.static[19, row] = dict(log_s=torch.from_numpy(logs[19]).to(wrapper.device))
    old = {(r['image_id'], r['step']): r for r in read(OLD/f'results{shard}.json')}
    dual = {(r['image_id'], r['step']): r for r in read(DUAL/f'results{shard}.json')}
    vision = []; encoder = []
    handle = core.visual.blocks[0].register_forward_hook(lambda *_: encoder.append(1))
    capture = core.visual.register_forward_hook(lambda _m, _a, out: vision.append(out))
    prompt = inputs['input_ids'].shape[1]; past = None; core.rope_deltas = None
    cases = {c['step']: c for c in group['cases']}; results = []; zero_checked = False; replays = 0
    started = time.time()
    def fingerprint(full=True):
        return digest([t if full else t[:, :, -1:] for layer in past.layers for t in [layer.keys, layer.values]])
    with torch.inference_mode(), Qwen3BatchLateAttention(model, span, frozenset(), 'sadi').capture(), reader.capture():
        for step in range(max(cases)+1):
            if step == 0: current = dict(inputs, cache_position=torch.arange(prompt, device=wrapper.device))
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
            case = cases[step]; token = case['token_id']; row = case['row']; reference = output.last_hidden_state[:, -1:].clone()
            all_native = model.get_output_embeddings()(reference).float()[:, 0]; native = all_native[row]
            assert int(native.argmax()) == token == responses[row][step]
            assert abs(float(native.log_softmax(0)[token])-old[image, step]['native_logp']) < 1e-7
            q, p, drift = reader.distributions(row, token)
            assert q.tolist() == dual[image, step]['evidence'] and p.tolist() == dual[image, step]['preference']
            residual = reader.ffn_residual[row, 0]; update = reader.current[19]['update'][row]
            gain = float(model.get_output_embeddings().weight[token].float()@(reader.decoder.norm(residual+update).float()-reader.decoder.norm(residual).float()))
            assert gain == old[image, step]['native_ffn_target_logit_gain']
            target = mapping[tokens.index(token)]; rival = graph['neighbors'][target][0]['name']
            y = names.index(target); r = names.index(rival)
            rho = masses[3].mean(0)/priors[3]
            eligible = float(rho[r]) > max(1., float(rho[y])) and gain > 0
            amount = movement_fraction(drift, gain) if eligible else 0.
            main = semantic_destination(masses[3], priors[3], y, r).float()
            source_weight = q/(1+reader.static[19, row]['log_s'][:, tokens.index(token)])
            source_weight /= source_weight.sum()
            destinations = dict(early3=main, weighted3=semantic_destination(masses[3], priors[3], y, r, source_weight.double()).float(),
                late19=semantic_destination(masses[19], priors[19], y, r).float(), shifted3=main.roll(len(main)//2), uniform_donor=main)
            full_before = fingerprint(); current_before = fingerprint(False)
            if not zero_checked:
                with semantic_route(span, row, main, p, main, 0., [20]):
                    zero = memory.replay(row, past, reader, {}, reference, zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[:, 0], all_native)
                assert fingerprint() == full_before; zero_checked = True
            top = native.topk(8).indices.tolist()
            rival_tokens = [t for t, c in zip(tokens, mapping) if c == rival]
            record = dict(case, target=target, rival=rival, target_rho=float(rho[y]), rival_rho=float(rho[r]),
                gain=gain, drift=drift, amount=amount, semantic_eligible=eligible, native_logp=float(native.log_softmax(0)[token]),
                native_top=[dict(id=t, text=wrapper.tokenizer.decode([t]), logit=float(native[t])) for t in top],
                native_rival_logits={str(t): float(native[t]) for t in rival_tokens}, trials=[])
            matched = None
            for scope, layers in SCOPES.items():
                for policy in POLICIES:
                    if amount:
                        with semantic_route(span, row, destinations[policy], p, main, amount, layers, policy == 'uniform_donor') as calls:
                            hidden = memory.replay(row, past, reader, {}, reference)
                        logits = model.get_output_embeddings()(hidden).float()[row, 0]; replays += 1
                        assert fingerprint(False) == current_before
                        if scope == 'next20':
                            removed = calls[0]['removed_mass_per_head']
                            if matched is None: matched = removed
                            else: assert removed == matched
                    else: logits = native; calls = []
                    chosen = int(logits.argmax())
                    record['trials'].append(dict(scope=scope, policy=policy, applied=amount > 0, new_token=chosen,
                        new_token_text=wrapper.tokenizer.decode([chosen]), target_logp=float(logits.log_softmax(0)[token]),
                        rival_logits={str(t): float(logits[t]) for t in rival_tokens}, action_calls=calls,
                        current36_cache_restored=True))
            assert fingerprint() == full_before
            record['full36_cache_unchanged_after_all_trials'] = True
            results.append(record); write(root/f'results{shard}.json', results)
            print(dict(image=image, step=step, target=target, rival=rival, amount=amount,
                choices={t['scope']+'/'+t['policy']: t['new_token_text'] for t in record['trials']}), flush=True)
    handle.remove(); assert len(encoder) == 1 and reader.head_calls == 0 and len(results) == len(cases); check(plan)
    write(root/f'worker{shard}.json', dict(passed=True, seconds=time.time()-started, encoder_calls=len(encoder),
        cases=len(results), new_semantic_head_calls=reader.head_calls, zero_exact=zero_checked,
        nonzero_tail_replays=replays, no_generation=True, all_cache_checks=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--verify', action='store_true')
    parser.add_argument('--worker', type=int, choices=[0, 1]); args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        elif args.verify: verify(args.root)
        elif args.worker is not None: worker(args.root, args.worker)
        else: parser.error('Select stage.')
    except BaseException:
        if args.root.exists(): write(args.root/f'failure{args.worker}.json', dict(traceback=traceback.format_exc(), failed_at=time.time()))
        raise
