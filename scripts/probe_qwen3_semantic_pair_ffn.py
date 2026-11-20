"""Add a controlled pairwise FFN auxiliary to semantic crop plus attention routing."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.probe_qwen3_decoupled_semantic_route import (
    CACHE, OLD, DUAL, PRIORS, SCOPES, check, read, write, sha)

PARENT = REPO/'outputs/qwen3_semantic_crop_route_probe_20261120'
GRAPH = REPO/'outputs/qwen3_relative_evidence_probe_20261120/head_class_neighbors.json'
POLICIES = ['baseline', 'pair', 'rotated', 'uniform', 'opposite']
SCOPES = dict(fixed_tail20_35=list(range(20, 36)))


def prepare(root):
    assert not root.exists()
    plan = dict(read(PARENT/'protocol.json')); check(plan)
    assert read(PARENT/'pipeline_complete.json')['passed']
    paths = [Path(__file__), REPO/'scripts/probe_qwen3_decoupled_semantic_route.py',
        PARENT/'protocol.json', PARENT/'pipeline_complete.json', GRAPH,
        REPO/'features/qwen3_pair_ffn_auxiliary.py',
        *[PARENT/f'results{s}.json' for s in [0,1]]]
    plan['source_sha256'].update({str(p): sha(p) for p in paths})
    plan.update(fixed_at=time.time(), policies=POLICIES,
        action='Keep frozen crop+attention route: fullSADI5--18, source19, fixed-tail20--35. Auxiliary removes eta*relu(u dot d)*d from source19 FFN update, d=unit(W_y-mean W_rival). Require positive native final-norm pair gain as well. Same eta as attention, no sweep.',
        controls='Baseline is frozen correct crop+route. Pair, half-dimension rotated direction, uniform FFN attenuation, and opposite direction all have the same intended FP32 removed norm. BF16 actual norms recorded separately. No GT in rectangle, rival or gate.',
        scope='Only fixed-tail20--35; native source19 budget not refreshed. Correct crop encoded/reconstructed once, reused by all5 branches. Baseline logits and native words must exactly match prior probe.',
        geometry='Crop peak uses semantic destination alone; previous failed region methods multiplied by current I and targeted cups/fries. Original B/P, rotary positions, history KV and current target fixed.',
        advance='Only mechanism check on already-known six positions. No automatic full-generation claim or500 advancement from word alone.')
    check(plan); root.mkdir(); (root/'source').mkdir(); (root/'crops').mkdir()
    (root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json', plan)
    (root/'source/qwen3_pair_ffn_auxiliary.py').write_bytes((REPO/'features/qwen3_pair_ffn_auxiliary.py').read_bytes())


def worker(root, shard):
    import numpy as np
    import torch
    from features.qwen3_decoupled_semantic_route import movement_fraction, semantic_destination, semantic_route
    from features.qwen3_post_sadi_ffn_attenuation import FFNResidualReader
    from features.qwen3_pair_ffn_auxiliary import pair_ffn_anchor
    from features.qwen3_region_memory import RegionMemory, rectangle
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch, digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan = read(root/'protocol.json'); check(plan)
    assert read(root/'math_proof.json')['passed']
    group = plan['groups'][shard]; wrapper = build_wrapper(); model = wrapper.model; core = model.model
    span, inputs, photos = batch(wrapper, group['ids'])
    archived = {r['image_id']: r for s in [0, 1] for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl') for r in g}
    responses = [archived[i]['response_token_ids'] for i in group['ids']]
    image = group['cases'][0]['image_id']; row = group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz', allow_pickle=False) as d:
        assert d['batch_ids'].tolist() == group['ids'] and d['span'].tolist() == list(span)
        tokens = d['token_ids'].tolist(); mapping = d['token_classes'].tolist(); names = d['class_names'].tolist()
        early = d['log_s'][3].astype(np.float64); late = d['log_s'][19].copy()
    with np.load(PRIORS, allow_pickle=False) as d:
        assert d['token_ids'].tolist() == tokens and d['class_names'].tolist() == names
        prior = torch.from_numpy(d['class_prior'][d['layers'].tolist().index(3)].copy()).to(wrapper.device)
    scaled = np.expm1(early)
    mass = torch.from_numpy(np.stack([scaled[:, np.array(mapping) == c].sum(1) for c in names], axis=1)).to(wrapper.device)
    graph = read(GRAPH); reader = FFNResidualReader(model, span, tokens)
    reader.static[19, row] = dict(log_s=torch.from_numpy(late).to(wrapper.device))
    old = {(r['image_id'], r['step']): r for r in read(OLD/f'results{shard}.json')}
    dual = {(r['image_id'], r['step']): r for r in read(DUAL/f'results{shard}.json')}
    previous = {(r['image_id'],r['step']):r for r in read(PARENT/f'results{shard}.json')}
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
            y = names.index(target); r = names.index(rival); rho = mass.mean(0)/prior
            eligible = float(rho[r]) > max(1., float(rho[y])) and gain > 0
            amount = movement_fraction(drift, gain) if eligible else 0.
            dest = semantic_destination(mass, prior, y, r).float()
            before = fingerprint(); current_before = fingerprint(False)
            if not zero_checked:
                zero = memory.replay(row, past, reader, {}, reference, zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[:, 0], all_native)
                assert fingerprint() == before; zero_checked = True
            resources = {}
            if amount:
                for location in ['evidence']:
                    rect = rectangle(dest, *memory.grid(row), location)
                    path = root/'crops'/f'{image}_{step}_{location}.jpg'
                    with reader.paused(): crop = memory.encode_crop(photos[image], image, row, rect, path)
                    refined = memory.keys(row, past, reader, crop=crop, rect=rect, amount=drift)
                    delta = max(float((keys-past.layers[l].keys[row:row+1, :, span[0]:span[1]]).abs().max()) for l, (keys, _) in refined.items())
                    assert delta > 0 and fingerprint() == before
                    resources[location] = dict(rect=rect, pixel_box=crop['pixel_box'], crop_file=str(path),
                        grid=memory.grid(row), crop_grid=crop['grid'], refined=refined, max_visual_key_difference=delta)
            rivals = [t for t, c in zip(tokens, mapping) if c == rival]
            record = dict(case, target=target, rival=rival, amount=amount, drift=drift, gain=gain,
                native_logp=float(native.log_softmax(0)[token]),
                native_rival_logits={str(t): float(native[t]) for t in rivals}, trials=[])
            for scope, layers in SCOPES.items():
                for policy in POLICIES:
                    if amount:
                        resource = resources['evidence']
                        refined = {l: resource['refined'][l] for l in layers}
                        rival_weight = model.get_output_embeddings().weight[rivals].float().mean(0)
                        direction = model.get_output_embeddings().weight[token].float()-rival_weight
                        pair_gain = float(direction@(reader.decoder.norm(residual+update).float()-reader.decoder.norm(residual).float()))
                        auxiliary_amount = amount if pair_gain > 0 else 0.
                        original_anchor = reader.anchor
                        reader.anchor, auxiliary = pair_ffn_anchor(original_anchor, reader.ffn_residual,
                            reader.current[19]['update'], row, direction, auxiliary_amount, policy)
                        try:
                            with semantic_route(span, row, dest, p, dest, amount, layers) as calls:
                                hidden = memory.replay(row, past, reader, refined, reference)
                        finally: reader.anchor = original_anchor
                        auxiliary.update(native_pair_final_norm_gain=pair_gain, amount=auxiliary_amount)
                        logits = model.get_output_embeddings()(hidden).float()[row, 0]; replays += 1
                        assert fingerprint(False) == current_before
                        extra = {k: v for k, v in resource.items() if k != 'refined'}
                        extra['auxiliary'] = auxiliary
                        if policy == 'baseline':
                            old_trial, = [t for t in previous[image,step]['trials'] if t['scope']==scope and t['policy']=='crop_and_route']
                            assert float(logits.log_softmax(0)[token]) == old_trial['target_logp']
                            assert int(logits.argmax()) == old_trial['new_token']
                            assert {str(t):float(logits[t]) for t in rivals} == old_trial['rival_logits']
                    else: logits = native; calls = []; extra = {}
                    chosen = int(logits.argmax()); top = logits.topk(5).indices.tolist()
                    record['trials'].append(dict(scope=scope, policy=policy, applied=amount > 0, new_token=chosen,
                        new_token_text=wrapper.tokenizer.decode([chosen]), target_logp=float(logits.log_softmax(0)[token]),
                        rival_logits={str(t): float(logits[t]) for t in rivals}, action_calls=calls,
                        top5=[dict(id=t, text=wrapper.tokenizer.decode([t]), logit=float(logits[t])) for t in top], **extra))
            assert fingerprint() == before
            record['full36_cache_unchanged_after_all_trials'] = True
            results.append(record); write(root/f'results{shard}.json', results)
            print(dict(image=image, step=step, target=target, rival=rival, amount=amount,
                choices={t['scope']+'/'+t['policy']: t['new_token_text'] for t in record['trials']}), flush=True)
    handle.remove(); assert len(encoder) == 1+memory.crop_encoder_calls and reader.head_calls == 0; check(plan)
    assert len(results) == len(cases)
    write(root/f'worker{shard}.json', dict(passed=True, seconds=time.time()-started, cases=len(results),
        original_encoder_calls=1, crop_encoder_calls=memory.crop_encoder_calls, actual_encoder_calls=len(encoder),
        cached_vision_prefill_reconstructions=memory.reconstruction_calls, new_semantic_head_calls=reader.head_calls,
        zero_exact=zero_checked, nonzero_tail_replays=replays, no_generation=True, all_cache_checks=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))



def verify(root):
    import torch
    from features.qwen3_pair_ffn_auxiliary import pair_ffn_anchor
    rng=torch.Generator().manual_seed(43);checks=0
    for dtype in [torch.float32,torch.bfloat16]:
        update=torch.randn((3,64),generator=rng).to(dtype)
        residual=torch.randn((3,1,64),generator=rng).to(dtype);anchor=residual+update[:,None]
        direction=update[1].float()+.1*torch.randn(64,generator=rng)
        outputs={}
        for policy in POLICIES:
            value,detail=pair_ffn_anchor(anchor,residual,update,1,direction,.4,policy)
            assert torch.equal(value[[0,2]],anchor[[0,2]]);checks+=1
            outputs[policy]=detail
            zero,_=pair_ffn_anchor(anchor,residual,update,1,direction,0.,policy)
            assert zero is anchor;checks+=1
        assert outputs['pair']['changed_pair_projection']<outputs['pair']['original_pair_projection'];checks+=1
        assert outputs['opposite']['changed_pair_projection']>outputs['opposite']['original_pair_projection'];checks+=1
        lengths=[outputs[p]['intended_removed_norm']for p in POLICIES if p!='baseline']
        assert max(lengths)==min(lengths);checks+=1
        if dtype==torch.float32:
            assert max(abs(outputs[p]['actual_update_difference_norm']-lengths[0])for p in POLICIES if p!='baseline')<1e-6;checks+=1
    write(root/'math_proof.json',dict(passed=True,checks=checks,cuda_initialized=torch.cuda.is_initialized()))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--verify', action='store_true'); p.add_argument('--worker', type=int, choices=[0, 1]); a = p.parse_args()
    try:
        if a.prepare: prepare(a.root)
        elif a.verify: verify(a.root)
        elif a.worker is not None: worker(a.root, a.worker)
        else: p.error('Select stage.')
    except BaseException:
        if a.root.exists(): write(a.root/f'failure{a.worker}.json', dict(traceback=traceback.format_exc(), failed_at=time.time()))
        raise
