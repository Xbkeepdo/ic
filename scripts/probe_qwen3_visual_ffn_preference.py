"""Compare visual FFN-output preference against hpre preference on fixed prefixes."""

import argparse
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.probe_qwen3_signal_erasure_contrast import CACHE, check

PREVIOUS = REPO/'outputs/qwen3_signal_erasure_contrast_probe_20261120'
VIEWS = ['uv_excess', 'hpre_excess', 'uniform', 'rolled_uv_excess']
DISAGREEMENT = REPO/'outputs/qwen3_signal_disagreement_contrast_probe_20261120'
FULL100 = REPO/'outputs/qwen3_live_preference_contrast100_20261120'


def prepare(root):
    assert not root.exists()
    old = read(PREVIOUS/'protocol.json'); check(old)
    assert read(PREVIOUS/'pipeline_complete.json')['passed']
    for name in ['pipeline_complete.json','analysis_complete.json','independent_summary.json','quality_complete.json']:
        assert read(FULL100/name)['passed'], name
    check(read(FULL100/'protocol.json'))
    assert read(DISAGREEMENT/'pipeline_complete.json')['passed']
    assert read(DISAGREEMENT/'independent_disagreement_audit.json')['passed']
    paths = [Path(__file__), REPO/'features/qwen3_signal_preference_contrast.py',
        REPO/'scripts/probe_qwen3_signal_erasure_contrast.py', PREVIOUS/'protocol.json',
        PREVIOUS/'pipeline_complete.json', *[PREVIOUS/f'results{s}.json' for s in [0, 1]]]
    paths += [REPO/'features/qwen3_signal_disagreement_contrast.py', FULL100/'protocol.json', FULL100/'analysis_complete.json', FULL100/'quality_complete.json']
    paths += [REPO/'features/qwen3_visual_ffn_preference.py',DISAGREEMENT/'protocol.json',DISAGREEMENT/'pipeline_complete.json',DISAGREEMENT/'independent_disagreement_audit.json']
    plan = dict(old); plan['source_sha256'] = {**old['source_sha256'], **{str(p): sha(p) for p in paths}}
    plan.update(fixed_at=time.time(), previous=str(PREVIOUS), views=VIEWS,
        selection='No patch removal. New Normalize(relu(P_uv-Q)), old Normalize(relu(P_hpre-Q)), uniform and rolled new excess; all same eta.',
        action='Read original source19 visual MLP outputs only during native prefill. P_uv=softmax(cos(u_visual_j,u_query)/.03). Common eta=.5*HellingerSquared(Q,P_uv); visual mass preserved, text attention unchanged.',
        decoding='Native SADI positive versus four negative priors; alpha1/beta.1. Same new D_uv strength for all to isolate representation, not coefficient changes.',
        relation='Compare query and image FFN updates from the same original MLP, with unchanged target-conditioned SAR evidence. No FFN activation edits or added FFN forward. FullSADI5--18 preserved, negative attention tail20--35 only.',
        scope='Same six old diagnostic positions. Neither raw-P nor excess(P_hpre-Q) showed unique false corrections over matched uniform. Inspect a different preference representation; no new captions, independent test or500 use.',
        advance='Only a defensible change beyond matched uniform/rolled motivates complete10 then full100. An immediate token flip is not mitigation evidence.')
    plan.pop('fraction'); plan.pop('policies'); check(plan)
    root.mkdir(); (root/'source').mkdir()
    for p in [Path(__file__), REPO/'features/qwen3_signal_preference_contrast.py', REPO/'features/qwen3_visual_ffn_preference.py']:
        (root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json', plan)


def verify(root):
    import torch
    from features.qwen3_signal_preference_contrast import mix_visual
    from features.qwen3_signal_erasure_contrast import contrast_logits
    from features.qwen3_signal_disagreement_contrast import coupled_priors
    from features.qwen3_visual_ffn_preference import ffn_output_preference, output_priors
    generator = torch.Generator().manual_seed(43); checks = 0
    for length in [11, 32, 192, 300]:
        probability = torch.randn((3, 4, 1, length+8), generator=generator).softmax(-1)
        prior = torch.rand(length, generator=generator); prior /= prior.sum()
        for row in range(3):
            for amount in [0., .125, .5]:
                got = mix_visual(probability, (3, length+3), row, prior, amount)
                independent = probability.clone()
                for head in range(4):
                    mass = sum(float(probability[row, head, 0, j]) for j in range(3, length+3))
                    for j in range(length):
                        independent[row, head, 0, j+3] = (1-amount)*probability[row, head, 0, j+3]+amount*mass*prior[j]
                assert torch.allclose(got, independent, rtol=1e-6, atol=1e-8)
                assert torch.allclose(got.sum(-1), probability.sum(-1), rtol=1e-6, atol=1e-7)
                assert torch.equal(got[[i for i in range(3) if i != row]], probability[[i for i in range(3) if i != row]])
                assert torch.equal(got[..., :3], probability[..., :3]) and torch.equal(got[..., length+3:], probability[..., length+3:])
                checks += 1
    for _ in range(20):
        a = torch.randn(29, generator=generator); b = torch.randn(29, generator=generator)
        value = contrast_logits(a, b)
        allowed = a >= a.max()+__import__('math').log(.1)
        assert torch.equal(value[allowed], (2*a-b)[allowed]) and torch.isneginf(value[~allowed]).all()
        checks += 1
    for length in [11,32,192,300]:
        q=torch.rand(length,generator=generator).softmax(0);p=torch.rand(length,generator=generator).softmax(0)
        views=coupled_priors(q,p)
        positive=[max(float(b)-float(a),0.) for a,b in zip(q,p)]
        independent=torch.tensor([v/sum(positive) for v in positive])
        assert torch.allclose(views['excess'],independent,rtol=1e-6,atol=1e-8)
        assert torch.equal(views['deficit'],coupled_priors(p,q)['excess'])
        assert torch.equal(views['rolled_excess'].roll(-(length//2)),views['excess'])
        for v in views.values(): assert torch.isfinite(v).all() and v.min()>=0 and abs(float(v.sum())-1)<1e-6
        equal=coupled_priors(q,q);assert all(torch.equal(v,torch.ones_like(q)/len(q)) for v in equal.values())
        checks+=5
    for length in [11,32,192,300]:
        visual=torch.randn((length,64),generator=generator);query=torch.randn(64,generator=generator)
        value=ffn_output_preference(visual,query)
        v,u=visual.double(),query.double();cosine=(v*u).sum(-1)/(v.norm(dim=-1)*u.norm()).clamp_min(1e-16)
        reference=(cosine/.03).softmax(0).float()
        assert torch.allclose(value,reference,rtol=1e-5,atol=1e-6)
        assert torch.allclose(ffn_output_preference(visual*3,query*7),value,rtol=1e-5,atol=1e-6)
        assert torch.equal(ffn_output_preference(visual,torch.zeros_like(query)),torch.ones_like(value)/length)
        q=torch.rand(length,generator=generator).softmax(0);old=torch.rand(length,generator=generator).softmax(0)
        priors=output_priors(q,old,value)
        assert torch.equal(priors['uv_excess'],coupled_priors(q,value)['excess'])
        assert torch.equal(priors['hpre_excess'],coupled_priors(q,old)['excess'])
        assert torch.equal(priors['rolled_uv_excess'].roll(-(length//2)),priors['uv_excess'])
        checks+=6
    write(root/'math_proof.json', dict(passed=True, checks=checks, cuda_initialized=torch.cuda.is_initialized(),cpu_fp64_tolerance=1e-6))


def worker(root, shard):
    import numpy as np
    import torch
    from features.qwen3_region_memory import RegionMemory
    from features.qwen3_visual_ffn_preference import VisualFFNPreferenceReader,output_priors
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from features.qwen3_signal_preference_contrast import visual_preference
    from features.qwen3_signal_erasure_contrast import contrast_logits
    from features.qwen3_signal_disagreement_contrast import coupled_priors
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
    reader = VisualFFNPreferenceReader(model,span,tokens,tracked_rows=[row]); reader.static[19,row]=dict(log_s=log_s)
    assert tuple(tokens) == reader.tokens
    vision = []; encoder = []
    handle = core.visual.blocks[0].register_forward_hook(lambda *_: encoder.append(1))
    capture = core.visual.register_forward_hook(lambda _m, _a, out: vision.append(out))
    prompt = inputs['input_ids'].shape[1]; past = None; core.rope_deltas = None
    cases = {c['step']: c for c in group['cases']}; results=[];queries=[];zero_checked=False;started=time.time()
    def cache_digest(): return digest([t for layer in past.layers for t in [layer.keys, layer.values]])
    def top(scores):
        ids = scores.topk(5).indices.tolist()
        return dict(ids=ids, text=[wrapper.tokenizer.decode([i]) for i in ids])
    old = {(r['image_id'], r['step']): r for r in read(PREVIOUS/f'results{shard}.json')}
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
            case = cases[step]; token = case['token_id']; row = case['row']
            reference = output.last_hidden_state[:, -1:].clone(); logits = model.get_output_embeddings()(reference).float()[:, 0]
            assert int(logits[row].argmax()) == token == responses[row][step]
            assert abs(float(logits[row].log_softmax(0)[token])-old[image, step]['native_logp']) < 1e-7
            fingerprint=cache_digest();q,p_hpre,p,drift_hpre,drift=reader.comparison_distributions(row,token)
            queries.append(reader.current[19]['update'][row].detach().clone())
            if not zero_checked:
                with visual_preference(span, row, q, 0.): zero = memory.replay(row, past, reader, {}, reference, zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[:, 0], logits)
                assert cache_digest() == fingerprint; zero_checked = True
            amount = .5*drift
            priors = output_priors(q,p_hpre,p)
            views = dict(native=logits[row]); calls_by_view = {}
            for name, prior in priors.items():
                with visual_preference(span, row, prior, amount) as calls:
                    hidden = memory.replay(row, past, reader, {}, reference)
                views[name] = model.get_output_embeddings()(hidden).float()[row, 0]
                assert cache_digest() == fingerprint; calls_by_view[name] = calls
            pairs = [('native',name) for name in VIEWS]
            record = dict(case, native=top(logits[row]), drift=drift, amount=amount,
                evidence=q.tolist(), preference=p.tolist(), preference_hpre=p_hpre.tolist(),drift_hpre=drift_hpre,action_layers_by_view=calls_by_view,
                views={n:top(v) for n,v in views.items()}, trials=[])
            for positive, negative in pairs:
                allowed = views[positive] >= views[positive].max()+__import__('math').log(plan['beta'])
                candidates = allowed.nonzero().flatten()
                scores = contrast_logits(views[positive], views[negative], plan['alpha'], plan['beta'])
                record['trials'].append(dict(positive=positive, negative=negative, contrast=top(scores),
                    new_token=int(scores.argmax()), new_token_text=wrapper.tokenizer.decode([int(scores.argmax())]), changed=int(scores.argmax()) != token,
                    eligible_ids=candidates.tolist(), positive_logits=views[positive][candidates].tolist(),
                    negative_logits=views[negative][candidates].tolist(), target_positive_logp=float(views[positive].log_softmax(0)[token]),
                    target_negative_logp=float(views[negative].log_softmax(0)[token]), all36_cache_unchanged=True))
            results.append(record); write(root/f'results{shard}.json', results)
            print(dict(image=image, step=step, native=record['native']['text'][0],
                views={n:v['text'][0] for n,v in record['views'].items()},
                contrast=[(t['positive'],t['negative'],t['contrast']['text'][0]) for t in record['trials']]), flush=True)
    handle.remove();assert len(encoder)==1 and reader.head_calls==0 and len(results)==len(cases)
    assert reader.visual_capture_calls==1 and len(reader.visual_updates)==1
    visual=reader.visual_updates[19,row].contiguous();query=torch.stack(queries).contiguous()
    assert visual.dtype==query.dtype==torch.bfloat16
    path=root/f'ffn_outputs{shard}.npz'
    np.savez_compressed(path,visual_update_bf16=visual.view(torch.uint16).cpu().numpy(),query_update_bf16=query.view(torch.uint16).cpu().numpy(),
        batch_ids=np.asarray(group['ids']),span=np.asarray(span),prompt=prompt,image_id=image,
        steps=np.asarray([r['step'] for r in results]),tokens=np.asarray([r['token_id'] for r in results]),source_layer=19,dtype='bfloat16')
    v=visual.double().cpu().numpy();u=query.double().cpu().numpy()
    numerator=u@v.T;denominator=np.linalg.norm(u,axis=1)[:,None]*np.linalg.norm(v,axis=1)[None,:]
    logits=numerator/np.maximum(denominator,1e-16)/.03;logits-=logits.max(1,keepdims=True)
    independent=np.exp(logits);independent/=independent.sum(1,keepdims=True)
    actual=np.asarray([r['preference'] for r in results]);error=float(np.max(np.abs(independent-actual)))
    assert error<1e-6,error
    write(root/f'output_preference_audit{shard}.json',dict(passed=True,max_absolute_error=error,reference='Independent numpy FP64 from saved native BF16 FFN outputs',cache_sha256=sha(path)))
    check(plan)
    write(root/f'worker{shard}.json', dict(passed=True, seconds=time.time()-started, encoder_calls=len(encoder),
        cached_semantic_head_calls=reader.head_calls, cases=len(results), zero_exact=zero_checked,
        all_trials_rolled_back=True,no_generation=True,visual_ffn_capture_calls=reader.visual_capture_calls,
        visual_ffn_bytes=visual.numel()*visual.element_size(),query_ffn_bytes=query.numel()*query.element_size(),
        saved_cache_bytes=path.stat().st_size,cache_sha256=sha(path),output_preference_max_error=error,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--verify', action='store_true')
    parser.add_argument('--worker', type=int, choices=[0,1]); args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        elif args.verify: verify(args.root)
        elif args.worker is not None: worker(args.root, args.worker)
        else: parser.error('Select a stage.')
    except BaseException:
        if args.root.exists(): write(args.root/f'failure{args.worker}.json', dict(traceback=traceback.format_exc(), failed_at=time.time()))
        raise
