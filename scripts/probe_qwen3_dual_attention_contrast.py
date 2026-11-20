"""Fixed-prefix test of two signal-guided attention paths after full SADI."""

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

PREVIOUS = REPO/'outputs/qwen3_visual_ffn_preference_probe_20261120'
COMPLETE10 = REPO/'outputs/qwen3_live_visual_ffn_contrast10_20261120'


def prepare(root):
    from features.qwen3_dual_attention_contrast import PAIRS
    assert not root.exists()
    old = read(PREVIOUS/'protocol.json'); check(old)
    for name in ['pipeline_complete.json','analysis_complete.json','quality_complete.json','false_identity_audit.json']:
        assert read(COMPLETE10/name)['passed'], name
    paths = [Path(__file__), REPO/'features/qwen3_dual_attention_contrast.py',
             PREVIOUS/'protocol.json', PREVIOUS/'pipeline_complete.json',
             COMPLETE10/'protocol.json', COMPLETE10/'quality_audit.json', COMPLETE10/'false_identity_audit.json']
    paths += [PREVIOUS/f'{name}{s}{suffix}' for s in [0,1]
              for name,suffix in [('worker','.json'),('results','.json'),('ffn_outputs','.npz')]]
    plan = dict(old)
    plan['source_sha256'] = {**old['source_sha256'], **{str(p):sha(p) for p in paths}}
    plan.update(fixed_at=time.time(), previous=str(PREVIOUS), pairs=PAIRS,
        action='FullSADI5--18 unchanged. Five read-only tail20--35 views: SAR evidence Q, original hpre FFN preference P, visual-output preference P_uv, uniform, half-span rolled P. All eta=.5*HellingerSquared(Q,P_hpre). Per-head visual mass preserved.',
        decoding='2*positive_logits-negative_logits; same native plausibility beta.1 for every pair. SAR-only uses identical SAR view in both arguments. No native positive branch in the primary pair.',
        relation='Two post-SADI attention paths jointly compare target-conditioned visual writes and FFN preference, instead of changing only the negative path.',
        scope='Old six fixed development prefixes, no new captions or independent500. Complete10 of output-excess alone rejected as repeated-real denominator change.',
        sharing='Reuse one SAR path and all five views across six score pairs per position. Existing309 layer19 log_s cache, original6/4 B/P, native visualFFN hook only. No full vocabulary hidden arrays saved.',
        advance='Require defensible wrong-target response beyond matched SAR-only/uniform/rolled controls and inspect complete captions before100/500.')
    plan.pop('views',None)
    check(plan); root.mkdir(); (root/'source').mkdir()
    for p in [Path(__file__),REPO/'features/qwen3_dual_attention_contrast.py']:
        (root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',plan)


def verify(root):
    import torch
    from features.qwen3_dual_attention_contrast import dual_attention_logits
    generator = torch.Generator().manual_seed(43); checks=0
    for _ in range(25):
        native,positive,negative = [torch.randn(41,generator=generator) for _ in range(3)]
        allowed = native >= native.max()+math.log(.1)
        scores = dual_attention_logits(native,positive,negative)
        independent = torch.tensor([2*float(a)-float(b) for a,b in zip(positive,negative)])
        assert torch.equal(scores[allowed],independent[allowed])
        assert torch.isneginf(scores[~allowed]).all()
        same=dual_attention_logits(native,positive,positive)
        assert torch.equal(same[allowed],positive[allowed])
        zero=dual_attention_logits(native,positive,negative,alpha=0.)
        assert torch.equal(zero[allowed],positive[allowed])
        assert torch.equal(dual_attention_logits(native,native,native).argmax(),native.argmax())
        checks+=4
    # The plausibility mask belongs to native, not the steered positive path.
    native=torch.tensor([0.,-4.,-1.]);positive=torch.tensor([0.,20.,-1.])
    assert int(dual_attention_logits(native,positive,torch.zeros(3)).argmax())==0
    checks+=1
    write(root/'math_proof.json',dict(passed=True,checks=checks,cuda_initialized=torch.cuda.is_initialized()))


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_region_memory import RegionMemory
    from features.qwen3_visual_ffn_preference import VisualFFNPreferenceReader
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from features.qwen3_signal_preference_contrast import visual_preference
    from features.qwen3_dual_attention_contrast import PAIRS,dual_attention_logits
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan=read(root/'protocol.json');check(plan);assert read(root/'math_proof.json')['passed']
    assert {k:tuple(v)for k,v in plan['pairs'].items()}==PAIRS
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    span,inputs,_=batch(wrapper,group['ids'])
    archived={r['image_id']:r for s in [0,1] for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl')for r in g}
    responses=[archived[i]['response_token_ids']for i in group['ids']]
    image=group['cases'][0]['image_id'];row=group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz',allow_pickle=False)as saved:
        assert saved['batch_ids'].tolist()==group['ids']and saved['span'].tolist()==list(span)
        assert int(saved['prompt_length'])==inputs['input_ids'].shape[1]
        tokens=saved['token_ids'].tolist();log_s=torch.from_numpy(saved['log_s'][19].copy()).to(wrapper.device)
    reader=VisualFFNPreferenceReader(model,span,tokens,tracked_rows=[row]);reader.static[19,row]=dict(log_s=log_s)
    old={(r['image_id'],r['step']):r for r in read(PREVIOUS/f'results{shard}.json')}
    vision=[];encoder=[]
    handle=core.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1))
    capture=core.visual.register_forward_hook(lambda _m,_a,out:vision.append(out))
    prompt=inputs['input_ids'].shape[1];past=None;core.rope_deltas=None
    cases={c['step']:c for c in group['cases']};results=[];zero_checked=False;started=time.time()
    def cache_digest():return digest([t for layer in past.layers for t in [layer.keys,layer.values]])
    with torch.inference_mode(),Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture(),reader.capture():
        for step in range(max(cases)+1):
            if step==0:current=dict(inputs,cache_position=torch.arange(prompt,device=wrapper.device))
            else:
                previous=[r[step-1]if step<=len(r)else wrapper.tokenizer.eos_token_id for r in responses]
                current=dict(input_ids=torch.tensor(previous,device=wrapper.device)[:,None],
                    attention_mask=torch.ones((len(responses),prompt+step),device=wrapper.device,dtype=inputs['attention_mask'].dtype),
                    past_key_values=past,cache_position=torch.tensor([prompt+step-1],device=wrapper.device))
            output=core(**current,use_cache=True,return_dict=True);past=output.past_key_values
            assert reader.step==step and all(past.get_seq_length(i)==prompt+step for i in range(36))
            if step==0:
                assert len(vision)==len(encoder)==1;capture.remove()
                memory=RegionMemory(wrapper,inputs,span,vision.pop(),output.last_hidden_state[:,-1:])
                with np.load(PREVIOUS/f'ffn_outputs{shard}.npz',allow_pickle=False)as saved:
                    wanted=torch.from_numpy(saved['visual_update_bf16'].copy()).view(torch.bfloat16).to(wrapper.device)
                assert torch.equal(reader.visual_updates[19,row],wanted)
            if step not in cases:continue
            case=cases[step];token=case['token_id'];row=case['row']
            reference=output.last_hidden_state[:,-1:].clone();native=model.get_output_embeddings()(reference).float()[row,0]
            assert int(native.argmax())==token==responses[row][step]
            old_native=old[image,step]['trials'][0];assert old_native['positive']=='native'
            native_logp=float(native.log_softmax(0)[token]);assert abs(native_logp-old_native['target_positive_logp'])<1e-7
            fingerprint=cache_digest();q,p,uv,drift,uv_drift=reader.comparison_distributions(row,token)
            assert q.tolist()==old[image,step]['evidence']and p.tolist()==old[image,step]['preference_hpre']
            assert uv.tolist()==old[image,step]['preference']
            if not zero_checked:
                with visual_preference(span,row,q,0.):zero=memory.replay(row,past,reader,{},reference,zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[row,0],native)
                assert cache_digest()==fingerprint;zero_checked=True
            amount=.5*drift;priors=dict(sar=q,ffn=p,uv=uv,uniform=torch.ones_like(q)/len(q),rolled_ffn=p.roll(len(p)//2))
            views={};calls_by_view={}
            for name,prior in priors.items():
                with visual_preference(span,row,prior,amount)as calls:hidden=memory.replay(row,past,reader,{},reference)
                views[name]=model.get_output_embeddings()(hidden).float()[row,0]
                assert cache_digest()==fingerprint and calls==plan['action_layers'];calls_by_view[name]=calls
            eligible=(native>=native.max()+math.log(.1)).nonzero().flatten()
            record=dict(case,native_logp=native_logp,native_text=wrapper.tokenizer.decode([token]),
                evidence=q.tolist(),preference=p.tolist(),preference_uv=uv.tolist(),drift=drift,uv_drift=uv_drift,amount=amount,
                eligible_ids=eligible.tolist(),native_logits=native[eligible].tolist(),
                view_logits={n:v[eligible].tolist()for n,v in views.items()},action_layers_by_view=calls_by_view,trials=[])
            for policy,(positive,negative)in PAIRS.items():
                scores=dual_attention_logits(native,views[positive],views[negative]);chosen=int(scores.argmax())
                record['trials'].append(dict(policy=policy,positive=positive,negative=negative,new_token=chosen,
                    new_token_text=wrapper.tokenizer.decode([chosen]),changed=chosen!=token,
                    scores=scores[eligible].tolist(),all36_cache_unchanged=True))
            results.append(record);write(root/f'results{shard}.json',results)
            print(dict(image=image,step=step,native=record['native_text'],choices={t['policy']:t['new_token_text']for t in record['trials']}),flush=True)
    handle.remove();assert len(encoder)==1 and reader.head_calls==0 and len(results)==len(cases)
    assert reader.visual_capture_calls==1;check(plan)
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,encoder_calls=len(encoder),
        cached_semantic_head_calls=reader.head_calls,cases=len(results),zero_exact=zero_checked,all_trials_rolled_back=True,
        no_generation=True,visual_ffn_capture_calls=reader.visual_capture_calls,source_arrays_exact_previous=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root)
        elif args.verify:verify(args.root)
        elif args.worker is not None:worker(args.root,args.worker)
        else:parser.error('Select prepare, verify or worker.')
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
