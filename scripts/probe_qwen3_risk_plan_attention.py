"""Move visual attention with the original K32/tau.03/w1 risk coupling."""

import argparse
import math
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.probe_qwen3_signal_erasure_contrast import CACHE,check

PREVIOUS=REPO/'outputs/qwen3_post_sadi_ffn_attenuation_probe_20261120'


def prepare(root):
    assert not root.exists()
    old=read(PREVIOUS/'protocol.json');check(old)
    assert read(PREVIOUS/'pipeline_complete.json')['passed']and read(PREVIOUS/'independent_attenuation_audit.json')['passed']
    paths=[Path(__file__),REPO/'features/qwen3_risk_plan_attention.py',REPO/'features/dgst_t.py',
        REPO/'features/qwen3_online_risk_f.py',PREVIOUS/'protocol.json',PREVIOUS/'pipeline_complete.json',
        PREVIOUS/'independent_attenuation_audit.json',*[PREVIOUS/f'results{s}.json'for s in [0,1]]]
    plan=dict(old);plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p)for p in paths}}
    plan.pop('ffn_constant',None)
    plan.update(fixed_at=time.time(),previous=str(PREVIOUS),modes=['plan','independent','reverse'],
        K=32,tau=.03,cost_coefficient=1.,endpoint_penalty_coefficient=1.,
        action='FullSADI5--18 unchanged. Source19 original risk P=cos(FFN,hpre)/.03 softmax; target=Normalize(mean attention*log_s). Original top16+16 union, cosine cost, exact balanced EMD plan. Reallocate only selected visual attention through the row-normalized plan at20--35.',
        strengths='All three kernels use eta=.5*risk/(1+risk), where risk=transport+both endpoint penalties, all coefficients1. No Hellinger proxy or added1 semantic factor.',
        controls='Independent coupling a*b has same source/target marginals but no transport geometry; reverse uses transposed plan conditioned by rows. Same support,amount,base and tail range.',
        decoding='Direct proposal logits, same native beta.1 plausibility set; also save unmasked actual greedy diagnosis. All proposal currentKV restored; no new captions.',
        scope='Old six diagnostic positions only. Original exact-risk scalar checked against original reader once per GPU. Plan/marginals/cost saved for independent LP verification.',
        relation='Use an intermediate coupling from original risk as a head-specific attention transition, rather than scalar risk gate or Hellinger overlap. Endpoint terms are constant for fixed balanced marginals; affect action magnitude, not optimal plan.',
        advance='Unique useful wrong-target response beyond independent/reverse plus real-target check can motivate complete10; no hallucination mitigation claim from instantaneous token change.')
    check(plan);root.mkdir();(root/'source').mkdir()
    for p in [Path(__file__),REPO/'features/qwen3_risk_plan_attention.py']:(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',plan)


def verify(root):
    import torch
    from features.dgst_t import _wasserstein_1_exact
    from features.qwen3_risk_plan_attention import row_kernel,move_attention
    a=torch.tensor([.75,.25]);b=torch.tensor([.25,.75]);cost=torch.tensor([[0.,1.],[1.,0.]])
    value,plan=_wasserstein_1_exact(a,b,cost,solver='emd')
    assert abs(value-.5)<1e-12 and torch.allclose(plan,torch.tensor([[.25,.5],[0.,.25]],dtype=torch.float64),atol=1e-12)
    forward=row_kernel(plan);reverse=row_kernel(plan.T)
    assert torch.allclose(a.double()@forward,b.double(),atol=1e-12)
    assert torch.allclose(b.double()@reverse,a.double(),atol=1e-12)
    zero=row_kernel(torch.zeros((2,2),dtype=torch.float64));assert torch.equal(zero,torch.eye(2,dtype=torch.float64))
    checks=4;generator=torch.Generator().manual_seed(43)
    for row in [0,1]:
        probability=torch.randn((2,3,1,9),generator=generator).softmax(-1);support=torch.tensor([1,4])
        kernel=forward.float()
        for amount in [0.,.125,.5]:
            got=move_attention(probability,(2,7),row,support,kernel,amount)
            other=1-row;assert torch.equal(got[other],probability[other])
            unchanged=[j for j in range(9)if j not in [3,6]]
            assert torch.equal(got[...,unchanged],probability[...,unchanged])
            assert torch.allclose(got.sum(-1),probability.sum(-1),atol=1e-7)
            for head in range(3):
                old=probability[row,head,0,[3,6]]
                independent=torch.tensor([(1-amount)*float(old[j])+amount*sum(float(old[i])*float(kernel[i,j])for i in range(2))for j in range(2)])
                assert torch.allclose(got[row,head,0,[3,6]],independent,atol=1e-8,rtol=1e-6)
            if amount==0:assert got is probability
            checks+=1
    write(root/'math_proof.json',dict(passed=True,checks=checks,cuda_initialized=torch.cuda.is_initialized()))


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_region_memory import RegionMemory,RegionEvidenceReader
    from features.qwen3_risk_plan_attention import original_risk_plan,plan_attention
    from features.qwen3_online_risk_f import exact_risk
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan=read(root/'protocol.json');check(plan);assert read(root/'math_proof.json')['passed']
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    span,inputs,_=batch(wrapper,group['ids'])
    archived={r['image_id']:r for s in [0,1]for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl')for r in g}
    responses=[archived[i]['response_token_ids']for i in group['ids']]
    image=group['cases'][0]['image_id'];row=group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz',allow_pickle=False)as saved:
        assert saved['batch_ids'].tolist()==group['ids']and saved['span'].tolist()==list(span)
        assert int(saved['prompt_length'])==inputs['input_ids'].shape[1]
        tokens=saved['token_ids'].tolist();log_s=torch.from_numpy(saved['log_s'][19].copy()).to(wrapper.device)
    reader=RegionEvidenceReader(model,span,tokens);reader.static[19,row]=dict(log_s=log_s)
    old={(r['image_id'],r['step']):r for r in read(PREVIOUS/f'results{shard}.json')}
    vision=[];encoder=[]
    handle=core.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1))
    capture=core.visual.register_forward_hook(lambda _m,_a,out:vision.append(out))
    prompt=inputs['input_ids'].shape[1];past=None;core.rope_deltas=None
    cases={c['step']:c for c in group['cases']};results=[];zero_checked=False;started=time.time();scalar_error=None
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
            if step not in cases:continue
            case=cases[step];token=case['token_id'];row=case['row']
            reference=output.last_hidden_state[:,-1:].clone();native=model.get_output_embeddings()(reference).float()[row,0]
            assert int(native.argmax())==token==responses[row][step]
            assert abs(float(native.log_softmax(0)[token])-old[image,step]['native_logp'])<1e-7
            fingerprint=cache_digest();state=reader.current[19]
            args=(reader.visual[19][row],state['hpre'][row],state['update'][row],state['attention'][row],log_s[:,reader.columns[token]])
            support,kernels,problem=original_risk_plan(*args)
            if not zero_checked:
                original,terms,_,_=exact_risk(*args);scalar_error=abs(original-problem['risk']);assert scalar_error<1e-7
                with plan_attention(span,row,support,kernels['plan'],0.):zero=memory.replay(row,past,reader,{},reference,zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[row,0],native)
                assert cache_digest()==fingerprint;zero_checked=True
            eligible=(native>=native.max()+math.log(.1)).nonzero().flatten()
            record=dict(case,native_logp=float(native.log_softmax(0)[token]),native_text=wrapper.tokenizer.decode([token]),
                support=support.tolist(),problem=problem,eligible_ids=eligible.tolist(),native_logits=native[eligible].tolist(),trials=[])
            for name,kernel in kernels.items():
                with plan_attention(span,row,support,kernel,problem['amount'])as calls:hidden=memory.replay(row,past,reader,{},reference)
                assert cache_digest()==fingerprint and calls==plan['action_layers']
                logits=model.get_output_embeddings()(hidden).float()[row,0];values=logits[eligible]
                chosen=int(eligible[values.argmax()]);unmasked=int(logits.argmax())
                record['trials'].append(dict(policy=name,new_token=chosen,new_token_text=wrapper.tokenizer.decode([chosen]),
                    unmasked_token=unmasked,unmasked_token_text=wrapper.tokenizer.decode([unmasked]),changed=chosen!=token,
                    eligible_logits=values.tolist(),target_logp=float(logits.log_softmax(0)[token]),
                    all36_cache_unchanged=True,action_layers=calls))
            results.append(record);write(root/f'results{shard}.json',results)
            print(dict(image=image,step=step,native=record['native_text'],choices={t['policy']:t['new_token_text']for t in record['trials']}),flush=True)
    handle.remove();assert len(encoder)==1 and reader.head_calls==0 and len(results)==len(cases);check(plan)
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,encoder_calls=len(encoder),
        cached_semantic_head_calls=reader.head_calls,cases=len(results),zero_exact=zero_checked,all_trials_rolled_back=True,
        no_generation=True,original_risk_max_absolute_error=scalar_error,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root)
        elif args.verify:verify(args.root)
        elif args.worker is not None:worker(args.root,args.worker)
        else:parser.error('Select prepare,verify or worker.')
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
