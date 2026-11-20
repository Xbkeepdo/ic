"""Compare relative and absolute signed budgets at adjacent or full-tail scope."""

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
DUAL=REPO/'outputs/qwen3_dual_attention_contrast_probe_20261120'
POLICIES=['relative','absolute','shifted','opposite']
SCOPES=dict(next20=[20],tail20_35=list(range(20,36)))


def prepare(root):
    assert not root.exists();old=read(PREVIOUS/'protocol.json');check(old)
    assert read(PREVIOUS/'pipeline_complete.json')['passed']and read(DUAL/'pipeline_complete.json')['passed']
    paths=[Path(__file__),REPO/'features/qwen3_relative_evidence_feedback.py',
        REPO/'features/qwen3_post_sadi_ffn_attenuation.py',PREVIOUS/'protocol.json',PREVIOUS/'pipeline_complete.json',
        DUAL/'protocol.json',*[PREVIOUS/f'results{s}.json'for s in [0,1]],*[DUAL/f'results{s}.json'for s in [0,1]]]
    plan=dict(old);plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p)for p in paths}}
    plan.pop('ffn_constant',None)
    plan.update(fixed_at=time.time(),modes=POLICIES,scopes=SCOPES,strength=5.,
        action='Full SADI5--18 unchanged. Source19 native final target y; Q=Normalize(native write_norm*(1+log_s)), P=cos(visual_hpre,FFNupdate)/.03 softmax. Positive source19 local FFN target-logit gain required.',
        budget='5*D*tanh(G/2)*f/span(f) ifG>0 else0; D=HellingerSquared(Q,P). Relative f=(Q-P)/(Q+P), absolute f=Q-P, shifted=half-span rolled relative, opposite=negative relative. Same peak-to-peak amplitude when span>1e-6.',
        precision='All policies apply visual-logsumexp correction in FP32, then BF16 attention@V. Original model remains BF16/eager; no FFN edits.',
        scope='Primary source19 -> next20 only. Secondary fixed-source19 budget at20--35, not refreshed per layer. Six old fixed positions, no new captions.',
        decoding='Primary actual unmasked greedy; save native beta.1 masked choice as secondary. No contrastive-logit formula.',
        checks='Original native target/logp/G/Q/P matched old caches. Zero replay exact36KV/hidden/logits. Every proposal known visual/current writes restored by frozen RegionMemory; current36 hashes per proposal, full36 before/after each case.',
        reuse='NegativeG/zero budget reuses native logits without extra tail/model calls. Old309 source19 semantic matrices reused, encoder1/group.',
        extra_cache='GPU0 saves cosine neighbors among62 class prototypes from mean unit-normalized output-head rows of309 lexical forms, before prefix replay. Diagnostic cache only, not used in current action or candidate selection.',
        advance='Inspect all scopes/policies/real and wrong targets; only defensible behavior beyond matched controls motivates complete-caption verification. No mitigation claim from token alone.')
    check(plan);root.mkdir();(root/'source').mkdir()
    for p in [Path(__file__),REPO/'features/qwen3_relative_evidence_feedback.py']:(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',plan)


def verify(root):
    import torch
    from features.qwen3_relative_evidence_feedback import evidence_budget,shift_visual_scores
    generator=torch.Generator().manual_seed(43);checks=0
    for length in [11,32,300]:
        q=torch.rand(length,generator=generator);q/=q.sum();p=torch.rand(length,generator=generator);p/=p.sum()
        reference=(q.double()-p.double())/(q.double()+p.double())
        value=evidence_budget(q,p,.4,1.2,'relative')
        expected=5*.4*math.tanh(.6)*reference/(reference.max()-reference.min())
        assert torch.allclose(value.double(),expected,rtol=1e-5,atol=1e-7)
        amplitude=5*.4*math.tanh(.6)
        budgets={n:evidence_budget(q,p,.4,1.2,n)for n in POLICIES}
        for b in budgets.values():assert abs(float(b.max()-b.min())-amplitude)<1e-6
        assert torch.equal(budgets['shifted'].roll(-(length//2)),value)
        assert torch.equal(budgets['opposite'],-value)
        assert not torch.count_nonzero(evidence_budget(q,p,.4,-.1,'relative'))
        assert not torch.count_nonzero(evidence_budget(q,q,.4,1.2,'relative'))
        scores=torch.randn((3,4,1,length+8),generator=generator)
        changed=shift_visual_scores(scores,(3,3+length),1,value)
        mask=torch.ones(length+8,dtype=torch.bool);mask[3:3+length]=False
        assert torch.equal(changed[[0,2]],scores[[0,2]])and torch.equal(changed[...,mask],scores[...,mask])
        assert torch.allclose(changed.softmax(-1)[1,:,:,3:3+length].sum(-1),scores.softmax(-1)[1,:,:,3:3+length].sum(-1),rtol=1e-6,atol=1e-7)
        assert shift_visual_scores(scores,(3,3+length),1,torch.zeros_like(value))is scores
        checks+=10
    write(root/'math_proof.json',dict(passed=True,checks=checks,cuda_initialized=torch.cuda.is_initialized()))


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_region_memory import RegionMemory
    from features.qwen3_post_sadi_ffn_attenuation import FFNResidualReader
    from features.qwen3_relative_evidence_feedback import evidence_budget,evidence_attention
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan=read(root/'protocol.json');check(plan);assert read(root/'math_proof.json')['passed']
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    span,inputs,_=batch(wrapper,group['ids'])
    archived={r['image_id']:r for s in [0,1]for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl')for r in g}
    responses=[archived[i]['response_token_ids']for i in group['ids']];image=group['cases'][0]['image_id'];row=group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz',allow_pickle=False)as d:
        assert d['batch_ids'].tolist()==group['ids']and d['span'].tolist()==list(span)
        assert int(d['prompt_length'])==inputs['input_ids'].shape[1]
        tokens=d['token_ids'].tolist();log_s=torch.from_numpy(d['log_s'][19].copy()).to(wrapper.device)
        class_names=d['class_names'].tolist();token_classes=d['token_classes'].tolist()
    if shard==0:
        from torch.nn import functional as F
        with torch.inference_mode():
            weights=F.normalize(model.get_output_embeddings().weight[tokens].float(),dim=-1)
            prototypes=F.normalize(torch.stack([weights[[i for i,c in enumerate(token_classes)if c==name]].mean(0)for name in class_names]),dim=-1)
            similarity=(prototypes@prototypes.T).cpu().numpy()
        neighbors={}
        for index,name in enumerate(class_names):
            order=[i for i in np.argsort(-similarity[index],kind='stable')if i!=index][:5]
            neighbors[name]=[dict(name=class_names[i],cosine=float(similarity[index,i]))for i in order]
        write(root/'head_class_neighbors.json',dict(passed=True,classes=class_names,tokens=tokens,token_classes=token_classes,
            similarity=similarity.tolist(),neighbors=neighbors,used_in_current_action=False,
            definition='Mean of unit-normalized output-head word-form vectors within each class, then normalized prototype cosine.'))
    reader=FFNResidualReader(model,span,tokens);reader.static[19,row]=dict(log_s=log_s)
    old={(r['image_id'],r['step']):r for r in read(PREVIOUS/f'results{shard}.json')}
    dual={(r['image_id'],r['step']):r for r in read(DUAL/f'results{shard}.json')}
    vision=[];encoder=[];handle=core.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1))
    capture=core.visual.register_forward_hook(lambda _m,_a,out:vision.append(out))
    prompt=inputs['input_ids'].shape[1];past=None;core.rope_deltas=None
    cases={c['step']:c for c in group['cases']};results=[];zero_checked=False;started=time.time();replays=0
    def fingerprint(full=True):return digest([t if full else t[:,:,-1:]for layer in past.layers for t in [layer.keys,layer.values]])
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
            case=cases[step];token=case['token_id'];row=case['row'];reference=output.last_hidden_state[:,-1:].clone()
            all_native=model.get_output_embeddings()(reference).float()[:,0];native=all_native[row]
            assert int(native.argmax())==token==responses[row][step]
            assert abs(float(native.log_softmax(0)[token])-old[image,step]['native_logp'])<1e-7
            q,p,drift=reader.distributions(row,token)
            assert q.tolist()==dual[image,step]['evidence']and p.tolist()==dual[image,step]['preference']
            residual=reader.ffn_residual[row,0];update=reader.current[19]['update'][row]
            assert torch.equal(reader.ffn_residual+reader.current[19]['update'][:,None],reader.anchor)
            gain=float(model.get_output_embeddings().weight[token].float()@(reader.decoder.norm(residual+update).float()-reader.decoder.norm(residual).float()))
            assert gain==old[image,step]['native_ffn_target_logit_gain']
            full_before=fingerprint();current_before=fingerprint(False)
            if not zero_checked:
                with evidence_attention(span,row,torch.zeros_like(q),[20]):zero=memory.replay(row,past,reader,{},reference,zero=True)
                assert torch.equal(model.get_output_embeddings()(zero).float()[:,0],all_native)
                assert fingerprint()==full_before;zero_checked=True
            eligible=(native>=native.max()+math.log(.1)).nonzero().flatten()
            record=dict(case,native_text=wrapper.tokenizer.decode([token]),native_logp=float(native.log_softmax(0)[token]),
                gain=gain,drift=drift,evidence=q.tolist(),preference=p.tolist(),eligible_ids=eligible.tolist(),native_logits=native[eligible].tolist(),trials=[])
            for scope,layers in SCOPES.items():
                for policy in POLICIES:
                    budget=evidence_budget(q,p,drift,gain,policy);active=bool(torch.count_nonzero(budget))
                    if active:
                        with evidence_attention(span,row,budget,layers)as(calls,errors):hidden=memory.replay(row,past,reader,{},reference)
                        logits=model.get_output_embeddings()(hidden).float()[row,0];replays+=1
                        assert fingerprint(False)==current_before
                    else:logits=native;calls=[];errors=[]
                    chosen=int(logits.argmax());masked=int(eligible[logits[eligible].argmax()])
                    record['trials'].append(dict(scope=scope,policy=policy,applied=active,new_token=chosen,new_token_text=wrapper.tokenizer.decode([chosen]),
                        masked_token=masked,masked_token_text=wrapper.tokenizer.decode([masked]),
                        eligible_logits=logits[eligible].tolist(),target_logp=float(logits.log_softmax(0)[token]),
                        budget_min=float(budget.min()),budget_max=float(budget.max()),budget_peak_to_peak=float(budget.max()-budget.min()),
                        action_calls=calls,maximum_visual_mass_error=max(errors,default=0.),current36_cache_restored=True))
            assert fingerprint()==full_before;record['full36_cache_unchanged_after_all_trials']=True
            results.append(record);write(root/f'results{shard}.json',results)
            print(dict(image=image,step=step,gain=gain,choices={(t['scope']+'/'+t['policy']):t['new_token_text']for t in record['trials']}),flush=True)
    handle.remove();assert len(encoder)==1 and reader.head_calls==0 and len(results)==len(cases);check(plan)
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,encoder_calls=len(encoder),cases=len(results),
        new_semantic_head_calls=reader.head_calls,zero_exact=zero_checked,nonzero_tail_replays=replays,
        all_case_full36_and_each_proposal_current36_verified=True,no_generation=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--verify',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root)
        elif args.verify:verify(args.root)
        elif args.worker is not None:worker(args.root,args.worker)
        else:parser.error('Select a stage.')
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
