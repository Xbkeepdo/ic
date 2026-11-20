"""Verify soft-dose no-ops, real route edits and cache identity before pilot."""

import argparse
from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import torch
from features.qwen3_conflict_route_feedback import Qwen3BatchConflictRoute
from features.qwen3_soft_conflict_route_feedback import Qwen3BatchSoftConflictRoute
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read,write,sha,source_groups


def verify(wrapper,allowed,parser,label):
    first=source_groups(REPO/'outputs/qwen3_conflict_route_pilot_20261119/coco100/sadi.shard0.jsonl')[0]
    image_ids=[485532,558955] if label=='B2' else [r['image_id'] for r in first]
    (span,group),=same_span_batches(wrapper,image_ids,COCO_ROOT,len(image_ids))
    inputs={k:torch.cat([item[k] for _,item in group]).to(wrapper.device) for k in group[0][1]}
    prompt=inputs['input_ids'].shape[1];eos=wrapper.model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]);shared_static={}

    def run(controller):
        logits=[];kv=[];previous=None;encoder_calls=gradients=past_checks=0
        original_grad=torch.autograd.grad
        def gradient(*args,**kwargs):
            nonlocal gradients
            gradients+=1;return original_grad(*args,**kwargs)
        def encoder(*_):
            nonlocal encoder_calls
            encoder_calls+=1
        def observe(_module,_args,output):
            nonlocal previous,past_checks
            logits.append(output.logits[:,-1].cpu().clone());cache=output.past_key_values
            assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
            if previous is not None:
                for layer,(keys,values) in zip(cache.layers,previous,strict=True):
                    assert torch.equal(layer.keys[:,:,:-1],keys) and torch.equal(layer.values[:,:,:-1],values);past_checks+=2
            kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
            previous=[(layer.keys,layer.values) for layer in cache.layers]
        torch.cuda.reset_peak_memory_stats(wrapper.device)
        with ExitStack() as stack:
            stack.enter_context(controller.capture());stack.enter_context(patch.object(torch.autograd,'grad',gradient))
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            stack.callback(wrapper.model.model.visual.register_forward_pre_hook(encoder).remove)
            with torch.inference_mode():
                output=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
        response=[trim_eos(row,eos) for row in output[:,prompt:].tolist()]
        stats=controller.statistics(response)
        assert encoder_calls==1 and all(not p.requires_grad for p in wrapper.model.parameters())
        if isinstance(controller,Qwen3BatchConflictRoute):
            assert max(s['gradient_probe_calls'] for s in stats)==gradients
            assert all(s['native_risk_solver_calls']==s['probe_calls']==0 for s in stats)
        return dict(tokens=response,logits=logits,kv=kv,stats=stats,gradients=gradients,
            model_calls=len(logits),historical_KV_checks=past_checks,encoder_calls=encoder_calls,
            peak=torch.cuda.max_memory_allocated(wrapper.device))

    def exact(a,b):
        assert a['tokens']==b['tokens']
        assert all(torch.equal(x,y) for x,y in zip(a['logits'],b['logits'],strict=True))
        assert all(torch.equal(x,y) for xx,yy in zip(a['kv'],b['kv'],strict=True) for x,y in zip(xx,yy,strict=True))
    native=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'))
    if label=='B16':assert native['tokens']==[r['response_token_ids'][:32] for r in first]
    args=dict(tokenizer=wrapper.tokenizer,history_parser=parser,shared_static=shared_static)
    cases=[('zero_route','signal',dict(diagnostic_zero_route=True)),
        ('forced_reject','signal',dict(diagnostic_reject_all=True)),
        ('neutral_signal','signal',dict(diagnostic_flat_signal=True)),
        ('neutral_flat','flat',dict(diagnostic_flat_signal=True)),
        ('live_signal','signal',{}),('live_flat','flat',{}),('live_roll','roll',{})]
    if label=='B2':cases=[('disabled','signal',dict(disable_guidance=True)),('gamma_zero','signal',dict(gamma=0.)),*cases]
    results={}
    for name,mode,options in cases:
        current=run(Qwen3BatchSoftConflictRoute(wrapper.model,span,allowed,dose_mode=mode,**args,**options))
        identity=name in ['disabled','gamma_zero','zero_route','forced_reject','neutral_signal','neutral_flat']
        if identity:exact(current,native)
        events=[e for s in current['stats'] for e in s['full_joint_events']]
        for stat in current['stats']:
            assert stat['soft_conflict_route'] and stat['dose_mode']==mode
            assert stat['maximum_probe_hidden_difference']==0
        for event in events:
            assert event['soft_conflict_route'] and event['no_learned_gate']
            total=0.
            for layer in event['layer_details'].values():
                vis,txt,risk=layer['visual_effect'],layer['textual_effect'],layer['raw_head_risk']
                chosen=sorted((h for h in range(32) if vis[h]<0 and txt[h]>0),
                    key=lambda h:(abs(vis[h])/(abs(vis[h])+abs(txt[h])+1e-8),h))[:8]
                assert chosen==layer['base_selected_heads']
                raw=[max(0.,risk[h]-.5) for h in chosen]
                applied=raw[:]
                if mode=='flat' and chosen:applied=[sum(raw)/len(raw)]*len(raw)
                if mode=='roll' and chosen:
                    shift=len(chosen)//2
                    applied=raw[-shift:]+raw[:-shift] if shift else raw[:]
                expected=[1.]*32
                for h,dose in zip(chosen,applied):expected[h]=1-dose
                if name=='zero_route':expected=[1.]*32
                actual=torch.tensor(layer['text_gates'])
                assert torch.equal(actual,torch.tensor(expected))
                assert (actual>=.5).all() and (actual<=1).all()
                independent=sum((g-1)*effect for g,effect in zip(layer['text_gates'],txt,strict=True))
                assert abs(independent-layer['predicted_log_probability_change'])<1e-9
                total+=independent
            assert abs(total-event['predicted_log_probability_change'])<1e-8
        assert current['peak']<30*1024**3
        results[name]=dict(proposals=len(events),kept=sum(e['accepted'] for e in events),
            exact_sadi_tokens_logits_current36_KV=identity,
            neutral_signal_is_zero_dose=name.startswith('neutral'),
            historical_KV_tensor_checks=current['historical_KV_checks'],
            gradient_calls=current['gradients'],model_calls=current['model_calls'],encoder_calls=1,peak=current['peak'],
            maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in current['stats']))
        del current
        gc.collect();torch.cuda.empty_cache()
    assert all(results['live_'+mode]['proposals']>0 for mode in ['signal','flat','roll'])
    assert results['zero_route']['proposals']>0
    return dict(passed=True,batches={label:dict(ids=image_ids,cases=results)})


if __name__=='__main__':
    from features.coco_object_history import ObjectClassHistory
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-label',choices=('B2','B16'),required=True)
    label=p.parse_args().batch_label;root=REPO/'outputs/qwen3_soft_conflict_route_preflight_20261119'
    request=read(root/'request.json');destination=root/f'verification.{label}.json';assert not destination.exists()
    for n,v in request['source_sha256'].items():assert sha(REPO/n)==v,n
    wrapper=build_wrapper();allowed=frozenset(read(REPO/'outputs/qwen3_conflict_route_pilot_20261119/object_lexicon.json')['token_ids'])
    result=verify(wrapper,allowed,ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'),label)
    for n,v in request['source_sha256'].items():assert sha(REPO/n)==v,n
    result.update(request_sha256=sha(root/'request.json'),source_sha256={n:sha(REPO/n) for n in request['new_files']})
    write(destination,result);print(__import__('json').dumps(result))
