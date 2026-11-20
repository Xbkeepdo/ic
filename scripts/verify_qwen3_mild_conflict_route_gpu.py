"""Verify zero/strong endpoints and half-strength replay before new100 generation."""

import argparse
from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import torch
from features.qwen3_conflict_route_feedback import Qwen3BatchConflictRoute
from features.qwen3_mild_conflict_route_feedback import Qwen3BatchMildConflictRoute
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
    strong=run(Qwen3BatchConflictRoute(wrapper.model,span,allowed,route_mode='signal',**args))
    mild_vri=run(Qwen3BatchMildConflictRoute(wrapper.model,span,allowed,route_mode='vri',**args))
    cases=[('scale_zero','signal',dict(attenuation_scale=0.)),
        ('scale_one','signal',dict(attenuation_scale=1.)),
        ('zero_route','signal',dict(diagnostic_zero_route=True)),
        ('forced_reject','signal',dict(diagnostic_reject_all=True)),
        ('flat_signal','signal',dict(diagnostic_flat_signal=True)),
        ('live_signal','signal',{}),('live_roll','roll',{})]
    if label=='B2':cases=[('disabled','signal',dict(disable_guidance=True)),('gamma_zero','signal',dict(gamma=0.)),*cases]
    results={}
    for name,mode,options in [('mild_vri','vri',None),*cases]:
        current=mild_vri if options is None else run(Qwen3BatchMildConflictRoute(wrapper.model,span,allowed,route_mode=mode,**args,**options))
        identity=name in ['disabled','gamma_zero','scale_zero','zero_route','forced_reject']
        if identity:exact(current,native)
        if name=='scale_one':exact(current,strong)
        if name=='flat_signal':exact(current,mild_vri)
        scale=0. if name=='scale_zero' else 1. if name=='scale_one' else .5
        events=[e for s in current['stats'] for e in s['full_joint_events']]
        for stat in current['stats']:
            assert stat['mild_conflict_route'] and stat['attenuation_scale']==scale
            assert stat['maximum_probe_hidden_difference']==0
        for event in events:
            assert event['attenuation_scale']==scale and event['no_learned_gate']
            total=0.
            for layer in event['layer_details'].values():
                strong_g=torch.tensor(layer['strong_text_gates']);actual=torch.tensor(layer['text_gates'])
                expected=strong_g if scale==1 else torch.ones_like(strong_g) if scale==0 else 1-scale*(1-strong_g)
                assert torch.equal(actual,expected) and layer['attenuation_scale']==scale
                assert torch.equal(actual<1,strong_g<1) if scale>0 else torch.equal(actual,torch.ones_like(actual))
                if name not in ['scale_zero','scale_one','zero_route']:
                    assert (actual[actual<1]>=.5).all() and (actual[actual<1]<=.75).all()
                independent=sum((g-1)*effect for g,effect in zip(layer['text_gates'],layer['textual_effect'],strict=True))
                assert abs(independent-layer['predicted_log_probability_change'])<1e-9
                total+=independent
            assert abs(total-event['predicted_log_probability_change'])<1e-8
        assert current['peak']<30*1024**3
        results[name]=dict(proposals=len(events),kept=sum(e['accepted'] for e in events),
            exact_sadi_tokens_logits_current36_KV=identity,exact_strong_at_scale_one=name=='scale_one',
            exact_mild_vri_when_flat=name=='flat_signal',historical_KV_tensor_checks=current['historical_KV_checks'],
            gradient_calls=current['gradients'],model_calls=current['model_calls'],encoder_calls=1,peak=current['peak'],
            maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in current['stats']))
        if options is not None:del current
        gc.collect();torch.cuda.empty_cache()
    assert results['live_signal']['proposals']>0 and results['scale_zero']['proposals']>0
    return dict(passed=True,batches={label:dict(ids=image_ids,cases=results)})


if __name__=='__main__':
    from features.coco_object_history import ObjectClassHistory
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-label',choices=('B2','B16'),required=True)
    label=p.parse_args().batch_label;root=REPO/'outputs/qwen3_mild_conflict_route_preflight_20261119'
    request=read(root/'request.json');destination=root/f'verification.{label}.json';assert not destination.exists()
    for n,v in request['source_sha256'].items():assert sha(REPO/n)==v,n
    wrapper=build_wrapper();allowed=frozenset(read(REPO/'outputs/qwen3_conflict_route_pilot_20261119/object_lexicon.json')['token_ids'])
    result=verify(wrapper,allowed,ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'),label)
    for n,v in request['source_sha256'].items():assert sha(REPO/n)==v,n
    result.update(request_sha256=sha(root/'request.json'),source_sha256={n:sha(REPO/n) for n in request['new_files']})
    write(destination,result);print(__import__('json').dumps(result))
