"""Verify full-SADI route gating, actual text contributions and current/history KV."""

import argparse
from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features import qwen3_conflict_route_feedback as route
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read,write,sha,source_groups


def verify(wrapper,allowed,parser,label):
    first=source_groups(REPO/'outputs/qwen3_evidence_head_routing_pilot_20261119/coco100/sadi.shard0.jsonl')[0]
    ids=[485532,558955] if label=='B2' else [r['image_id'] for r in first]
    (span,group),=same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
    inputs={k:torch.cat([item[k] for _,item in group]).to(wrapper.device) for k in group[0][1]}
    prompt=inputs['input_ids'].shape[1];eos=wrapper.model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos]);shared_static={}

    def run(controller):
        logits=[];kv=[];previous=None;past_checks=encoder_calls=gradient_calls=route_checks=0
        text_reference={};context_reference={};phase={'replay':False,'index':19,'rows':[]}
        native_eager=None
        native_apply=route.apply_text_gates
        original_gradient=torch.autograd.grad

        def eager(module,query,key,value,attention_mask,scaling,dropout=0.,**kw):
            output,attention=native_eager(module,query,key,value,attention_mask,scaling,dropout,**kw)
            index=getattr(module,'layer_idx',None)
            if isinstance(module,qwen.Qwen3VLTextAttention) and index in range(19,36) and query.shape[-2]==1:
                if not getattr(controller,'_replaying',False) or phase['replay']:
                    values=qwen.repeat_kv(value,module.num_key_value_groups)
                    a=torch.cat((attention[:,:,:,:span[0]],attention[:,:,:,span[1]:]),-1)
                    v=torch.cat((values[:,:,:span[0]],values[:,:,span[1]:]),-2)
                    text_reference[index]=(a@v)[:,:,0].detach()
                    context_reference[index]=output[:,0].detach()
                    if phase['replay']:phase['index']=index
            return output,attention

        def apply(context,textual,gates):
            nonlocal route_checks
            index=phase['index'];rows=phase['rows']
            assert phase['replay'] and torch.equal(textual,text_reference[index][rows])
            assert torch.equal(context,context_reference[index][rows])
            result=native_apply(context,textual,gates)
            independent=(context.float()-(1-gates)[...,None]*textual.float()).to(context.dtype)
            assert torch.equal(result,independent)
            route_checks+=1;return result

        def gradient(*args,**kwargs):
            nonlocal gradient_calls
            gradient_calls+=1;return original_gradient(*args,**kwargs)

        def encoder(*_):
            nonlocal encoder_calls
            encoder_calls+=1

        def observe(_module,_args,output):
            nonlocal previous,past_checks
            logits.append(output.logits[:,-1].cpu().clone());cache=output.past_key_values
            assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
            if previous is not None:
                for layer,(keys,values) in zip(cache.layers,previous,strict=True):
                    assert torch.equal(layer.keys[:,:,:-1],keys) and torch.equal(layer.values[:,:,:-1],values)
                    past_checks+=2
            kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
            previous=[(layer.keys,layer.values) for layer in cache.layers]

        torch.cuda.reset_peak_memory_stats(wrapper.device)
        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            native_eager=qwen.eager_attention_forward
            stack.enter_context(patch.object(qwen,'eager_attention_forward',eager))
            stack.enter_context(patch.object(route,'apply_text_gates',apply))
            stack.enter_context(patch.object(torch.autograd,'grad',gradient))
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            stack.callback(wrapper.model.model.visual.register_forward_pre_hook(encoder).remove)
            if isinstance(controller,route.Qwen3BatchConflictRoute):
                replay=controller._replay
                def traced_replay(rows,gates):
                    phase.update(replay=True,index=19,rows=rows)
                    try:return replay(rows,gates)
                    finally:phase['replay']=False
                stack.enter_context(patch.object(controller,'_replay',traced_replay))
            with torch.inference_mode():
                tokens=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
        response=[trim_eos(row,eos) for row in tokens[:,prompt:].tolist()]
        stats=controller.statistics(response);assert encoder_calls==1
        if isinstance(controller,route.Qwen3BatchConflictRoute):
            assert max(s['gradient_probe_calls'] for s in stats)==gradient_calls
            assert max(s['route_layer_calls'] for s in stats)==route_checks
            assert all(s['probe_calls']==s['native_risk_solver_calls']==0 for s in stats)
        assert all(not p.requires_grad for p in wrapper.model.parameters())
        return dict(tokens=response,logits=logits,kv=kv,stats=stats,encoders=encoder_calls,
            gradient_calls=gradient_calls,route_checks=route_checks,past_checks=past_checks,
            peak=torch.cuda.max_memory_allocated(wrapper.device))

    def exact(a,b):
        assert a['tokens']==b['tokens']
        assert all(torch.equal(x,y) for x,y in zip(a['logits'],b['logits'],strict=True))
        assert all(torch.equal(x,y) for xx,yy in zip(a['kv'],b['kv'],strict=True) for x,y in zip(xx,yy,strict=True))

    base=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'))
    if label=='B16':assert base['tokens']==[r['response_token_ids'][:32] for r in first]
    args=dict(tokenizer=wrapper.tokenizer,history_parser=parser,shared_static=shared_static)
    reference=run(route.Qwen3BatchConflictRoute(wrapper.model,span,allowed,route_mode='vri',**args))
    cases=[('zero_route','signal',dict(diagnostic_zero_route=True)),
        ('forced_reject','signal',dict(diagnostic_reject_all=True)),
        ('flat_signal','signal',dict(diagnostic_flat_signal=True)),
        ('live_signal','signal',{}),('live_roll','roll',{})]
    if label=='B2':cases=[('disabled','signal',dict(disable_guidance=True)),('gamma_zero','signal',dict(gamma=0.)),*cases]
    results={}
    for name,mode,options in [('vri_reference','vri',None),*cases]:
        current=reference if options is None else run(route.Qwen3BatchConflictRoute(wrapper.model,span,allowed,route_mode=mode,**args,**options))
        unchanged=name in ('disabled','gamma_zero','zero_route','forced_reject')
        if unchanged:exact(current,base)
        if name=='flat_signal':exact(current,reference)
        events=[e for s in current['stats'] for e in s['full_joint_events']]
        for event in events:
            assert event['probe_p_real'] is event['probe_real_threshold'] is None
            assert event['no_learned_gate'] and event['joint_layers']==list(range(19,36))
            assert event['predicted_log_probability_change']<=1e-6
            for layer in event['layer_details'].values():
                v=torch.tensor(layer['visual_effect']);t=torch.tensor(layer['textual_effect'])
                conflict=(v<0)&(t>0);g=torch.tensor(layer['text_gates']);risk=torch.tensor(layer['raw_head_risk'])
                assert layer['native_conflict_mask']==conflict.tolist()
                assert torch.equal(g[~conflict],torch.ones_like(g[~conflict]))
                if name=='zero_route':assert torch.equal(g,torch.ones_like(g))
                else:
                    count=min(8,int(conflict.sum()));assert int((g<1).sum())==count
                    expected=torch.cat((torch.linspace(0,.5,count) if count>1 else torch.tensor([0.]) if count else torch.tensor([]),torch.ones(len(g)-count))).sort().values
                    assert torch.equal(g.sort().values,expected)
                    heads=conflict.nonzero().flatten();vri=v.abs()/(v.abs()+t.abs()+1e-8)
                    ordered=heads[torch.argsort(vri[heads],stable=True)]
                    if mode!='vri':
                        applied=risk.roll(len(risk)//2) if mode=='roll' else risk
                        ordered=ordered[torch.argsort(applied[ordered],descending=True,stable=True)]
                    assert set(layer['selected_heads'])==set(ordered[:8].tolist())
        assert current['peak']<30*1024**3
        results[name]=dict(proposals=len(events),kept=sum(e['accepted'] for e in events),
            exact_sadi_tokens_logits_current36_KV=unchanged,exact_vri_when_flat=name=='flat_signal',
            historical_KV_tensor_checks=current['past_checks'],actual_current_route_checks=current['route_checks'],
            gradient_calls=current['gradient_calls'],encoders=1,model_calls=len(current['logits']),peak=current['peak'],
            maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in current['stats']),
            maximum_replayed_text_route_difference=max(s['maximum_replayed_text_route_difference'] for s in current['stats']))
        assert results[name]['maximum_probe_hidden_difference']==0.
        if options is not None:del current
        gc.collect();torch.cuda.empty_cache()
    assert results['vri_reference']['proposals']>0 and results['live_signal']['proposals']>0
    assert results['zero_route']['actual_current_route_checks']>0
    return dict(passed=True,batches={label:dict(ids=ids,cases=results)})


if __name__=='__main__':
    from features.coco_object_history import ObjectClassHistory
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch-label',required=True,choices=('B2','B16'))
    label=parser.parse_args().batch_label;root=REPO/'outputs/qwen3_conflict_route_preflight_20261119'
    request=read(root/'request.json');destination=root/f'verification.{label}.json';assert not destination.exists()
    for n,v in request['source_sha256'].items():assert sha(REPO/n)==v,n
    assert read(REPO/'outputs/qwen3_conflict_route_cpu_20261119/verification.json')['passed']
    wrapper=build_wrapper();allowed=frozenset(read(REPO/'outputs/qwen3_block_acg_pilot_20261118/object_lexicon.json')['token_ids'])
    result=verify(wrapper,allowed,ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'),label)
    for n,v in request['source_sha256'].items():assert sha(REPO/n)==v,n
    result.update(request_sha256=sha(root/'request.json'),source_sha256={n:sha(REPO/n) for n in request['new_files']})
    write(destination,result);print(__import__('json').dumps(result))
