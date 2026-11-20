"""Check the new veto, disabled-veto equivalence and exact rejection at B2/B16."""

from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
from features.qwen3_collective_visual_gate import Qwen3BatchCollectiveVisualGate
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read,write,source_groups,sha


def verify(wrapper,probe,allowed,parser,threshold):
    first=source_groups(REPO/'outputs/qwen3_block_acg_pilot_20261118/coco100/sadi.shard0.jsonl')[0]
    report=dict(passed=True,batches={});live_proposals=live_vetoes=0
    for label,ids in [('B2',[485532,558955]),('B16',[r['image_id'] for r in first])]:
        (span,group),=same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
        inputs={k:torch.cat([d[k] for _,d in group]).to(wrapper.device) for k in group[0][1]}
        prompt=inputs['input_ids'].shape[1];eos=wrapper.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos]);static={}
        def run(controller):
            logits,kv,encoders=[],[],[];backwards=[]
            def observe(_m,_a,out):
                logits.append(out.logits[:,-1].cpu().clone());cache=out.past_key_values
                assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
                kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
            grad=torch.autograd.grad
            def backward(*a,**kw):backwards.append(1);return grad(*a,**kw)
            with ExitStack() as stack:
                stack.enter_context(controller.capture());stack.enter_context(patch.object(torch.autograd,'grad',backward))
                stack.callback(wrapper.model.register_forward_hook(observe).remove)
                stack.callback(wrapper.model.model.visual.register_forward_pre_hook(lambda *_:encoders.append(1)).remove)
                output=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
            tokens=[trim_eos(t,eos) for t in output[:,prompt:].tolist()]
            assert len(logits)==32 and len(encoders)==1
            stat=controller.statistics(tokens)
            if isinstance(controller,Qwen3BatchBlockACGLogged):
                assert max(s['gradient_probe_calls'] for s in stat)==len(backwards)
            return tokens,logits,kv,stat,len(backwards)
        def exact(a,b):
            assert a[0]==b[0]
            assert all(torch.equal(x,y) for x,y in zip(a[1],b[1],strict=True))
            assert all(torch.equal(x,y) for xx,yy in zip(a[2],b[2],strict=True) for x,y in zip(xx,yy,strict=True))
        arguments=dict(probe=probe,semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,
            history_parser=parser,real_threshold=threshold,shared_static=static,signal_guided=True,
            diagnostic_force_probe_gate=True)
        with torch.inference_mode():
            native=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'));gc.collect();torch.cuda.empty_cache()
            original=run(Qwen3BatchBlockACGLogged(wrapper.model,span,allowed,**arguments));gc.collect();torch.cuda.empty_cache()
            records={}
            for name,options in [('veto_disabled',dict(use_collective_veto=False)),
                    ('reject',dict(diagnostic_reject_all=True)),('live',{})]:
                controller=Qwen3BatchCollectiveVisualGate(wrapper.model,span,allowed,**arguments,**options)
                current=run(controller)
                if name=='veto_disabled':exact(current,original)
                if name=='reject':exact(current,native)
                stats=current[3];events=[e for s in stats for e in s['full_joint_events']]
                decisions=[d for s in stats for d in s['full_collective_decisions']]
                if name!='veto_disabled':
                    assert all(e['collective_visual_effect']<0 for e in events)
                    assert all(d['veto']==(d['collective_visual_effect']>=0) for d in decisions)
                if name=='live':live_proposals+=len(events);live_vetoes+=sum(d['veto'] for d in decisions)
                records[name]=dict(proposals=len(events),kept=sum(e['accepted'] for e in events),
                    vetoes=sum(d['veto'] for d in decisions),backward_calls=current[4],encoder_calls=1,
                    all36_current_KV_logits_tokens_exact_to='original_guided' if name=='veto_disabled' else 'sadi' if name=='reject' else None)
                del controller,current;gc.collect();torch.cuda.empty_cache()
        report['batches'][label]=dict(ids=ids,cases=records)
        del native,original,inputs,static;gc.collect();torch.cuda.empty_cache()
    assert live_proposals>0 and live_vetoes>0
    return report


if __name__=='__main__':
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    root=REPO/'outputs/qwen3_collective_visual_preflight'
    plan=read(REPO/'outputs/qwen3_block_acg_pilot_20261118/protocol.json')
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(read(REPO/'outputs/qwen3_block_acg_pilot_20261118/object_lexicon.json')['token_ids'])
    result=verify(wrapper,probe,allowed,ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'),plan['real_threshold'])
    result['source_sha256']={n:sha(REPO/n) for n in ['features/qwen3_collective_visual_gate.py',
        'scripts/verify_qwen3_collective_visual_gate.py','scripts/verify_qwen3_collective_visual_gpu.py']}
    write(root/'verification.json',result);print(__import__('json').dumps(result,indent=2))
