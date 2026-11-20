"""Verify native108 inputs, context reuse and exact rejection of bounded tilts."""

from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_bounded_visual_tilt import Qwen3BatchBoundedVisualTilt
from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read,write,source_groups,sha


def verify(wrapper,probe,allowed,parser,threshold):
    first=source_groups(REPO/'outputs/qwen3_block_acg_pilot_20261118/coco100/sadi.shard0.jsonl')[0]
    report=dict(passed=True,batches={})
    for label,ids in [('B2',[485532,558955]),('B16',[r['image_id'] for r in first])]:
        (span,group),=same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
        inputs={k:torch.cat([d[k] for _,d in group]).to(wrapper.device) for k in group[0][1]}
        prompt=inputs['input_ids'].shape[1];eos=wrapper.model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos]);static={}
        def run(controller):
            logits,kv,features,encoders=[],[],[],[]
            def observe(_m,_a,out):
                logits.append(out.logits[:,-1].cpu().clone());cache=out.past_key_values
                assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
                kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
            def forbid(*_a,**_kw):raise AssertionError('Bounded attention tilts must not backpropagate')
            torch.cuda.reset_peak_memory_stats(wrapper.device)
            with ExitStack() as stack:
                stack.enter_context(controller.capture());stack.enter_context(patch.object(torch.autograd,'grad',forbid))
                stack.callback(wrapper.model.register_forward_hook(observe).remove)
                stack.callback(wrapper.model.model.visual.register_forward_pre_hook(lambda *_:encoders.append(1)).remove)
                stack.callback(probe.register_forward_pre_hook(lambda _m,a:features.append(a[0].cpu().clone())).remove)
                output=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
            tokens=[trim_eos(t,eos) for t in output[:,prompt:].tolist()]
            assert len(logits)==32 and len(encoders)==1
            return tokens,logits,kv,controller.statistics(tokens),features,torch.cuda.max_memory_allocated(wrapper.device)
        def exact(a,b):
            assert a[0]==b[0] and all(torch.equal(x,y) for x,y in zip(a[1],b[1],strict=True))
            assert all(torch.equal(x,y) for xx,yy in zip(a[2],b[2],strict=True) for x,y in zip(xx,yy,strict=True))
        args=dict(probe=probe,semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
            real_threshold=threshold,shared_static=static,diagnostic_force_probe_gate=True)
        with torch.inference_mode():
            native=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'));gc.collect();torch.cuda.empty_cache()
            reference=run(Qwen3BatchBlockACGLogged(wrapper.model,span,allowed,signal_guided=False,diagnostic_reject_all=True,**args))
            exact(reference,native);gc.collect();torch.cuda.empty_cache()
            cases=[('zero_tilt','evidence',dict(diagnostic_zero_tilt=True)),
                ('reject_evidence','evidence',dict(diagnostic_reject_all=True)),('live_evidence','evidence',{})]
            if label=='B2':cases=[('disabled','evidence',dict(disable_guidance=True)),
                ('zero_gamma','evidence',dict(gamma=0.)),('reject_fixed','fixed',dict(diagnostic_reject_all=True)),
                ('reject_drift','drift',dict(diagnostic_reject_all=True)),*cases]
            records={}
            for name,mode,options in cases:
                controller=Qwen3BatchBoundedVisualTilt(wrapper.model,span,allowed,tilt_mode=mode,**args,**options)
                current=run(controller);is_exact=not name.startswith('live')
                if is_exact:exact(current,native)
                if is_exact and name!='disabled':
                    assert all(torch.equal(x,y) for x,y in zip(current[4],reference[4],strict=True))
                    for a,b in zip(current[3],reference[3],strict=True):
                        assert a['full_class_decisions']==b['full_class_decisions'] and a['full_gate_events']==b['full_gate_events']
                stats=current[3];events=[e for s in stats for e in s['full_joint_events']]
                assert all(s['gradient_probe_calls']==0 for s in stats)
                assert all(v['visual_logit_shift_min']>=0 and 0<=v['coefficient_min']<=v['coefficient_max']<=1
                    for e in events for v in e['layer_details'].values())
                if name=='zero_tilt':assert not events and max(s['zero_tilt_exact_checks'] for s in stats)>0
                if name.startswith(('reject','live')):assert events
                assert current[5]<29*2**30
                records[name]=dict(proposals=len(events),kept=sum(e['accepted'] for e in events),
                    exact_native_tokens_logits_current36_KV=is_exact,
                    exact_original108_gate_inputs=is_exact and name!='disabled',
                    zero_tilt_native_context_checks=max(s['zero_tilt_exact_checks'] for s in stats),
                    encoder_calls=1,model_calls=32,backward_calls=0,peak_allocated=current[5])
                del controller,current;gc.collect();torch.cuda.empty_cache()
        report['batches'][label]=dict(ids=ids,cases=records)
        del native,reference,inputs,static;gc.collect();torch.cuda.empty_cache()
    return report


if __name__=='__main__':
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    root=REPO/'outputs/qwen3_bounded_visual_tilt_preflight'
    plan=read(REPO/'outputs/qwen3_block_acg_pilot_20261118/protocol.json')
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(read(REPO/'outputs/qwen3_block_acg_pilot_20261118/object_lexicon.json')['token_ids'])
    result=verify(wrapper,probe,allowed,ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'),plan['real_threshold'])
    result['source_sha256']={n:sha(REPO/n) for n in ['features/qwen3_bounded_visual_tilt.py',
        'scripts/verify_qwen3_bounded_visual_tilt.py','scripts/verify_qwen3_bounded_visual_tilt_gpu.py']}
    write(root/'verification.json',result);print(__import__('json').dumps(result,indent=2))
