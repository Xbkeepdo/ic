"""Check scalar-only SADI/SVAR proposals against their actual native bases."""

from contextlib import ExitStack
import gc
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_scalar_visual_feedback import Qwen3BatchScalarVisualFeedback
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read,write,source_groups


def verify(wrapper,probe,allowed,parser,threshold):
    first=source_groups(REPO/'outputs/qwen3_block_acg_pilot_20261118/coco100/svar.shard0.jsonl')[0]
    cohorts=[('B2',[485532,558955]),('B16',[r['image_id'] for r in first])]
    report=dict(passed=True,batches={})
    for label,ids in cohorts:
        (span,group),=same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
        inputs={k:torch.cat([d[k] for _,d in group]).to(wrapper.device) for k in group[0][1]}
        prompt=inputs['input_ids'].shape[1]
        eos=wrapper.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        def run(controller):
            logits,kv,encoders=[],[],[]
            def observe(_m,_a,out):
                logits.append(out.logits[:,-1].cpu().clone());cache=out.past_key_values
                assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
                kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
            def forbid(*_a,**_kw):raise AssertionError('Scalar guidance must not invoke autograd.grad')
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.autograd,'grad',forbid))
                stack.enter_context(controller.capture())
                stack.callback(wrapper.model.register_forward_hook(observe).remove)
                stack.callback(wrapper.model.model.visual.register_forward_pre_hook(lambda *_:encoders.append(1)).remove)
                output=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
            tokens=[trim_eos(t,eos) for t in output[:,prompt:].tolist()]
            assert len(logits)==32 and len(encoders)==1
            return tokens,logits,kv,controller.statistics(tokens)
        native={}
        with torch.inference_mode():
            for base in (('sadi','svar') if label=='B2' else ('svar',)):
                native[base]=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,base))
                gc.collect();torch.cuda.empty_cache()
            modes=[('svar',True,True,1.4),('svar',True,False,1.4)]
            if label=='B2':modes += [('sadi',True,True,1.4),('svar',False,True,1.4),('svar',True,False,0.)]
            entries=[]
            for base,use_drift,reject,gamma in modes:
                controller=Qwen3BatchScalarVisualFeedback(wrapper.model,span,allowed,probe=probe,
                    semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                    real_threshold=threshold,base_variant=base,use_drift=use_drift,gamma=gamma,
                    diagnostic_force_probe_gate=True,diagnostic_reject_all=reject)
                current=run(controller)
                exact=reject or gamma==0
                if exact:
                    assert current[0]==native[base][0]
                    assert all(torch.equal(a,b) for a,b in zip(current[1],native[base][1],strict=True))
                    assert all(torch.equal(a,b) for x,y in zip(current[2],native[base][2],strict=True) for a,b in zip(x,y,strict=True))
                stats=current[3]
                assert all(s['gradient_probe_calls']==0 and not s['head_gradient_guidance'] for s in stats)
                proposals=sum(s['total_proposals'] for s in stats)
                if gamma:assert proposals>0
                entries.append(dict(base=base,use_drift=use_drift,reject=reject,gamma=gamma,
                    proposals=proposals,kept=sum(s['total_accepted'] for s in stats),
                    all36_current_KV_logits_tokens_native_exact=exact,autograd_calls=0,encoder_calls=1,model_calls=32))
                del controller,current;gc.collect();torch.cuda.empty_cache()
        report['batches'][label]=dict(ids=ids,cases=entries)
        del native,inputs;gc.collect();torch.cuda.empty_cache()
    return report


if __name__=='__main__':
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    out=REPO/'outputs/qwen3_scalar_visual_preflight'
    plan=read(REPO/'outputs/qwen3_block_acg_pilot_20261118/protocol.json')
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(read(REPO/'outputs/qwen3_block_acg_pilot_20261118/object_lexicon.json')['token_ids'])
    result=verify(wrapper,probe,allowed,ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'),plan['real_threshold'])
    from scripts.run_qwen3_signed_feedback import sha
    result['source_sha256']={n:sha(REPO/n) for n in ('features/qwen3_scalar_visual_feedback.py','scripts/verify_qwen3_scalar_visual.py','scripts/verify_qwen3_scalar_visual_gpu.py')}
    write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))
