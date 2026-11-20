"""Check joint17-layer replay, exact rejection and the actual B16 footprint."""

from contextlib import ExitStack
import gc
from pathlib import Path
from unittest.mock import patch

import torch
from features.qwen3_block_acg_signal import Qwen3BatchBlockACG, POST
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read, source_groups


def verify(wrapper, probe, allowed, parser, threshold):
    source=Path('/root/rivermind-data/ic-ac/outputs/qwen3_acg_signal_pilot_20261118')
    groups=source_groups(source/'coco100/sadi.shard0.jsonl')
    report=dict(passed=True,layer_block=list(POST),cases={},checked_current_KV_layers=list(range(36)))
    for group_name, ids in [('B2',[485532,558955]),('B16',[r['image_id'] for r in groups[0]])]:
        (span,group),=same_span_batches(wrapper,ids,COCO_ROOT,16)
        assert [i for i,_ in group]==ids
        inputs={k:torch.cat([d[k] for _,d in group]).to(wrapper.device) for k in group[0][1]}
        prompt=inputs['input_ids'].shape[1]
        shared={}
        def run(controller):
            logits,kv,counts=[],[],dict(encoder=0,model=0,backward=0)
            def observe(_module,_args,out):
                logits.append(out.logits[:,-1].cpu().clone())
                cache=out.past_key_values
                assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
                kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
            def count(name):
                def hook(*_):counts[name]+=1
                return hook
            grad=torch.autograd.grad
            def backward(*args,**kwargs):counts['backward']+=1;return grad(*args,**kwargs)
            torch.cuda.reset_peak_memory_stats(wrapper.device)
            with ExitStack() as stack:
                stack.enter_context(controller.capture())
                stack.enter_context(patch.object(torch.autograd,'grad',backward))
                stack.callback(wrapper.model.register_forward_hook(observe).remove)
                stack.callback(wrapper.model.register_forward_pre_hook(count('model')).remove)
                stack.callback(wrapper.model.model.visual.register_forward_pre_hook(count('encoder')).remove)
                tokens=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
            peak=torch.cuda.max_memory_allocated(wrapper.device)
            assert counts['model']==32 and counts['encoder']==1
            stats=controller.statistics(tokens[:,prompt:].tolist())
            return tokens.cpu(),logits,kv,stats,counts,peak
        with torch.inference_mode():
            baseline=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'))
            cases=[('reject_guided',True,dict(diagnostic_reject_all=True)),('live_guided',True,{})] if group_name=='B16' else [
                ('disabled',True,dict(disable_guidance=True)),('zero_gamma',True,dict(gamma=0.)),
                ('zero_joint',True,dict(diagnostic_zero_replay=True)),
                ('reject_fixed',False,dict(diagnostic_reject_all=True)),
                ('reject_guided',True,dict(diagnostic_reject_all=True)),
                ('live_fixed',False,{}),('live_guided',True,{})]
            for name,guided,options in cases:
                controller=Qwen3BatchBlockACG(wrapper.model,span,allowed,probe=probe,
                    semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                    real_threshold=threshold,shared_static=shared,signal_guided=guided,
                    diagnostic_force_probe_gate=True,**options)
                current=run(controller)
                exact=not name.startswith('live_')
                if exact:
                    assert torch.equal(current[0],baseline[0]),(group_name,name)
                    assert all(torch.equal(a,b) for a,b in zip(current[1],baseline[1],strict=True)),(group_name,name)
                    assert all(torch.equal(a,b) for x,y in zip(current[2],baseline[2],strict=True)
                        for a,b in zip(x,y,strict=True)),(group_name,name)
                stats=current[3]
                proposals=sum(s['guided_attention_calls'] for s in stats)
                kept=sum(s['accepted_guidance_calls'] for s in stats)
                if name.startswith(('reject_','live_')):
                    assert proposals>0
                    assert max(s['maximum_probe_hidden_difference'] for s in stats)==0
                if guided:
                    assert max(s['gradient_probe_calls'] for s in stats)==current[4]['backward']
                else:assert current[4]['backward']==0
                if name=='live_guided':assert kept>0
                report['cases'][group_name+'_'+name]=dict(exact_native_logits_tokens_current36_KV=exact,
                    proposals=proposals,kept=kept,counts=current[4],peak_allocated=current[5],
                    host_shared_hits=max(s['shared_static_hits'] for s in stats))
                if group_name=='B16':assert current[5]<29*2**30, 'Insufficient headroom for the512-token pilot'
                del controller,current
                gc.collect();torch.cuda.empty_cache()
        del shared,inputs,baseline
        gc.collect();torch.cuda.empty_cache()
    assert report['cases']['B2_reject_guided']['host_shared_hits']>0
    return report
