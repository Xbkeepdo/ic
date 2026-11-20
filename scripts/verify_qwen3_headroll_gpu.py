"""Check exact rollback for the new head association control."""

from contextlib import ExitStack
import gc
import torch
from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGHeadRoll
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper,probe,allowed,parser,threshold):
    (span,group),=same_span_batches(wrapper,[485532,558955],COCO_ROOT,16)
    inputs={k:torch.cat([d[k] for _,d in group]).to(wrapper.device) for k in group[0][1]}
    prompt=inputs['input_ids'].shape[1]
    eos=wrapper.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    def run(controller):
        logits,kv,encoders=[],[],[]
        def observe(_m,_a,out):
            logits.append(out.logits[:,-1].cpu().clone())
            cache=out.past_key_values
            assert all(cache.get_seq_length(i)==prompt+len(logits)-1 for i in range(36))
            kv.append(tuple(t[:,:,-1:].cpu().clone() for layer in cache.layers for t in (layer.keys,layer.values)))
        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            stack.callback(wrapper.model.model.visual.register_forward_pre_hook(lambda *_:encoders.append(1)).remove)
            output=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=32))
        tokens=[trim_eos(t,eos) for t in output[:,prompt:].tolist()]
        assert len(encoders)==1
        return tokens,logits,kv,controller.statistics(tokens)
    shared={};report=dict(passed=True,cases={})
    with torch.inference_mode():
        native=run(Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'))
        for name,reject in [('reject',True),('live',False)]:
            controller=Qwen3BatchBlockACGHeadRoll(wrapper.model,span,allowed,probe=probe,
                semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                real_threshold=threshold,shared_static=shared,signal_guided=True,
                diagnostic_force_probe_gate=True,diagnostic_reject_all=reject)
            current=run(controller)
            if reject:
                assert current[0]==native[0]
                assert all(torch.equal(a,b) for a,b in zip(current[1],native[1],strict=True))
                assert all(torch.equal(a,b) for x,y in zip(current[2],native[2],strict=True) for a,b in zip(x,y,strict=True))
            stats=current[3]
            assert sum(s['total_proposals'] for s in stats)>0
            assert max(s['maximum_probe_hidden_difference'] for s in stats)==0
            report['cases'][name]=dict(proposals=sum(s['total_proposals'] for s in stats),
                kept=sum(s['total_accepted'] for s in stats),terminal_EOS=sum(s['terminal_eos_accepts'] for s in stats),
                native_logits_tokens_current36_KV_exact=reject,encoder_calls=1)
            del controller,current
            gc.collect();torch.cuda.empty_cache()
    return report
