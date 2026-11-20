"""Matched signed attention budgets from absolute or relative Q/P differences."""

from contextlib import contextmanager
from unittest.mock import patch
import math
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen


def evidence_budget(evidence,preference,drift,gain,policy,strength=5.):
    assert policy in ['relative','absolute','shifted','opposite']
    assert evidence.shape==preference.shape and evidence.ndim==1
    assert 0<=drift<=1 and strength>=0
    if gain<=0 or drift==0 or strength==0:return torch.zeros_like(evidence)
    difference=evidence-preference
    if policy=='absolute':raw=difference
    else:raw=difference/(evidence+preference).clamp_min(torch.finfo(evidence.dtype).tiny)
    if policy=='shifted':raw=raw.roll(len(raw)//2)
    if policy=='opposite':raw=-raw
    span=(raw.max()-raw.min()).clamp_min(1e-6)
    return strength*drift*math.tanh(gain/2)*raw/span


def shift_visual_scores(scores,span,row,budget):
    assert scores.ndim==4 and scores.shape[2]==1
    assert budget.shape==(span[1]-span[0],)and torch.isfinite(budget).all()
    if not torch.count_nonzero(budget):return scores
    result=scores.float().clone();visual=result[row,:,0,span[0]:span[1]]
    changed=visual+budget[None]
    correction=changed.logsumexp(-1,keepdim=True)-visual.logsumexp(-1,keepdim=True)
    result[row,:,0,span[0]:span[1]]=changed-correction
    return result


@contextmanager
def evidence_attention(span,row,budget,layers):
    original=qwen.eager_attention_forward;calls=[];errors=[]
    active=bool(torch.count_nonzero(budget))
    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        index=getattr(module,'layer_idx',None)
        if not isinstance(module,qwen.Qwen3VLTextAttention)or index not in layers or not active:
            return original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        assert not dropout and not module.training and query.shape[-2]==1
        key=qwen.repeat_kv(key,module.num_key_value_groups);value=qwen.repeat_kv(value,module.num_key_value_groups)
        scores=(query@key.transpose(2,3))*scaling
        if attention_mask is not None:scores=scores+attention_mask[...,:key.shape[-2]]
        native=scores.softmax(-1,dtype=torch.float32)
        probability=shift_visual_scores(scores,span,row,budget).softmax(-1,dtype=torch.float32)
        before=native[row,:,0,span[0]:span[1]].sum(-1)
        after=probability[row,:,0,span[0]:span[1]].sum(-1)
        assert torch.allclose(after,before,rtol=1e-5,atol=1e-8)
        errors.append(float((after-before).abs().max()));calls.append(index)
        probability=probability.to(query.dtype)
        return(probability@value).transpose(1,2).contiguous(),probability
    with patch.object(qwen,'eager_attention_forward',attention):yield calls,errors
    assert calls==(list(layers)if active else [])
