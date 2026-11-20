"""Use the original exact-risk EMD coupling to move visual attention."""

from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.dgst_t import (_cosine_distance_matrix,_renormalize,
                            _topk_union_indices,_wasserstein_1_exact)
from features.qwen3_region_memory import LAYERS


def row_kernel(plan):
    assert plan.ndim==2 and plan.shape[0]==plan.shape[1]
    assert torch.isfinite(plan).all()and plan.min()>=0
    total=plan.sum(1,keepdim=True)
    result=plan/total.clamp_min(torch.finfo(plan.dtype).tiny)
    result=torch.where(total>0,result,torch.eye(len(plan),dtype=plan.dtype,device=plan.device))
    return result


def original_risk_plan(visual_hpre,prediction_hpre,update,attention,log_s):
    source=F.cosine_similarity(update.float()[None],visual_hpre.float(),dim=-1).div(.03).softmax(0)
    mean=attention.float().mean(0);mean=mean/mean.sum().clamp_min(1e-30)
    target=mean*log_s;target=target/target.sum().clamp_min(1e-30)
    support=_topk_union_indices(source,target,32)
    a=_renormalize(source[support]).cpu();b=_renormalize(target[support]).cpu()
    cost=_cosine_distance_matrix(visual_hpre[support]).float().cpu()
    endpoint=(1-F.cosine_similarity(prediction_hpre.float()[None],visual_hpre.float(),dim=-1)[support].clamp(-1,1)).cpu()
    transport,plan=_wasserstein_1_exact(a,b,cost,solver='emd')
    penalty=float(np.dot(a.numpy(),endpoint.numpy())+np.dot(b.numpy(),endpoint.numpy()))
    risk=transport+penalty
    assert risk>=0 and abs(float(plan.sum())-1)<1e-7
    assert torch.allclose(plan.sum(1),a.double(),atol=1e-7,rtol=1e-7)
    assert torch.allclose(plan.sum(0),b.double(),atol=1e-7,rtol=1e-7)
    kernels=dict(plan=row_kernel(plan),independent=b.double()[None].expand(len(b),-1).clone(),
                 reverse=row_kernel(plan.T))
    kernels={n:k.to(device=visual_hpre.device,dtype=torch.float32)for n,k in kernels.items()}
    # Endpoint penalties have fixed marginals and thus do not change the
    # balanced optimal coupling. They affect the shared action magnitude.
    return support,kernels,dict(source=a.tolist(),target=b.tolist(),cost=cost.tolist(),
        endpoint=endpoint.tolist(),plan=plan.tolist(),transport=transport,penalty=penalty,risk=risk,
        amount=.5*risk/(1+risk),source_distribution='original P_FFN tau.03',
        target_distribution='original mean_attention * log_s, no added1 or write-norm factor')


def move_attention(probability,span,row,support,kernel,amount):
    assert probability.ndim==4 and probability.shape[2]==1 and 0<=amount<=.5
    assert kernel.shape==(len(support),len(support))
    assert torch.isfinite(kernel).all()and kernel.min()>=0
    assert torch.allclose(kernel.sum(1),torch.ones(len(support),device=kernel.device),atol=1e-6,rtol=1e-6)
    if amount==0:return probability
    positions=support+span[0];result=probability.clone()
    selected=probability[row,:,0,positions]
    changed=(1-amount)*selected+amount*(selected@kernel)
    result[row,:,0,positions]=changed
    assert torch.allclose(changed.sum(-1),selected.sum(-1),atol=1e-8,rtol=1e-5)
    return result


@contextmanager
def plan_attention(span,row,support,kernel,amount):
    original=qwen.eager_attention_forward;calls=[]
    def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
        index=getattr(module,'layer_idx',None)
        if not isinstance(module,qwen.Qwen3VLTextAttention)or index not in LAYERS or amount==0:
            return original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        assert not dropout and not module.training and query.shape[-2]==1
        key=qwen.repeat_kv(key,module.num_key_value_groups);value=qwen.repeat_kv(value,module.num_key_value_groups)
        scores=(query@key.transpose(2,3))*scaling
        if attention_mask is not None:scores=scores+attention_mask[...,:key.shape[-2]]
        probability=scores.softmax(-1,dtype=torch.float32)
        probability=move_attention(probability,span,row,support,kernel,amount).to(query.dtype)
        calls.append(index)
        return(probability@value).transpose(1,2).contiguous(),probability
    with patch.object(qwen,'eager_attention_forward',attention):yield calls
    assert calls==(list(LAYERS)if amount else [])
