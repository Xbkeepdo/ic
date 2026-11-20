"""Prospective SAR support budget for history/visual attention transport.

Pure tensor proposal, not a new decoder or a completed mitigation experiment.
S combines relative object-class support and the original uniform-vocabulary
semantic scale. Actual write norms weight the global reliability budget.
"""

import torch
from features.qwen3_probe_attention_recycle import normalize_nonnegative


def relative_class_support(probabilities,class_columns,vocab_size):
    assert probabilities.ndim==2 and class_columns and vocab_size>0
    assert torch.isfinite(probabilities).all() and probabilities.min()>=0
    target=probabilities[:,class_columns].amax(-1)
    peak=probabilities.amax(-1)
    relative=torch.where(peak>0,target/peak,torch.zeros_like(peak))
    evidence=float(vocab_size)*target
    absolute=evidence/(1+evidence) # 1-exp(-log1p(V*p_class))
    result=(relative*absolute).clamp(0,1)
    assert torch.isfinite(result).all()
    return result


def matched_permuted_removal(attention,eligible,support,cap,target_mass):
    """Permute source choice while matching each head's removal and rate cap."""
    raw=cap*(1-support.roll(support.shape[-1]//3,-1))[:,None]
    low,high=-cap.clone(),cap.clone()
    for _ in range(32):
        middle=(low+high)/2
        rates=torch.minimum((raw+middle).clamp_min(0),cap)*eligible
        mass=(attention*rates).sum(-1,keepdim=True)
        small=mass<target_mass
        low=torch.where(small,middle,low);high=torch.where(small,high,middle)
    rates=torch.minimum((raw+(low+high)/2).clamp_min(0),cap)*eligible
    removed=attention*rates
    return torch.where(target_mass>0,removed,torch.zeros_like(removed))


def semantic_donor_transport(actual,effect,preference,write_norms,support,*,start,end,prompt_length,mode='semantic'):
    if mode not in ('history','semantic','support_history','permuted'):raise ValueError(mode)
    assert actual.shape==effect.shape and 0<=start<end<=prompt_length<=actual.shape[-1]
    assert preference.shape==write_norms.shape==support.shape==(actual.shape[0],end-start)
    assert torch.isfinite(support).all() and support.min()>=0 and support.max()<=1
    visual=actual[:,:,start:end]
    negative=(-visual*effect[:,:,start:end]).clamp_min(0)
    valid=negative.amax(-1,keepdim=True)>torch.finfo(negative.dtype).tiny
    evidence=normalize_nonnegative(negative.sum(1))
    drift=(1-(evidence*preference).sqrt().sum(-1)).clamp(0,1)
    deficit=(evidence-preference).clamp_min(0)
    spatial=.25+.75*deficit/deficit.amax(-1,keepdim=True).clamp_min(1e-12)
    destination=normalize_nonnegative(normalize_nonnegative(negative)*spatial[:,None])
    unsupported=(normalize_nonnegative(write_norms)*(1-support)).sum(-1)
    unsupported=torch.where(write_norms.amax(-1)>0,unsupported,torch.ones_like(unsupported)).clamp(0,1)
    amount=.5*drift
    if mode!='history':amount=amount*unsupported
    history=actual[:,:,prompt_length:]
    movable=(effect[:,:,prompt_length:]>0)&valid
    removed_history=amount[:,None,None]*history*movable
    removed_visual=torch.zeros_like(visual)
    if mode in ('semantic','permuted'):
        eligible=(effect[:,:,start:end]>0)&valid
        removed_visual=amount[:,None,None]*visual*eligible*(1-support)[:,None]
        if mode=='permuted':
            removed_visual=matched_permuted_removal(visual,eligible,support,
                amount[:,None,None].expand(actual.shape[0],actual.shape[1],1),removed_visual.sum(-1,keepdim=True))
    history_mass=removed_history.sum(-1,keepdim=True)
    visual_mass=removed_visual.sum(-1,keepdim=True)
    delta=torch.zeros_like(actual)
    delta[:,:,start:end]=(history_mass+visual_mass)*destination-removed_visual
    delta[:,:,prompt_length:]-=removed_history
    predicted=(delta*effect).sum((1,2))
    assert torch.isfinite(delta).all() and delta.sum(-1).abs().max()<1e-6
    assert (actual+delta).min()>=-1e-7 and (predicted<=1e-6).all()
    return delta,dict(drift=drift,unsupported_fraction=unsupported,predicted_change=predicted,
        history_mass=history_mass.sum((1,2)),visual_mass=visual_mass.sum((1,2)),
        per_head_history_mass=history_mass,per_head_visual_mass=visual_mass,amount=amount)
