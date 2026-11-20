"""Conflict-B text-route gates, with a separate SAR/FFN head-ranking control.

The route decomposition and VRI ranking follow CRG's published idea:
https://arxiv.org/html/2605.24024v1
Fixed k=8 and linear gates [0,.5] are this project's prospective settings.
"""

import torch


def split_head_routes(scores, values, span):
    """Joint-softmax visual/text routes; preserve original context for identity."""
    assert scores.ndim==values.ndim==4 and scores.shape[-2]==1
    start,end=span
    probability=scores.softmax(-1,dtype=torch.float32).to(values.dtype)
    context=(probability@values)[:, :, 0]
    visual=(probability[:,:,:,start:end]@values[:,:,start:end])[:,:,0]
    nonvisual_probability=torch.cat((probability[:,:,:,:start],probability[:,:,:,end:]),-1)
    nonvisual_values=torch.cat((values[:,:,:start],values[:,:,end:]),-2)
    textual=(nonvisual_probability@nonvisual_values)[:,:,0]
    return context,visual,textual,probability


def preference_over_evidence(conditional_visual, evidence, preference):
    """Per-head mass on positions preferred by FFN more than target evidence."""
    assert conditional_visual.ndim==3 and evidence.shape==preference.shape
    assert evidence.shape==(conditional_visual.shape[0],conditional_visual.shape[-1])
    q,p=evidence.float(),preference.float()
    total=q+p
    ratio=torch.where(total>0,p/total.clamp_min(torch.finfo(torch.float32).tiny),.5)
    risk=(conditional_visual.float()*ratio[:,None]).sum(-1)
    return torch.where((q==p).all(-1,keepdim=True),torch.full_like(risk,.5),risk)


def conflict_text_gates(visual_effect, textual_effect, head_risk, *, mode, k=8,
                        minimum_gate=0., maximum_gate=.5):
    """Match gate multisets; change only ranking within Conflict-B heads.

    Base ranks by low VRI. Signal ranks by high FFN/evidence risk, breaking
    ties with the same VRI order. Roll permutes only the signal, retaining
    the identical native conflict membership and gate multiset.
    """
    assert mode in ('vri','signal','roll') and k>0
    assert visual_effect.shape==textual_effect.shape==head_risk.shape
    assert 0<=minimum_gate<=maximum_gate<1
    vis,txt=visual_effect.float(),textual_effect.float()
    assert all(torch.isfinite(t).all() for t in (vis,txt,head_risk))
    assert (head_risk>=-1e-6).all() and (head_risk<=1+1e-6).all()
    conflict=(vis<0)&(txt>0)
    vri=vis.abs()/(vis.abs()+txt.abs()+1e-8)
    risk=head_risk.float().roll(head_risk.shape[-1]//2,-1) if mode=='roll' else head_risk.float()
    gates=torch.ones_like(vis);selected=torch.zeros_like(conflict)
    for row in range(len(vis)):
        heads=conflict[row].nonzero().flatten()
        # Stable VRI sorting supplies a deterministic secondary order.
        heads=heads[torch.argsort(vri[row,heads],stable=True)]
        if mode!='vri':
            heads=heads[torch.argsort(risk[row,heads],descending=True,stable=True)]
        heads=heads[:k]
        if len(heads)==0:continue
        schedule=torch.linspace(minimum_gate,maximum_gate,len(heads),device=vis.device) if len(heads)>1 else vis.new_tensor([minimum_gate])
        gates[row,heads]=schedule;selected[row,heads]=True
    predicted=((gates-1)*txt).sum(-1)
    assert (predicted<=1e-7).all() and torch.equal(gates[~conflict],torch.ones_like(gates[~conflict]))
    return gates,dict(conflict=conflict,vri=vri,applied_head_risk=risk,selected=selected,
                      predicted_log_probability_change=predicted)


def apply_text_gates(native_context, textual_route, gates):
    """No attention renormalization; remove only the selected text contribution."""
    assert native_context.shape==textual_route.shape and gates.shape==native_context.shape[:2]
    assert torch.isfinite(gates).all() and (gates>=0).all() and (gates<=1).all()
    return (native_context.float()+(gates-1)[...,None]*textual_route.float()).to(native_context.dtype)
