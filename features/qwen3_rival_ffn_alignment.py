"""Auxiliary FFN direction step toward evidence-conditioned visual preference."""

from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import torch
from torch.nn import functional as F
from features.qwen3_supported_rival_guidance import Qwen3SupportedRivalGuidance
from features.qwen3_region_consistency import RegionConsistency


def preference_objective(visual,update,evidence,tau=.03):
    unit=F.normalize(update,dim=-1)
    probability=(visual@unit/tau).softmax(0)
    overlap=(evidence*probability).sqrt()
    coefficient=overlap.sum()
    # Derivative w.r.t. the unit update direction, then its tangent part.
    gradient=visual.T@(.5*(coefficient*probability-overlap))/tau
    tangent=gradient-unit*(unit*gradient).sum()
    return (1-coefficient).clamp(0,1),tangent,probability


def align_update(visual_hpre,update,evidence,strength):
    assert strength>=0 and torch.isfinite(update).all()
    dtype=torch.float64 if update.dtype==torch.float64 else torch.float32
    visual=F.normalize(visual_hpre.to(dtype),dim=-1);u=update.to(dtype);q=evidence.to(dtype)
    q=q/q.sum().clamp_min(1e-30)
    before,tangent,_=preference_objective(visual,u,q)
    step=min(float(strength),1.)
    for attempt in range(4):
        direction=F.normalize(F.normalize(u,dim=-1)-step*tangent,dim=-1)
        proposed=(u.norm()*direction).to(update.dtype)
        after,_,_=preference_objective(visual,proposed.to(dtype),q)
        if strength and float(after)<float(before):
            return proposed,dict(applied=True,before=float(before),after=float(after),step=step,
                attempts=attempt+1,gradient_norm=float(tangent.norm()),
                relative_update_change=float((proposed.to(dtype)-u).norm()/u.norm().clamp_min(1e-30)),
                relative_norm_change=float((proposed.to(dtype).norm()-u.norm()).abs()/u.norm().clamp_min(1e-30)))
        step*=.5
    return update,dict(applied=False,before=float(before),after=float(before),step=0.,attempts=4,
                       gradient_norm=float(tangent.norm()),relative_update_change=0.,relative_norm_change=0.)


class Qwen3RivalFFNAlignment(Qwen3SupportedRivalGuidance):
    def __init__(self,*args,policy='aux',**kwargs):
        assert policy in ['aux','aux_shifted','calibrated']
        super().__init__(*args,policy='calibrated',**kwargs)
        self.ffn_shifted=policy=='aux_shifted';self.policy='aux_shifted' if self.ffn_shifted else 'aux'
        self.ffn_residual=None;self.ffn_events={}

    @contextmanager
    def capture(self):
        original=RegionConsistency.attenuate
        def residual(_module,args):
            if not self.suspended:self.ffn_residual=args[0][:,-1:].clone()
        def replay(memory,row,cache,reader,rect,strength,reference,zero=False):
            if memory is not self.memory or zero or not strength:
                return original(memory,row,cache,reader,rect,strength,reference,zero=zero)
            choice=self.rival_choices[row][-1];assert choice['step']==self.step
            target,rival=choice['target'],choice['rival']
            a,ma=self.class_maps[row,target];b,mb=self.class_maps[row,rival]
            heat=(b/max(mb,1e-30)-a/max(ma,1e-30)).clamp_min(0)*reader.last_write_norms[row]
            evidence=heat/heat.sum().clamp_min(1e-30)
            if self.ffn_shifted:evidence=evidence.roll(len(evidence)//2)
            update=reader.current[19]['update'][row]
            old_anchor=self.anchor
            assert torch.equal(self.ffn_residual+reader.current[19]['update'][:,None],old_anchor)
            new_update,details=align_update(reader.visual[19][row],update,evidence,abs(strength))
            details.update(step=self.step,mode='shifted_evidence' if self.ffn_shifted else 'evidence',
                           extra_mlp_or_model_calls=0,autograd_calls=0)
            self.ffn_events.setdefault(row,[]).append(details)
            if details['applied']:
                self.anchor=old_anchor.clone()
                self.anchor[row,0]=(self.ffn_residual[row,0]+new_update).to(old_anchor.dtype)
            try:return original(memory,row,cache,reader,rect,strength,reference,zero=zero)
            finally:self.anchor=old_anchor
        with ExitStack() as stack:
            stack.enter_context(super().capture())
            stack.callback(self.decoder.layers[19].post_attention_layernorm.register_forward_pre_hook(residual).remove)
            stack.enter_context(patch.object(RegionConsistency,'attenuate',replay))
            yield self

    def statistics(self,responses):
        stats=super().statistics(responses)
        for row,stat in enumerate(stats):
            events={e['step']:e for e in self.ffn_events.get(row,[])}
            for e in stat['full_events']:
                if e['step'] in events:e['ffn_alignment']=events[e['step']]
            stat.update(ffn_alignment_events=list(events.values()),attention_operator_unchanged=True,
                        ffn_edit_layer=19,ffn_alignment_backtracking_max=4,ffn_norm_preserved_before_bf16_rounding=True)
        return stats

    def close(self):
        self.ffn_residual=None
        super().close()
