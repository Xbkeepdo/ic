"""Bounded positive visual-logit tilts from target evidence and FFN preference."""

from contextlib import contextmanager,ExitStack

import torch
from torch.nn import functional as F

from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
from features.qwen3_block_acg_signal import POST
from features.qwen3_acg_signal_feedback import normalize_nonnegative


def evidence_preference(write_norms,log_s,visual_hpre,ffn_update):
    q=normalize_nonnegative(write_norms*(1+log_s))
    p=F.cosine_similarity(visual_hpre.float(),ffn_update.float()[None],dim=-1).div(.03).softmax(-1)
    return q,p


def bounded_coefficients(q,p,mode):
    assert mode in ('fixed','drift','evidence') and q.shape==p.shape
    assert torch.isfinite(q).all() and torch.isfinite(p).all() and (q>=0).all() and (p>=0).all()
    drift=(1-(q*p).sqrt().sum(-1)).clamp(0,1)
    if mode=='fixed':return torch.full_like(q,.5),torch.ones_like(drift)
    weight=torch.full_like(q,.5) if mode=='drift' else q/(q+p).clamp_min(torch.finfo(q.dtype).tiny)
    coefficient=drift[...,None]*weight
    assert (coefficient>=0).all() and (coefficient<=1).all()
    return coefficient,drift


def positive_visual_shift(visual_scores,coefficient,scale):
    assert visual_scores.shape[0]==coefficient.shape[0] and visual_scores.shape[-1]==coefficient.shape[-1]
    assert scale>=0 and torch.isfinite(coefficient).all()
    shift=visual_scores.float().abs()*coefficient[:,None,None,:]*float(scale)
    assert torch.isfinite(shift).all() and (shift>=0).all()
    return shift


def original_sam_rows(owner):
    """Keep the frozen108 gate and prefix-class filtering before a new operator."""
    rows=[]
    for row,(token,active) in enumerate(zip(owner.final_tokens.tolist(),owner.active.tolist(),strict=True)):
        if not active or token not in owner.nouns or token in owner.seen[row]:continue
        prefix=owner.tokenizer.decode(owner.output_history[row],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        assert len(owner.output_history[row])==owner.step
        prior,canonical=owner.history_parser.classes(prefix),owner.token_classes[token]
        blocked=canonical in prior
        owner.class_decisions.setdefault(row,[]).append(dict(step=owner.step,token=token,
            canonical=canonical,prior_classes=sorted(prior),blocked=blocked))
        if not blocked or owner.force_probe_gate:rows.append(row)
    if not rows:return [],None
    real=owner.probe(owner.reader.features(rows,owner.final_tokens[rows].tolist()))
    for row,probability in zip(rows,real.tolist(),strict=True):
        owner.probe_scores.setdefault(row,[]).append(probability)
        owner.gate_events.setdefault(row,[]).append(dict(step=owner.step,token=int(owner.final_tokens[row]),
            p_real=probability,threshold=owner.real_threshold,positive=probability<owner.real_threshold))
    chosen=list(range(len(rows))) if owner.force_probe_gate else (real<owner.real_threshold).nonzero().flatten().tolist()
    return [rows[i] for i in chosen],real[chosen]


class Qwen3BatchBoundedVisualTilt(Qwen3BatchBlockACGLogged):
    def __init__(self,*args,tilt_mode='evidence',gamma=1.,diagnostic_zero_tilt=False,**kwargs):
        assert tilt_mode in ('fixed','drift','evidence')
        assert not kwargs.pop('signal_guided',False)
        super().__init__(*args,signal_guided=False,gamma=gamma,**kwargs)
        self.tilt_mode=tilt_mode;self.zero_tilt=bool(diagnostic_zero_tilt)
        self.native_context={};self.native_context_captures=0;self.zero_tilt_exact_checks=0

    def _context_hook(self,index):
        def remember(_module,args):
            if self.step>0 and not self._replaying:
                value=args[0];heads=self.scratch[index]['values'].shape[1]
                assert value.shape[1]==1
                self.native_context[index]=value[:,0].reshape(value.shape[0],heads,-1).detach()
                self.native_context_captures+=1
        return remember

    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            stack.enter_context(super().capture())
            if self.enabled:
                for index in POST:
                    stack.callback(self.decoder.layers[index].self_attn.o_proj.register_forward_pre_hook(self._context_hook(index)).remove)
            yield self

    def _budget(self,_source):
        rows,real=original_sam_rows(self)
        if not rows or not self.gamma:return [],None,[]
        assert set(self.native_context)==set(POST)
        self.first_context=self.native_context[19]
        budgets={};layers={}
        for index in POST:
            state=self.scratch[index]
            if self.tilt_mode=='fixed':
                coefficient=state['scores'].new_full((len(rows),self.end-self.start),.5,dtype=torch.float32)
                drift=coefficient.new_ones(len(rows))
            else:
                distributions=[evidence_preference(self.reader.write_norms[index,row],
                    self.reader._static(index,row)['log_s'][:,self.reader.columns[int(self.final_tokens[row])]],
                    self.reader.visual[index][row],state['update'][row]) for row in rows]
                q,p=(torch.stack([pair[i] for pair in distributions]) for i in (0,1))
                coefficient,drift=bounded_coefficients(q,p,self.tilt_mode)
            if self.zero_tilt:coefficient=torch.zeros_like(coefficient)
            scores=state['scores'].float().clone()
            shift=positive_visual_shift(scores[rows,:,:,self.start:self.end],coefficient,self.gamma)
            selected=scores[rows]
            selected[:,:,:,self.start:self.end]+=shift
            scores[rows]=selected
            # Preserve the original B/H/query dimensions, including a zero tilt.
            probability=scores.softmax(-1,dtype=torch.float32).to(state['values'].dtype)
            context=(probability@state['values']).squeeze(2)
            if self.zero_tilt:
                assert torch.equal(context,self.native_context[index])
                self.zero_tilt_exact_checks+=1
            budgets[index]=(context.float()-self.native_context[index].float())[rows]
            layers[index]=[dict(drift=float(drift[i]),budget_max=float(budgets[index][i].abs().max()),
                coefficient_mean=float(coefficient[i].mean()),coefficient_min=float(coefficient[i].min()),
                coefficient_max=float(coefficient[i].max()),visual_logit_shift_min=float(shift[i].min()),
                visual_logit_shift_max=float(shift[i].max()),bounded_positive_visual_tilt=True)
                for i in range(len(rows))]
        maximum=torch.stack([b.abs().amax((1,2)) for b in budgets.values()]).amax(0)
        effective=(maximum>0).nonzero().flatten().tolist()
        details=[dict(token=int(self.final_tokens[rows[i]]),gain=0.,
            drift=sum(layers[k][i]['drift'] for k in POST)/len(POST),budget_max=float(maximum[i]),
            probe_p_real=float(real[i]),probe_real_threshold=self.real_threshold,signal_guided=False,
            joint_layers=list(POST),layer_details={str(k):layers[k][i] for k in POST},
            bounded_visual_tilt=True,tilt_mode=self.tilt_mode) for i in effective]
        return [rows[i] for i in effective],{k:b[effective] for k,b in budgets.items()},details

    def _finish_decoder(self,module,args,output):
        try:return super()._finish_decoder(module,args,output)
        finally:self.native_context.clear()

    def statistics(self,responses):
        result=super().statistics(responses)
        for stat in result:
            stat.update(bounded_visual_tilt=True,tilt_mode=self.tilt_mode,tilt_scale=self.gamma,
                transport_mode='bounded_positive_visual_logit_tilt',gradient_sources=[],one_backward_for_all_layers=False,
                native_context_captures=self.native_context_captures,zero_tilt_exact_checks=self.zero_tilt_exact_checks,
                diagnostic_zero_tilt=self.zero_tilt,
                acg_scope='Shared joint19--35 replay; budget is a bounded visual-logit tilt, not orthogonal ACG.',
                borrowed_operator='CAI equation5 form; original108 gate and Q/(Q+P),D are this project adaptation.')
        return result
