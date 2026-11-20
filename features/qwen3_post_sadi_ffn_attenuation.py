"""Auxiliary source-FFN attenuation with post-SADI attention guidance."""

from contextlib import contextmanager
import torch
from features.qwen3_region_memory import RegionEvidenceReader
from features.qwen3_signal_preference_contrast import visual_preference


def attenuated_anchor(anchor, residual, update, row, amount):
    assert anchor.ndim == residual.ndim == 3 and anchor.shape == residual.shape
    assert update.shape == (anchor.shape[0],anchor.shape[-1]) and 0 <= amount <= .5
    assert anchor.dtype == residual.dtype == update.dtype
    assert torch.equal(residual+update[:,None],anchor)
    if amount == 0: return anchor
    result=anchor.clone()
    result[row,0]=residual[row,0]+update[row]*(1-amount)
    return result


class FFNResidualReader(RegionEvidenceReader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.ffn_residual=None

    def _residual(self,_module,args):
        if not self._ignore():self.ffn_residual=args[0][:,-1:].clone()

    @contextmanager
    def capture(self):
        with super().capture():
            handle=self.decoder.layers[19].post_attention_layernorm.register_forward_pre_hook(self._residual)
            try:yield self
            finally:handle.remove()


def replay_auxiliary(memory,reader,span,row,cache,reference,prior,attention_amount,ffn_amount):
    original=reader.anchor
    try:
        reader.anchor=attenuated_anchor(original,reader.ffn_residual,reader.current[19]['update'],row,ffn_amount)
        with visual_preference(span,row,prior,attention_amount)as calls:
            result=memory.replay(row,cache,reader,{},reference,zero=attention_amount==ffn_amount==0)
        return result,calls
    finally:
        reader.anchor=original


def configurations(evidence,drift,constant):
    assert 0 <= drift <= 1 and 0 <= constant <= .5
    amount=.5*drift
    return dict(attention=(evidence,amount,0.),ffn=(evidence,0.,amount),
                joint=(evidence,amount,amount),joint_constant=(evidence,amount,constant),
                joint_shifted=(evidence.roll(len(evidence)//2),amount,amount),
                ffn_constant=(evidence,0.,constant))
