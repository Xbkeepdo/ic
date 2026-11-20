"""Preserve the complete visual contrast; use SAR/FFN drift only for strength."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from features import qwen3_sar_ems_layer_feedback as feedback
from features.qwen3_svar_layer_feedback import svar_visual_logits
from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
from features.qwen3_block_acg_signal import POST
from features.qwen3_acg_signal_feedback import sar_ffn_drift


def scalar_rescale(budgets,details,drifts,gamma):
    """Keep each head direction; update all recorded coefficients and row order."""
    assert set(budgets)==set(drifts)==set(POST)
    count=len(details)
    result={}
    for layer in POST:
        drift=drifts[layer]
        assert drift.shape==(count,) and torch.isfinite(drift).all()
        assert (drift>=0).all() and (drift<=1).all()
        assert budgets[layer].shape[0]==count
        result[layer]=budgets[layer]*drift[:,None,None]
        for index,detail in enumerate(details):
            item=detail['layer_details'][str(layer)]
            scale=float(gamma*drift[index]);heads=len(item['head_scales'])
            item.update(drift=float(drift[index]),head_scales=[scale]*heads,
                maximum_head_scale=scale,nonzero_heads=heads if scale>0 else 0,
                budget_max=float(result[layer][index].abs().max()))
    maximum=torch.stack([b.abs().amax((1,2)) for b in result.values()]).amax(0)
    for index,detail in enumerate(details):
        detail.update(drift=sum(float(drifts[k][index]) for k in POST)/len(POST),
            budget_max=float(maximum[index]),uses_scalar_sar_ffn_drift=True,
            head_gradient_guidance=False)
    keep=(maximum>0).nonzero().flatten().tolist()
    return keep,{k:b[keep] for k,b in result.items()},[details[i] for i in keep]


class Qwen3BatchScalarVisualFeedback(Qwen3BatchBlockACGLogged):
    def __init__(self,*args,base_variant='svar',use_drift=True,**kwargs):
        assert base_variant in ('sadi','svar')
        assert not kwargs.pop('signal_guided',False),'This policy never selects negative-gradient heads.'
        super().__init__(*args,signal_guided=False,**kwargs)
        self.base_variant=base_variant;self.use_drift=bool(use_drift)

    def _budget(self,source):
        rows,budgets,details=super()._budget(source)
        for detail in details:
            detail.update(base_variant=self.base_variant,uses_scalar_sar_ffn_drift=self.use_drift,
                head_gradient_guidance=False)
        if not rows or not self.use_drift:return rows,budgets,details
        drifts={}
        for layer in POST:
            drifts[layer]=torch.stack([sar_ffn_drift(self.reader.write_norms[layer,row],
                self.reader._static(layer,row)['log_s'][:,self.reader.columns[int(self.final_tokens[row])]],
                self.reader.visual[layer][row],self.scratch[layer]['update'][row]) for row in rows])
        keep,budgets,details=scalar_rescale(budgets,details,drifts,self.gamma)
        return [rows[i] for i in keep],budgets,details

    @contextmanager
    def capture(self):
        if self.base_variant=='svar':
            with patch.object(feedback,'sadi_visual_logits',svar_visual_logits):
                with super().capture():yield self
        else:
            with super().capture():yield self

    def statistics(self,responses):
        result=super().statistics(responses)
        for stat in result:
            stat.update(base_variant=self.base_variant,uses_scalar_sar_ffn_drift=self.use_drift,
                head_gradient_guidance=False,gradient_sources=[],one_backward_for_all_layers=False,
                legacy_signal_guided_flag='False denotes no head-gradient branch; scalar drift is recorded separately.',
                acg_scope='Complete base5--18, all-head joint19--35 contrast; optional gamma1.4*D, no backward.')
        return result
