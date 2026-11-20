"""Veto head-wise suppression when the complete visual direction supports y."""

import math

from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
from features.qwen3_block_acg_signal import POST


def collective_effect(detail):
    """Unweighted directional derivative; independent of the FFN drift scale."""
    assert detail['signal_guided'] and set(detail['layer_details'])==set(map(str,POST))
    effects=[value for layer in POST for value in detail['layer_details'][str(layer)]['head_effects']]
    assert len(effects)==17*32 and all(math.isfinite(x) for x in effects)
    return math.fsum(effects)


def retain_collective_opposition(rows,budgets,details):
    assert len(rows)==len(details) and set(budgets)==set(POST)
    effects=[collective_effect(d) for d in details]
    keep=[i for i,effect in enumerate(effects) if effect<0]
    retained=[]
    for index in keep:
        retained.append(dict(details[index],collective_visual_effect=effects[index],
            collective_visual_support_veto=True))
    return [rows[i] for i in keep],{layer:b[keep] for layer,b in budgets.items()},retained,effects


class Qwen3BatchCollectiveVisualGate(Qwen3BatchBlockACGLogged):
    def __init__(self,*args,use_collective_veto=True,**kwargs):
        assert kwargs.get('signal_guided',True)
        kwargs['signal_guided']=True
        super().__init__(*args,**kwargs)
        self.use_collective_veto=bool(use_collective_veto)
        self.collective_decisions={}

    def _budget(self,source):
        rows,budgets,details=super()._budget(source)
        if not rows:return rows,budgets,details
        selected,filtered,retained,effects=retain_collective_opposition(rows,budgets,details)
        for row,detail,effect in zip(rows,details,effects,strict=True):
            self.collective_decisions.setdefault(row,[]).append(dict(step=self.step,token=detail['token'],
                collective_visual_effect=effect,veto=self.use_collective_veto and effect>=0,
                effective_before_veto=True))
        return (selected,filtered,retained) if self.use_collective_veto else (rows,budgets,details)

    def statistics(self,responses):
        result=super().statistics(responses)
        for row,(tokens,stat) in enumerate(zip(responses,result,strict=True)):
            decisions=self.collective_decisions.get(row,[])
            assert all(d['step']<=len(tokens) for d in decisions)
            stat.update(collective_visual_support_veto=self.use_collective_veto,
                full_collective_decisions=list(decisions),collective_vetoed_steps=sum(d['veto'] for d in decisions),
                collective_effect_scope='Sum of all17 native layers and32 heads before D/head selection; E>=0 vetoes the entire proposed replay.',
                collective_gate_is_first_order=True)
        return result
