"""Head association control and complete event logs, including terminal EOS."""

from unittest.mock import patch
import torch

from features import qwen3_block_acg_signal as block
from features.qwen3_acg_signal_feedback import head_scales


def rolled_head_scales(direction, gradient, drift, gamma):
    scales, _, effects=head_scales(direction,gradient,drift,gamma)
    scales=scales.roll(scales.shape[-1]//3,-1)
    return scales,(scales*effects).sum(-1),effects


def complete_logs(events,gates,decisions,tokens,eos_ids):
    result=[]
    for e in events:
        emitted=e['proposed_token'] if e['accepted'] else e['token']
        terminal=e['step']==len(tokens)
        assert e['step']<=len(tokens)
        if terminal:assert emitted in eos_ids
        else:assert tokens[e['step']]==emitted
        result.append(dict(e,emitted_token=emitted,terminal_eos=terminal))
    assert all(g['step']<=len(tokens) for g in gates)
    assert all(d['step']<=len(tokens) for d in decisions)
    return dict(full_joint_events=result,full_gate_events=list(gates),full_class_decisions=list(decisions),
        total_proposals=len(result),total_accepted=sum(e['accepted'] for e in result),
        terminal_eos_accepts=sum(e['accepted'] and e['terminal_eos'] for e in result),
        full_event_log_includes_terminal_eos=True)


class Qwen3BatchBlockACGLogged(block.Qwen3BatchBlockACG):
    def statistics(self,responses):
        result=super().statistics(responses)
        for row,(tokens,stat) in enumerate(zip(responses,result,strict=True)):
            stat.update(complete_logs(self.events.get(row,()),self.gate_events.get(row,()),
                self.class_decisions.get(row,()),tokens,self.eos_ids))
        return result


class Qwen3BatchBlockACGHeadRoll(Qwen3BatchBlockACGLogged):
    def _budget(self,source):
        with patch.object(block,'head_scales',rolled_head_scales):
            rows,budgets,details=super()._budget(source)
        for detail in details:detail['head_scale_roll_by']=32//3
        return rows,budgets,details

    def statistics(self,responses):
        result=super().statistics(responses)
        for stat in result:
            stat.update(permuted_head_control=True,head_scale_roll_by=32//3,
                acg_scope='Same native17-layer proposal and scale multiset; fixed head permutation H//3.')
        return result
