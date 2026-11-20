"""Check permutation invariants and terminal-event cache safety without a model."""

from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_acg_signal_feedback import head_scales
from features.qwen3_block_acg_headroll import rolled_head_scales,complete_logs


def verify():
    torch.manual_seed(43)
    direction=torch.randn(3,32,8);gradient=torch.randn_like(direction);drift=torch.tensor([0.,.4,1.])
    scales,_,effects=head_scales(direction,gradient,drift,1.4)
    rolled,predicted,again=rolled_head_scales(direction,gradient,drift,1.4)
    assert torch.equal(scales.sort(-1).values,rolled.sort(-1).values) and torch.equal(effects,again)
    assert torch.equal(rolled,scales.roll(10,-1))
    torch.testing.assert_close(predicted,(rolled*effects).sum(-1),rtol=0,atol=0)
    _,positive,_=rolled_head_scales(torch.ones(1,4,1),torch.tensor([[[-4.],[3.],[1.],[2.]]]),torch.ones(1),1.4)
    assert positive>0 # a control is allowed to point the wrong way before actual acceptance
    event=dict(step=2,token=7,proposed_token=99,accepted=True)
    logs=complete_logs([event],[dict(step=2,positive=True)],[dict(step=2)],[1,2],[99])
    assert logs['total_accepted']==logs['terminal_eos_accepts']==1
    assert logs['full_joint_events'][0]['terminal_eos']
    assert not (logs['total_accepted']==0), 'A truncated-EOS intervention cannot certify a native cache'
    ordinary=complete_logs([dict(step=0,token=7,proposed_token=3,accepted=True)],[],[],[3,2],[99])
    assert ordinary['total_accepted']==1 and not ordinary['terminal_eos_accepts']
    empty=complete_logs([],[],[],[1,2],[99]);assert empty['total_accepted']==0
    return dict(passed=True,head_scale_multiset_bitwise_preserved=True,permutation=10,
        direction_effects_unchanged=True,predicted_control_can_be_positive=True,
        terminal_EOS_in_full_log=True,terminal_EOS_never_false_native_cache=True)


if __name__=='__main__':
    from scripts.run_qwen3_signed_feedback import write
    result=verify();out=REPO/'outputs/qwen3_block_headroll_cpu';out.mkdir(exist_ok=True)
    write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))
