"""Check complete-direction signs, native budgets and sparse-row alignment."""

import copy
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_collective_visual_gate import POST,collective_effect,retain_collective_opposition
from scripts.run_qwen3_signed_feedback import write


def verify():
    # A positive collective visual effect can contain negative heads. Those
    # heads must not alone authorize suppressing the current word.
    vectors=[[2.,-1.]+[0.]*30,[-2.,1.]+[0.]*30,[0.]*32]
    details=[dict(signal_guided=True,token=token,
        layer_details={str(k):dict(head_effects=list(vector),drift=.7) for k in POST})
        for token,vector in zip([7,11,19],vectors,strict=True)]
    budgets={k:torch.arange(3*32*4).reshape(3,32,4).float()+k for k in POST}
    original=copy.deepcopy(details)
    rows,result,retained,effects=retain_collective_opposition([9,4,12],budgets,details)
    assert effects==[17.,-17.,0.] and rows==[4] and retained[0]['token']==11
    assert all(torch.equal(result[k],budgets[k][[1]]) for k in POST)
    assert details==original
    order=[2,1,0]
    rows2,reordered,info2,effects2=retain_collective_opposition([12,4,9],
        {k:b[order] for k,b in budgets.items()},[details[i] for i in order])
    assert rows2==rows and info2==retained and effects2==[0.,-17.,17.]
    assert all(torch.equal(reordered[k],result[k]) for k in POST)
    for d in details:
        for item in d['layer_details'].values():item['drift']=0.
    assert [collective_effect(d) for d in details]==effects
    return dict(passed=True,positive_collective_effect_vetoes_negative_head_selection=True,
        zero_effect_no_action=True,negative_effect_keeps_original_budget_exact=True,
        sparse_reordered_rows_exact=True,independent_of_ffn_drift=True,
        scope='CPU algebra only; GPU rollback, activity and generation effect remain untested.')


if __name__=='__main__':
    out=REPO/'outputs/qwen3_collective_visual_cpu';out.mkdir(exist_ok=True)
    result=verify();write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))
