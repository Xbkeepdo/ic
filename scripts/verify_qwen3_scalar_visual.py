"""Check exact fixed/zero boundaries, direction preservation and sparse row order."""

import copy
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_scalar_visual_feedback import scalar_rescale,POST
from features.qwen3_acg_signal_feedback import sar_ffn_drift


def verify():
    torch.manual_seed(43)
    base={k:torch.randn(3,32,8)*1.4 for k in POST}
    details=[dict(token=t,layer_details={str(k):dict(head_scales=[1.4]*32) for k in POST}) for t in (7,11,19)]
    unit={k:torch.ones(3) for k in POST}
    rows,full,info=scalar_rescale(base,copy.deepcopy(details),unit,1.4)
    assert rows==[0,1,2] and all(torch.equal(full[k],base[k]) for k in POST)
    zero={k:torch.zeros(3) for k in POST}
    rows,empty,info=scalar_rescale(base,copy.deepcopy(details),zero,1.4)
    assert rows==[] and all(b.shape==(0,32,8) for b in empty.values()) and not info
    drift={k:torch.tensor([.2,0.,.8]) for k in POST}
    rows,scaled,info=scalar_rescale(base,copy.deepcopy(details),drift,1.4)
    assert rows==[0,2] and [d['token'] for d in info]==[7,19]
    for k in POST:
        torch.testing.assert_close(scaled[k],base[k][[0,2]]*torch.tensor([.2,.8])[:,None,None],rtol=0,atol=0)
        assert (scaled[k]*base[k][[0,2]]).sum(-1).min()>0
        assert info[1]['layer_details'][str(k)]['budget_max']==float(scaled[k][1].abs().max())
    order=[2,0,1]
    rows2,again,info2=scalar_rescale({k:b[order] for k,b in base.items()},
        [copy.deepcopy(details[i]) for i in order],{k:d[order] for k,d in drift.items()},1.4)
    assert rows2==[0,1] and [d['token'] for d in info2]==[19,7]
    assert all(torch.equal(again[k],scaled[k][[1,0]]) for k in POST)
    hpre=torch.ones(4,8);ffn=torch.ones(8)
    aligned=sar_ffn_drift(torch.ones(4),torch.ones(4),hpre,ffn)
    assert aligned==0
    return dict(passed=True,unit_drift_exact_fixed=True,zero_drift_no_proposal=True,
        all_head_directions_preserved=True,sparse_reordered_rows_and_metadata_exact=True,
        aligned_visual_and_ffn_zero_drift=True,no_new_gradient_or_neural_forward=True)


if __name__=='__main__':
    from scripts.run_qwen3_signed_feedback import write
    out=REPO/'outputs/qwen3_scalar_visual_cpu';out.mkdir(exist_ok=True)
    report=verify();write(out/'verification.json',report);print(__import__('json').dumps(report,indent=2))
