"""Check bounded evidence tilts, zero drift, support ranking and visual mass."""

from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_bounded_visual_tilt import bounded_coefficients,positive_visual_shift
from scripts.run_qwen3_signed_feedback import write


def verify():
    q=torch.tensor([[.8,.1,.1],[.2,.3,.5]])
    p=torch.tensor([[.1,.8,.1],[.2,.3,.5]])
    coefficient,drift=bounded_coefficients(q,p,'evidence')
    assert coefficient[0,0]>coefficient[0,2]>coefficient[0,1]>=0
    assert drift[1]==0 and torch.equal(coefficient[1],torch.zeros(3))
    fixed,_=bounded_coefficients(q,p,'fixed');uniform,_=bounded_coefficients(q,p,'drift')
    assert torch.equal(fixed,torch.full_like(q,.5))
    assert torch.equal(uniform,drift[:,None]*torch.full_like(q,.5))
    scores=torch.tensor([[[[-3.,2.,-1.,4.,-2.]]],[[[1.,-2.,3.,-4.,0.]]]])
    delta=positive_visual_shift(scores[:,:,:,1:4],coefficient,1.)
    assert (delta>=0).all() and (delta<=scores[:,:,:,1:4].abs()).all()
    tilted=scores.clone();tilted[:,:,:,1:4]+=delta
    mass_before=scores.softmax(-1)[:,:,:,1:4].sum(-1)
    mass_after=tilted.softmax(-1)[:,:,:,1:4].sum(-1)
    assert (mass_after>=mass_before).all() and mass_after[0]>mass_before[0]
    assert torch.equal(tilted[1],scores[1]) and torch.equal(tilted[:,:,:,[0,4]],scores[:,:,:,[0,4]])
    order=[1,0]
    reordered,_=bounded_coefficients(q[order],p[order],'evidence')
    assert torch.equal(reordered,coefficient[order])
    assert torch.equal(positive_visual_shift(scores[:,:,:,1:4],coefficient,0.),torch.zeros_like(delta))
    return dict(passed=True,evidence_over_ffn_ranking=True,aligned_QP_no_action=True,
        fixed_and_drift_controls_exact=True,bounded_nonnegative_visual_shift=True,
        visual_mass_increases_without_editing_nonvisual_logits=True,sparse_row_order_exact=True,
        scope='CPU algebra; native-context and GPU KV rollback checks still required.')


if __name__=='__main__':
    out=REPO/'outputs/qwen3_bounded_visual_tilt_cpu';out.mkdir(exist_ok=True)
    result=verify();write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))
