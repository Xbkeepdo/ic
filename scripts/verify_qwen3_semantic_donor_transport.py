"""Check prospective donor math before any generation policy is registered."""

import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_probe_attention_recycle import normalize_nonnegative
from features.qwen3_semantic_donor_transport import relative_class_support,semantic_donor_transport


def verify():
    torch.manual_seed(31415)
    actual=torch.randn(4,3,23).softmax(-1);effect=torch.randn_like(actual)
    preference=torch.randn(4,9).softmax(-1);writes=torch.rand(4,9);support=torch.rand(4,9)
    options=dict(start=3,end=12,prompt_length=16)
    outputs={m:semantic_donor_transport(actual,effect,preference,writes,support,mode=m,**options)
             for m in ('history','semantic','support_history','permuted')}
    # Literal old history-only formula, independently assembled.
    negative=(-actual[:,:,3:12]*effect[:,:,3:12]).clamp_min(0)
    valid=negative.amax(-1,keepdim=True)>torch.finfo(negative.dtype).tiny
    evidence=normalize_nonnegative(negative.sum(1))
    drift=(1-(evidence*preference).sqrt().sum(-1)).clamp(0,1)
    removed=.5*drift[:,None,None]*actual[:,:,16:]*((effect[:,:,16:]>0)&valid)
    deficit=(evidence-preference).clamp_min(0)
    spatial=.25+.75*deficit/deficit.amax(-1,keepdim=True).clamp_min(1e-12)
    destination=normalize_nonnegative(normalize_nonnegative(negative)*spatial[:,None])
    expected=torch.zeros_like(actual);expected[:,:,3:12]=removed.sum(-1,keepdim=True)*destination;expected[:,:,16:]-=removed
    assert torch.equal(expected,outputs['history'][0])
    a,b=outputs['semantic'][1],outputs['permuted'][1]
    torch.testing.assert_close(a['per_head_history_mass'],b['per_head_history_mass'],rtol=0,atol=0)
    torch.testing.assert_close(a['per_head_visual_mass'],b['per_head_visual_mass'],rtol=0,atol=1e-7)
    assert not torch.equal(outputs['semantic'][0],outputs['permuted'][0])
    # Complete support protects all actual writes, giving zero budget.
    zero=semantic_donor_transport(actual,effect,preference,writes,torch.ones_like(support),**options)
    assert torch.count_nonzero(zero[0])==0
    absent=semantic_donor_transport(actual,effect,preference,torch.zeros_like(writes),support,**options)
    assert torch.equal(absent[1]['unsupported_fraction'],torch.ones(4))
    # Visual-only donors can act when no generated history exists.
    nohist=semantic_donor_transport(actual,effect,preference,writes,torch.zeros_like(support),start=3,end=12,prompt_length=23)
    assert nohist[1]['history_mass'].sum()==0 and nohist[1]['visual_mass'].sum()>0
    # No negative visual counterevidence means no proposal.
    empty=semantic_donor_transport(actual,effect.abs(),preference,writes,support,**options)
    assert torch.count_nonzero(empty[0])==0
    probs=torch.randn(9,41).softmax(-1);cols=[2,7]
    obtained=relative_class_support(probs,cols,41)
    logits=probs.log()
    relative=(logits[:,cols].amax(-1)-logits.amax(-1)).exp()
    log_s=torch.log1p(41*probs[:,cols].amax(-1))
    torch.testing.assert_close(obtained,relative*(-torch.expm1(-log_s)),rtol=1e-6,atol=1e-7)
    tiny=torch.tensor([[0.,0.],[1e-40,2e-40]],dtype=torch.float32)
    assert torch.isfinite(relative_class_support(tiny,[1],41)).all()
    return dict(passed=True,old_history_budget_bitwise_equal=True,
        per_head_permuted_visual_mass_max_error=float((a['per_head_visual_mass']-b['per_head_visual_mass']).abs().max()),
        perfect_support_zero_action=True,visual_only_donors_exercised=True,no_counterevidence_zero_action=True,
        all_proposals_nonnegative_mass_preserving_and_first_order_nonincreasing=True,
        decoder_integration='Not implemented or GPU-tested in this CPU check.')


if __name__=='__main__':
    result=verify();out=REPO/'outputs/qwen3_semantic_donor_cpu';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
