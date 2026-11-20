"""Independent softmax, image independence and semantic-direction CPU checks."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from features.qwen3_evidence_head_routing import evidence_head_scores, head_routing_attention

ROOT=REPO/'outputs/qwen3_evidence_head_routing_cpu_20261119'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    assert not ROOT.exists()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.manual_seed(43)
    source={str(REPO/n):sha(REPO/n) for n in ('features/qwen3_evidence_head_routing.py',
        'scripts/verify_qwen3_evidence_head_routing.py')}
    ROOT.mkdir()
    for name in source:
        original=Path(name);target=ROOT/'source'/original.relative_to(REPO)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    (ROOT/'request.json').write_text(json.dumps(dict(created=time.time(),source_sha256=source,
        source_formula='Q/P head affinity gap, centered/max-scaled, times Hellinger2; fixed gamma1.',
        reference='.5*mean_abs_visual_logit per head; constant visual bias.',
        matching='Image-wise mean across heads of visual attention mass,32 bisections.',
        scope='CPU math only; no decoder or mitigation outcome.'),indent=2)+'\n')
    scores=torch.randn(4,6,1,19)*4
    values=torch.randn(4,6,19,7);span=(3,14);rows=[0,2]
    q=torch.randn(2,11).softmax(-1);p=torch.randn(2,11).softmax(-1)
    saved=scores.clone();saved_values=values.clone()
    context,probability,d=head_routing_attention(scores,values,span,rows,q,p)
    independent=scores.float().clone()
    independent[rows,:,0,3:14]+=d['visual_bias'][...,None]
    reference=independent.softmax(-1)
    assert torch.equal(probability,reference)
    assert torch.equal(context,(reference@values).transpose(1,2).contiguous())
    uniform=scores.clone()
    uniform[rows,:,0,3:14]+=.5*scores[rows,:,0,3:14].abs().mean(-1)[...,None]
    uniform_probability=uniform.softmax(-1)
    torch.testing.assert_close(probability[rows,:,:,3:14].sum(-1).mean(1),
        uniform_probability[rows,:,:,3:14].sum(-1).mean(1),atol=32*2**-23,rtol=0)
    assert torch.equal(probability[[1,3]],scores.softmax(-1)[[1,3]])
    assert torch.equal(scores,saved) and torch.equal(values,saved_values)
    for mode,flat in [('uniform',False),('evidence',True),('roll',True)]:
        _,result,_=head_routing_attention(scores,values,span,rows,q,p,mode=mode,flat_score=flat)
        assert torch.equal(result,uniform_probability)
    _,zero,_=head_routing_attention(scores,values,span,rows,q,p,zero_tilt=True)
    assert torch.equal(zero,scores.softmax(-1))
    _,same,diag=head_routing_attention(scores,values,span,rows,q,q)
    assert torch.equal(same,uniform_probability) and torch.count_nonzero(diag['raw_head_score'])==0
    reversed_context,reversed_prob,_=head_routing_attention(scores,values,span,rows[::-1],q.flip(0),p.flip(0))
    assert torch.equal(context,reversed_context) and torch.equal(probability,reversed_prob)
    for slot,row in enumerate(rows):
        single_context,single,_=head_routing_attention(scores[row:row+1],values[row:row+1],span,[0],q[slot:slot+1],p[slot:slot+1])
        assert torch.equal(single,probability[row:row+1]) and torch.equal(single_context,context[row:row+1])
    # Two heads with equal initial mass prefer opposite patches. Swapping
    # evidence and FFN preferences must reverse which head gains visual mass.
    toy=torch.tensor([[[[0.,3.,0.,0.]],[[0.,0.,3.,0.]]]])
    val=torch.randn(1,2,4,3);a=torch.tensor([[1.,0.]]);b=1-a
    _,f,fd=head_routing_attention(toy,val,(1,3),[0],a,b)
    _,g,gd=head_routing_attention(toy,val,(1,3),[0],b,a)
    fm=f[0,:,0,1:3].sum(-1);gm=g[0,:,0,1:3].sum(-1)
    assert fm[0]>fm[1] and gm[0]<gm[1]
    torch.testing.assert_close(fm.flip(0),gm,atol=2e-7,rtol=0)
    bf16=scores.to(torch.bfloat16);bv=values.to(torch.bfloat16)
    _,_,bd=head_routing_attention(bf16,bv,span,rows,q,p)
    report=dict(passed=True,source_sha256=source,images_independent_exact=True,
        reversed_rows_exact=True,input_and_unselected_rows_unchanged=True,
        flat_or_equal_QP_exact_uniform=True,zero_tilt_exact_native=True,
        semantic_reversal_swaps_heads=True,
        fp32_mean_visual_mass_max_error=float(d['fp32_mean_visual_mass_difference'].max()),
        bf16_mean_visual_mass_max_error=float(bd['bf16_mean_visual_mass_difference'].max()),
        conditional_visual_max_error=float(d['conditional_visual_difference'].max()),
        individual_head_mass_change=float(d['maximum_individual_head_mass_change'].max()),
        cuda_initialized=torch.cuda.is_initialized(),model_loaded=False)
    assert report['fp32_mean_visual_mass_max_error']<=32*2**-23
    assert report['bf16_mean_visual_mass_max_error']<=2**-7
    assert report['conditional_visual_max_error']<=1e-5
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    verify()
