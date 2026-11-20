"""CPU derivative and direction-step verification, no model weights loaded."""

from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from torch.nn import functional as F
from features.qwen3_rival_ffn_alignment import preference_objective,align_update
from scripts.run_qwen3_signed_feedback import write,sha


def main(destination):
    assert not destination.exists();torch.manual_seed(43)
    h=F.normalize(torch.randn(19,13,dtype=torch.float64),dim=-1)
    u=torch.randn(13,dtype=torch.float64,requires_grad=True);q=torch.rand(19,dtype=torch.float64);q/=q.sum()
    value,tangent,_=preference_objective(h,u,q)
    actual,=torch.autograd.grad(value,u)
    expected=tangent/u.detach().norm()
    error=float((actual-expected).abs().max());assert error<1e-12,error
    finite=[]
    for index in [0,4,12]:
        plus,minus=u.detach().clone(),u.detach().clone();plus[index]+=1e-6;minus[index]-=1e-6
        difference=float((preference_objective(h,plus,q)[0]-preference_objective(h,minus,q)[0])/2e-6)
        finite.append(abs(difference-float(expected[index])))
    assert max(finite)<1e-7
    zero,zero_info=align_update(h,u.detach(),q,0.)
    assert torch.equal(zero,u.detach()) and not zero_info['applied']
    output,info=align_update(h,u.detach(),q,.7)
    assert info['applied'] and info['after']<info['before'] and abs(float(output.norm()-u.detach().norm()))<1e-12
    assert float((expected*u.detach()).sum().abs())<1e-12
    write(destination,dict(passed=True,analytic_vs_autograd_max_error=error,finite_difference_max_error=max(finite),
        zero_update_exact=True,norm_preserving_step=info,no_model_weights_loaded=True,cuda_initialized=torch.cuda.is_initialized(),
        source_sha256={str(p):sha(p) for p in [Path(__file__),REPO/'features/qwen3_rival_ffn_alignment.py']}))
    print(dict(passed=True,gradient_error=error,finite_difference_error=max(finite),step=info),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
