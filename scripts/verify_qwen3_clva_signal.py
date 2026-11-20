"""CPU checks of anchor selection and full-key probability normalization."""

from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_clva_signal import anchor_map,salient,reanchor
from scripts.run_qwen3_signed_feedback import write,sha


def main(path):
    assert not path.exists();torch.manual_seed(43)
    p=torch.randn(2,4,3,8).softmax(-1);v=torch.randn(2,4,8,5)
    context=(p@v).transpose(1,2).contiguous();old=p.clone()
    positive=torch.tensor([[True,False,False,False],[False,False,False,False]])
    negative=torch.tensor([[False,False,True,False],[False,False,False,False]])
    c,w,mass=reanchor(p,v,context,(2,6),positive,negative,torch.tensor([14.,14.]),torch.tensor([True,True]))
    factor=torch.ones(8);factor[2]=15.;factor[4]=.1
    expected=p[0,:,-1:]*factor;expected/=expected.sum(-1,keepdim=True)
    assert torch.allclose(w[0,:,-1:],expected,rtol=0,atol=1e-7)
    assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),rtol=0,atol=2e-7)
    assert torch.equal(c[1],context[1]) and torch.equal(w[1],p[1]) and torch.equal(c[:,:-1],context[:,:-1])
    assert torch.equal(p,old)
    a,b,_=reanchor(p,v,context,(2,6),positive,negative,torch.tensor([14.,14.]),torch.tensor([False,False]))
    assert a is context and b is p
    uniform=torch.ones(2,4,7)/7
    anchor,counts=anchor_map(uniform,True);assert not counts.any() and not salient(anchor).any()
    write(path,dict(passed=True,input_unchanged=True,nonlast_queries_and_unselected_rows_exact=True,
        inactive_is_native=True,empty_head_set_has_no_mask=True,no_model_weights=True,cuda_initialized=torch.cuda.is_initialized(),
        probability_max_error=float((w[0,:,-1:]-expected).abs().max()),
        source_sha256={str(p):sha(p) for p in [Path(__file__),REPO/'features/qwen3_clva_signal.py']}))
    print(dict(passed=True,mass=mass),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
