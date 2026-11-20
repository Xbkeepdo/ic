"""Check reused feature norms and 17-layer joint linearization independently."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from features import qwen3_native_risk_i_el as native
from features.qwen3_block_acg_signal import BlockRiskReader, Qwen3BatchBlockACG, POST


def verify():
    torch.manual_seed(43); torch.set_num_threads(1)
    reader = object.__new__(BlockRiskReader)
    reader.head = SimpleNamespace(weight=torch.randn(13, 8))
    reader.decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(
        o_proj=torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16))) for _ in range(36)])
    reader.feature_seconds = 0.; reader.feature_calls = reader.risk_solver_calls = 0
    reader.columns = {1: 0, 2: 1}
    reader.visual = {i: torch.randn(2, 4, 8) for i in range(36)}
    reader.values = {i: torch.randn(2, 2, 4, 2).to(torch.bfloat16) for i in range(36)}
    reader.current = {i: dict(hpre=torch.randn(2, 8), update=torch.randn(2, 8),
        attention=torch.randn(2, 4, 4).softmax(-1).to(torch.bfloat16)) for i in range(36)}
    static = {(i, row):dict(log_s=torch.rand(4, 2),E_l=torch.rand(2)) for i in range(36) for row in range(2)}
    reader._static = lambda i, row:static[i, row]
    original = native.native_write_norms
    expected = native.NativeRiskIEl.features(reader, [1, 0], [1, 2])
    actual = reader.features([1, 0], [1, 2])
    assert torch.equal(expected, actual) and native.native_write_norms is original
    for slot, row in enumerate((1, 0)):
        for layer in POST:
            assert float(reader.write_norms[layer, row].sum()) == float(expected[slot, 36+layer])
    # The scoped interceptor must restore the frozen module even on failure.
    saved = reader._static
    def fail(*_):raise RuntimeError('deliberate feature failure')
    reader._static = fail
    try:reader.features([1], [1])
    except RuntimeError:pass
    else:raise AssertionError('Expected deliberate feature failure')
    assert native.native_write_norms is original
    reader._static = saved

    width, batch = 5, 3
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.RMSNorm(width, eps=1e-6)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 8),torch.nn.Tanh(),torch.nn.Linear(8, width))
            self.attention = torch.nn.Linear(width, width, bias=False)
            self.perturbation = None
        def forward(self, hidden, **_kwargs):
            residual = hidden+.03*self.attention(hidden).tanh()
            if self.perturbation is not None:residual = residual+self.perturbation
            return residual+self.mlp(self.post_attention_layernorm(residual))
    decoder = SimpleNamespace(layers=[None]*19+[Layer().eval().requires_grad_(False) for _ in POST],
        norm=torch.nn.RMSNorm(width, eps=1e-6).eval().requires_grad_(False))
    obj = object.__new__(Qwen3BatchBlockACG)
    obj.decoder, obj.head = decoder, torch.nn.Linear(width, 7, bias=False).eval().requires_grad_(False)
    obj.enabled=True; obj.replaying_tail=obj._replaying=False
    obj.maximum_probe_hidden_difference=0.; obj.gradient_probes=0
    obj.tail_kwargs = dict(past_key_values=SimpleNamespace(layers=[]),
        position_embeddings=(torch.ones(1),torch.ones(1)),cache_position=torch.tensor([8]))
    obj.final_tokens=torch.tensor([1, 3, 5]); rows=[2, 0]
    residual=torch.randn(batch, width)
    obj.scratch={19:dict(ffn_residual=residual)}
    def forward(changes=None, capture=False):
        changes={} if changes is None else changes
        r=residual[:,None]+changes.get(19,0)
        update=decoder.layers[19].mlp(decoder.layers[19].post_attention_layernorm(r))
        if capture:obj.scratch[19]['update']=update[:,0].detach()
        current=r+update
        with ExitStack() as stack:
            for i in POST[1:]:
                layer=decoder.layers[i]
                if i in changes:
                    layer.perturbation=changes[i]
                    stack.callback(setattr,layer,'perturbation',None)
                if capture:
                    obj.scratch[i]={}
                    def save(_module,args,i=i):obj.scratch[i]['ffn_residual']=args[0][:,0].detach()
                    stack.callback(layer.post_attention_layernorm.register_forward_pre_hook(save).remove)
                current=layer(current)
        norm=decoder.norm(current)
        return obj.head(norm).float()[:,0].log_softmax(-1)[rows,obj.final_tokens[rows]].sum(),norm
    with torch.no_grad():obj.reference_hidden=forward(capture=True)[1]
    grad=torch.autograd.grad; calls=[]
    def counted(*args,**kwargs):calls.append(1);return grad(*args,**kwargs)
    with patch.object(torch.autograd,'grad',counted):obj._block_gradient(rows)
    assert len(calls)==1 and set(obj.block_gradients)==set(POST)
    max_error=0.; step=.002
    directions={i:torch.randn(batch,1,width)/len(POST)**.5 for i in POST}
    with torch.no_grad():
        for i in POST:
            delta=directions[i]
            finite=float((forward({i:step*delta})[0]-forward({i:-step*delta})[0])/(2*step))
            predicted=float((obj.block_gradients[i]*delta[rows,0]).sum())
            max_error=max(max_error,abs(finite-predicted))
        finite=float((forward({i:step*d for i,d in directions.items()})[0]-forward({i:-step*d for i,d in directions.items()})[0])/(2*step))
        predicted=sum(float((obj.block_gradients[i]*directions[i][rows,0]).sum()) for i in POST)
        joint_error=abs(finite-predicted)
    assert max_error<6e-4 and joint_error<1e-3,(max_error,joint_error)
    counts=[len(decoder.layers[i].post_attention_layernorm._forward_pre_hooks) for i in POST]
    old=obj.reference_hidden;obj.reference_hidden=old+1
    try:obj._block_gradient(rows)
    except AssertionError:pass
    else:raise AssertionError('Expected deliberate native point mismatch')
    finally:obj.reference_hidden=old
    assert obj.enabled and not obj.replaying_tail and not obj._replaying
    assert counts==[len(decoder.layers[i].post_attention_layernorm._forward_pre_hooks) for i in POST]
    return dict(passed=True,layers=list(POST),feature108_bitwise_unchanged=True,
        saved_norms_match_all34_original_I_entries=True,feature_interceptor_restored_on_failure=True,
        one_backward_for_all17_layers=len(calls)==1,per_layer_finite_difference_error=max_error,
        joint_finite_difference_error=joint_error,sparse_reordered_rows=True,flags_hooks_restored_on_failure=True)


if __name__=='__main__':
    from scripts.run_qwen3_signed_feedback import write
    result=verify();out=REPO/'outputs/qwen3_block_acg_cpu';out.mkdir(exist_ok=True)
    write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))
