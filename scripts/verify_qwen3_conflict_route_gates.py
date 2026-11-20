"""Independent CPU checks for route splitting, effect signs and matched gates."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from features.qwen3_conflict_route_gates import (
    split_head_routes,preference_over_evidence,conflict_text_gates,apply_text_gates)


def verify(output):
    assert not output.exists()
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(2026111904)
    hashes={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in
        ['features/qwen3_conflict_route_gates.py','scripts/verify_qwen3_conflict_route_gates.py']}
    output.mkdir();(output/'request.json').write_text(json.dumps(dict(source_sha256=hashes,
        before_model_generation=True,k=8,gate_range=[0.,.5],base_layers=[5,18],new_layers=[19,35],
        source_scope='Native joint gates; replay applies gates to actual per-layer nonvisual routes.',
        source='CRG Conflict-B adaptation plus SAR/FFN head ranking, not the full paper.'),indent=2)+'\n')
    scores=torch.randn(3,32,1,19);values=torch.randn(3,32,19,11);span=(2,13)
    context,visual,textual,probability=split_head_routes(scores,values,span)
    # An independent double-precision summation over explicit source indices.
    expected_context=torch.einsum('bhk,bhkd->bhd',scores.double().softmax(-1)[:,:,0],values.double())
    error=float((context.double()-expected_context).abs().max());assert error<2e-6
    split_error=float((context-visual-textual).abs().max());assert split_error<1e-6
    zero=apply_text_gates(context,textual,torch.ones(3,32));assert torch.equal(zero,context)
    assert torch.equal(apply_text_gates(context.bfloat16(),textual.bfloat16(),torch.ones(3,32)),context.bfloat16())
    q=torch.rand(3,11);q/=q.sum(-1,keepdim=True)
    p=torch.rand(3,11);p/=p.sum(-1,keepdim=True)
    conditional=probability[:,:,0,2:13].float().clone();conditional/=conditional.sum(-1,keepdim=True)
    risk=preference_over_evidence(conditional,q,p)
    independent=(conditional.double()*(p.double()/(p.double()+q.double()))[:,None]).sum(-1)
    risk_error=float((risk.double()-independent).abs().max());assert risk_error<2e-7
    assert torch.allclose(preference_over_evidence(conditional,q,q),torch.full_like(risk,.5),atol=1e-7,rtol=0)
    projected=torch.randn_like(context);vis=(visual*projected).sum(-1);txt=(textual*projected).sum(-1)
    records={};allgates={}
    for mode in ('vri','signal','roll'):
        gates,d=conflict_text_gates(vis,txt,risk,mode=mode)
        allgates[mode]=gates
        updated=apply_text_gates(context,textual,gates)
        # For a linear downstream objective, the first-order estimate is exact.
        actual=((updated-context)*projected).sum((1,2))
        errors=(actual-d['predicted_log_probability_change']).abs()
        bound=8*torch.finfo(torch.float32).eps*(1+((updated-context)*projected).abs().sum((1,2)))
        assert (errors<bound).all();effect_error=float(errors.max())
        assert torch.equal(updated[~d['selected']],context[~d['selected']])
        assert (gates<1).sum(-1).tolist()==d['conflict'].sum(-1).clamp_max(8).tolist()
        for row in range(3):
            alone=conflict_text_gates(vis[row:row+1],txt[row:row+1],risk[row:row+1],mode=mode)[0]
            assert torch.equal(alone,gates[row:row+1])
        records[mode]=dict(selected_per_row=d['selected'].sum(-1).tolist(),linear_effect_error=effect_error)
    for gates in allgates.values():assert torch.equal(gates.sort(-1).values,allgates['vri'].sort(-1).values)
    flat=torch.full_like(risk,.5)
    assert torch.equal(conflict_text_gates(vis,txt,flat,mode='vri')[0],conflict_text_gates(vis,txt,flat,mode='signal')[0])
    positive=torch.ones_like(vis)
    assert torch.equal(conflict_text_gates(positive,positive,risk,mode='signal')[0],positive)
    # A crafted conflict isolates the intended signal-driven ranking reversal.
    v=-torch.ones(1,4);t=torch.ones_like(v);s=torch.tensor([[.1,.3,.6,.9]])
    first=conflict_text_gates(v,t,s,mode='signal',k=1)[0]
    second=conflict_text_gates(v,t,1-s,mode='signal',k=1)[0]
    assert first.argmin(-1).item()==3 and second.argmin(-1).item()==0
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,source_sha256=hashes,context_error=error,route_split_error=split_error,
        risk_error=risk_error,matched_gate_multisets=True,image_independence_exact=True,
        flat_signal_recovers_vri=True,no_conflict_identity=True,signal_reversal_changes_selected_head=True,
        fp32_and_bf16_identity_exact=True,cases=records,cuda_initialized=False,
        seconds=time.perf_counter()-started)
    for name,value in hashes.items():assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==value
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    verify(parser.parse_args().output)
