"""Check the attenuation endpoints, half removal and saved first-order effects."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import torch
from features.qwen3_mild_conflict_route_feedback import attenuate_route_payload
from features.qwen3_conflict_route_gates import conflict_text_gates,apply_text_gates


def verify(root):
    assert not root.exists();root.mkdir();started=time.perf_counter()
    torch.set_num_threads(1);torch.manual_seed(2026111905)
    files=['features/qwen3_mild_conflict_route_feedback.py','scripts/verify_qwen3_mild_conflict_route.py']
    hashes={n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in files}
    (root/'request.json').write_text(json.dumps(dict(source_sha256=hashes,formal_scale=.5,
        formal_selected_gate_range=[.5,.75],before_model_generation=True,
        endpoint_tests=[0.,1.],rank_scope='VRI/signal/roll unchanged before attenuation.'),indent=2)+'\n')
    context=torch.randn(3,32,9);textual=torch.randn_like(context);projected=torch.randn_like(context)
    visual_effect=torch.randn(3,32);textual_effect=(textual*projected).sum(-1);risk=torch.rand(3,32)
    maxima={};scaled={}
    for mode in ['vri','signal','roll']:
        gates,_=conflict_text_gates(visual_effect,textual_effect,risk,mode=mode)
        payload={index:gates.clone() for index in range(19,36)}
        texts={index:textual.clone() for index in payload}
        details=[dict(token=row,layer_details={str(index):dict(text_gates=gates[row].tolist(),
            textual_effect=textual_effect[row].tolist(),budget_max=0.,predicted_log_probability_change=0.,
            selected_heads=(gates[row]<1).nonzero().flatten().tolist()) for index in payload}) for row in range(3)]
        preserved=deepcopy(details);original={index:g.clone() for index,g in payload.items()}
        for scale in [0.,.5,1.]:
            current,records=attenuate_route_payload(payload,details,texts,scale)
            assert details==preserved and all(torch.equal(payload[i],original[i]) for i in payload)
            for index in current:
                g=current[index]
                if scale==0:assert torch.equal(g,torch.ones_like(g))
                elif scale==1:assert torch.equal(g,gates)
                else:
                    assert torch.equal(g<1,gates<1)
                    assert (g[g<1]>=.5).all() and (g[g<1]<=.75).all()
                    assert torch.equal(torch.argsort(g,stable=True),torch.argsort(gates,stable=True))
                out=apply_text_gates(context,textual,g)
                strong=apply_text_gates(context,textual,gates)
                if scale==0:assert torch.equal(out,context)
                if scale==1:assert torch.equal(out,strong)
                actual=((out-context).double()*projected.double()).sum((1,2))
                predicted=torch.tensor([r['layer_details'][str(index)]['predicted_log_probability_change'] for r in records],dtype=torch.float64)
                error=(actual-predicted).abs()
                bound=16*torch.finfo(torch.float32).eps*(1+((out-context).double()*projected.double()).abs().sum((1,2)))
                assert (error<=bound).all()
                if scale==.5:
                    half_error=(out.float()-context.float()-.5*(strong.float()-context.float())).abs().max().item()
                    assert half_error<2e-6
                for row,record in enumerate(records):
                    layers=record['layer_details'];layer=layers[str(index)]
                    assert layer['attenuation_scale']==scale
                    assert layer['strong_text_gates']==gates[row].tolist()
                    assert layer['text_gates']==g[row].tolist()
                    assert abs(record['predicted_log_probability_change']-sum(v['predicted_log_probability_change'] for v in layers.values()))<1e-9
            maxima[mode+':'+str(scale)]=float(error.max())
            if scale==.5:scaled[mode]=current[19]
    for value in scaled.values():assert torch.equal(value.sort(-1).values,scaled['vri'].sort(-1).values)
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,source_sha256=hashes,scale0_exact_identity=True,scale1_exact_strong=True,
        half_removal_checked=True,native_payload_and_details_unchanged=True,
        head_selection_and_order_preserved=True,gate_multisets_match=True,
        independent_linear_effect_max_errors=maxima,cuda_initialized=False,seconds=time.perf_counter()-started)
    for n,v in hashes.items():assert hashlib.sha256((REPO/n).read_bytes()).hexdigest()==v
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);verify(p.parse_args().root)
