"""Independent route-objective, dose-matching and no-op checks on CPU."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from features.qwen3_conflict_route_gates import apply_text_gates
from features.qwen3_soft_conflict_route_feedback import soft_route_payload
from scripts.analyze_qwen3_soft_conflict_dose import proof


def verify(root):
    assert not root.exists()
    root.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(2026111906)
    files = ['features/qwen3_soft_conflict_dose.py',
             'features/qwen3_soft_conflict_route_feedback.py',
             'scripts/analyze_qwen3_soft_conflict_dose.py', __file__]
    source = {n: hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in files}
    (root/'request.json').write_text(json.dumps(dict(source_sha256=source,
        formal_rule='d_h=relu(R_h-.5), maximum .5; lowest-VRI8 Conflict-B heads',
        controls=['signal', 'flat', 'roll'], before_gpu_model_runs=True), indent=2)+'\n')
    started = time.perf_counter()
    proof_result = proof()
    context = torch.randn(4,32,9)
    text = torch.randn_like(context)
    gradient = torch.randn_like(context)
    textual = (text*gradient).sum(-1)
    visual = torch.randn(4,32)
    risk = torch.rand(4,32)
    # Include exact neutral and full-strength endpoints in actual payloads.
    risk[0] = .5
    risk[1] = 1.
    details = [dict(token=row, layer_details={str(i):dict(
        visual_effect=visual[row].tolist(), textual_effect=textual[row].tolist(),
        raw_head_risk=risk[row].tolist(), applied_head_risk=risk[row].tolist(),
        text_gates=[.2]*32, selected_heads=list(range(32))) for i in range(19,36)})
        for row in range(4)]
    original = deepcopy(details)
    routes = {i:text.clone() for i in range(19,36)}
    worst = 0.
    modes = {}
    for mode in ['signal','flat','roll']:
        payload, records = soft_route_payload(details, routes, mode=mode)
        modes[mode] = payload[19]
        assert details == original
        for index, gates in payload.items():
            assert torch.equal(gates[0], torch.ones(32))
            assert (gates >= .5).all() and (gates <= 1).all()
            output = apply_text_gates(context, text, gates)
            objective_change = ((output-context).double()*gradient.double()).sum((1,2))
            predicted = torch.tensor([r['layer_details'][str(index)]['predicted_log_probability_change']
                                      for r in records], dtype=torch.float64)
            error = (objective_change-predicted).abs()
            bound = 16*torch.finfo(torch.float32).eps*(1+((output-context).double()*gradient.double()).abs().sum((1,2)))
            assert (error <= bound).all()
            worst = max(worst, error.max().item())
            # Native text routes and per-row selection must remain unchanged.
            assert torch.equal(routes[index], text)
            for row, record in enumerate(records):
                layer = record['layer_details'][str(index)]
                assert all(visual[row,h] < 0 and textual[row,h] > 0 for h in layer['base_selected_heads'])
                assert len(layer['base_selected_heads']) <= 8
                actual = gates[row].tolist()
                assert actual == layer['text_gates']
                assert layer['budget_max'] == float(((gates[row]-1)[:,None]*text[row]).abs().max())
    assert torch.equal(torch.sort(modes['signal'],dim=-1).values,
                       torch.sort(modes['roll'],dim=-1).values)
    assert torch.allclose((1-modes['signal']).sum(-1), (1-modes['flat']).sum(-1), atol=5e-7, rtol=0.)
    zero, _ = soft_route_payload(details, routes, mode='signal', zero=True)
    assert all(torch.equal(apply_text_gates(context,text,gates),context) for gates in zero.values())
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, source_sha256=source, direct_arithmetic=proof_result,
        independent_effect_maximum_error=worst, signal_roll_exact_multiset=True,
        flat_mean_dose_tolerance=5e-7, zero_payload_exact_identity=True,
        parent_details_and_native_text_unchanged=True, cuda_initialized=False,
        seconds=time.perf_counter()-started)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    verify(parser.parse_args().root)
