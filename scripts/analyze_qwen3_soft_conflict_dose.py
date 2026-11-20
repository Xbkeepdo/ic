"""Prospective soft-dose rule: CPU formula proof and frozen-cache coverage."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_soft_conflict_dose import soft_conflict_gates


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def proof():
    # Independent four-head arithmetic: only heads 0 and 1 are Conflict-B.
    v, t, r = [-1., -3., 2., -4.], [3., 1., 2., -2.], [.6, .9, 1., 1.]
    expected = {'signal': [.9, .6, 1., 1.],
                'flat': [.75, .75, 1., 1.], 'roll': [.6, .9, 1., 1.]}
    maximum_error = 0.
    for mode in expected:
        g, d = soft_conflict_gates(v, t, r, mode=mode)
        assert d['base_selected_heads'] == [0, 1]
        assert max(abs(a-b) for a, b in zip(g, expected[mode])) < 1e-14
        # A direct two-dimensional route edit, evaluated by an independent
        # linear objective, must agree with the reported first-order sum.
        texts = [[1., 2.], [-1., 2.], [0., 2.], [0., -2.]]
        gradient = [1., 1.]
        route_delta = [math.fsum((g[h]-1.)*texts[h][j] for h in range(4))
                       for j in range(2)]
        actual = math.fsum(a*b for a, b in zip(route_delta, gradient))
        error = abs(actual-d['predicted_change'])
        assert error < 1e-14
        maximum_error = max(maximum_error, error)
    for neutral in [0., .5]:
        for mode in expected:
            gates, _ = soft_conflict_gates(v, t, [neutral]*4, mode=mode)
            assert gates == [1.]*4
    for mode in expected:
        gates, _ = soft_conflict_gates(v, t, [1.]*4, mode=mode)
        assert gates == [.5, .5, 1., 1.]
    gates, detail = soft_conflict_gates(v, t, r, mode='signal', k=1)
    assert gates == [.9, 1., 1., 1.] and detail['base_selected_heads'] == [0]
    assert soft_conflict_gates([0., 1.], [1., -1.], [1., 1.], mode='signal')[0] == [1., 1.]
    return dict(passed=True, independent_effect_maximum_error=maximum_error,
                neutral_and_low_risk_exact_identity=True,
                maximum_dose_half=True, no_model_or_torch_import=True)


def analyze(strong, common, output):
    assert not output.exists(), 'Inspect existing output rather than rerunning.'
    files = [strong/'protocol.json', strong/'final_provenance.json', strong/'cohort.json',
             common/'protocol.json', common/'analysis.json',
             REPO/'features/qwen3_soft_conflict_dose.py', Path(__file__)]
    files += [strong/f'coco100/{m}.shard{s}.jsonl'
              for m in ('route_signal', 'route_roll') for s in (0, 1)]
    inputs = {str(p): sha(p) for p in files}
    output.mkdir()
    write(output/'protocol.json', dict(fixed_at=time.time(), input_sha256=inputs,
        before_inspecting_soft_dose_distribution=True,
        hypothesis='Rank-only gates may apply a large dose even near the neutral Q=P point. Use the amount of positive FFN/evidence imbalance as dose.',
        formula='d_h=0.5*relu(2*R_h-1)=relu(R_h-0.5); g_h=1-d_h',
        heads='At most8 native Conflict-B heads, smallest VRI; same selection in all3 modes.',
        controls='Flat preserves mean removal; roll permutes the same selected-head dose multiset.',
        no_training_or_threshold_search=True, neural_generation=False,
        scope='Old development100, exact common pre-action points only; no counterfactual caption claims.',
        followup='Coverage and independent arithmetic first; no500 before full implementation and development checks.'))
    verification = proof()
    shared = read(common/'analysis.json')
    assert shared['passed'] and read(strong/'final_provenance.json')['passed']
    keys = {(x['image_id'], x['step'], x['token_id']) for x in shared['records']}
    lookup = {}
    for method in ('route_signal', 'route_roll'):
        for shard in (0, 1):
            with (strong/f'coco100/{method}.shard{shard}.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)
                    for event in row['sar_ems_statistics']['full_joint_events']:
                        key = row['image_id'], event['step'], event['token']
                        if key in keys:
                            lookup[method, key] = event['layer_details']
    records = []
    for point in shared['records']:
        key = point['image_id'], point['step'], point['token_id']
        a, b = lookup['route_signal', key], lookup['route_roll', key]
        assert all(a[str(i)]['raw_head_risk'] == b[str(i)]['raw_head_risk']
                   for i in range(19, 36)), key
        layers = []
        for i in range(19, 36):
            d = a[str(i)]
            arguments = d['visual_effect'], d['textual_effect'], d['raw_head_risk']
            modes = {mode: soft_conflict_gates(*arguments, mode=mode)
                     for mode in ('signal', 'flat', 'roll')}
            gates, measures = modes['signal']
            chosen = measures['base_selected_heads']
            assert all(m[1]['base_selected_heads'] == chosen for m in modes.values())
            assert sorted(modes['signal'][1]['selected_raw_doses']) == sorted(modes['roll'][1]['selected_applied_doses'])
            assert abs(math.fsum(1-g for g in gates)-math.fsum(1-g for g in modes['flat'][0])) < 1e-12
            layers.append(dict(layer=i, selected_heads=chosen,
                selected_risk=[d['raw_head_risk'][h] for h in chosen],
                positive_doses=sum(g < 1 for g in gates),
                dose_sum=measures['selected_dose_sum'], maximum_dose=max(1-g for g in gates),
                predicted_change={m: values[1]['predicted_change'] for m, values in modes.items()}))
        records.append(dict(image_id=point['image_id'], step=point['step'], token_id=point['token_id'],
            canonical=point['canonical'], automatic_hallucination=point['automatic_hallucination'],
            context=point['context'], positive_layers=sum(x['positive_doses'] > 0 for x in layers),
            maximum_dose=max(x['maximum_dose'] for x in layers),
            mean_dose_per_selected_head=math.fsum(x['dose_sum'] for x in layers)/sum(len(x['selected_heads']) for x in layers),
            layers=layers))
    groups = {}
    for name, value in [('automatic_real', False), ('automatic_hallucination', True), ('unlabelled', None)]:
        rows = [x for x in records if x['automatic_hallucination'] is value]
        groups[name] = dict(points=len(rows), zero_dose_points=sum(x['positive_layers'] == 0 for x in rows),
            total_selected_heads=sum(len(l['selected_heads']) for x in rows for l in x['layers']),
            positive_head_doses=sum(l['positive_doses'] for x in rows for l in x['layers']),
            maximum_dose=max((x['maximum_dose'] for x in rows), default=0.),
            mean_dose_per_selected_head=math.fsum(l['dose_sum'] for x in rows for l in x['layers'])/max(1, sum(len(l['selected_heads']) for x in rows for l in x['layers'])))
    assert len(records) == shared['shared_native_points']
    for path, value in inputs.items():
        assert sha(Path(path)) == value, path
    write(output/'verification.json', verification)
    result = dict(passed=True, protocol_sha256=sha(output/'protocol.json'), groups=groups,
        common_points=len(records), records=records,
        limitations='Automatic labels include colour/sense/annotation ambiguity. These are dose and arithmetic diagnostics, not estimated CHAIR or a calibrated detector.',
        no_new_model_calls=True, new_generation=False, reserved500_untouched=True)
    write(output/'analysis.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('strong', type=Path)
    parser.add_argument('common', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.strong, args.common, args.output)
