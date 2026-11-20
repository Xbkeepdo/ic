"""Pair images across a risk gate and every prespecified fixed hash control."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_qwen3_fresh_caption500 import boot, calculate


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root, risk, output=None):
    assert read(root / 'run_status.json')['status'] == 'complete'
    plan = read(root / 'protocol.json')
    audit = read(root / 'matched_hash_audit.json')
    assert audit['passed'] and audit['seeds'] == plan['gate_seeds'] == [43, 44, 45]
    summary = read(root / 'independent_summary.json')
    assert summary['passed']
    names = plan['candidates']
    assert len(names) == 3 and len(set([risk, *names])) == 4
    ids = read(root / 'cohort.json')[plan['cohort']]
    assert len(ids) == len(set(ids)) == plan['images']
    keys = ('chair_s', 'chair_i', 'object_recall', 'object_f1')
    report = dict(scope=f'Mean of all three fixed seeds on the same {len(ids)} images. '
        'Resample one common image index vector for the risk gate and every control. '
        'No best-seed selection; intervals are conditional on these three fixed seeds.',
        images=len(ids), risk=risk, controls=names, seeds=plan['gate_seeds'],
        analysis_code_sha256=sha(Path(__file__)), inputs_sha256={}, modes={})
    for path in (root / 'protocol.json', root / 'matched_hash_audit.json',
                 root / 'independent_summary.json', root / 'cohort.json'):
        report['inputs_sha256'][str(path)] = sha(path)
    draws = np.random.default_rng(43).integers(0, len(ids), (5000, len(ids)))
    for mode in ('sentence', 'legacy'):
        per, = (root / f'evaluations/{mode}_v1/{plan["cohort"]}').glob('chair_per_image*.jsonl')
        report['inputs_sha256'][str(per)] = sha(per)
        rows = [json.loads(line) for line in per.read_bytes().splitlines()]
        assert [r['image_id'] for r in rows] == ids
        calculated = {name: calculate([r['methods'][name] for r in rows]) for name in [risk, *names]}
        for name in calculated:
            for key in keys:
                np.testing.assert_allclose(calculated[name][key], summary['eval'][mode][name][key],
                                           rtol=0, atol=1e-12)
        samples = {name: boot([r['methods'][name] for r in rows], draws) for name in calculated}
        mean = {k: float(np.mean([calculated[n][k] for n in names])) for k in keys}
        sd = {k: float(np.std([calculated[n][k] for n in names], ddof=1)) for k in keys}
        delta = {k: calculated[risk][k] - mean[k] for k in keys}
        intervals = {k: np.quantile(samples[risk][k] - np.mean(
            [samples[n][k] for n in names], axis=0), [.025, .975]).tolist() for k in keys}
        report['modes'][mode] = dict(fixed_hash_mean=mean, fixed_hash_sample_sd=sd,
            risk_minus_hash_mean=delta, risk_minus_hash_mean_ci95=intervals,
            methods=calculated)
    target = output or root / 'risk_vs_fixed_hash_mean.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['modes']['sentence'], indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--risk', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summarize(args.root, args.risk, args.output)
