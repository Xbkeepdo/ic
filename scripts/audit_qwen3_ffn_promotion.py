"""Recompute class history and verify the positive-FFN gate and readout reuse."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from scripts.audit_qwen3_probe_first_class import audit as class_audit
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def audit(root):
    class_audit(root)
    plan = read(root / 'protocol.json')
    name, = plan['candidates']
    baseline = {r['image_id']: r for shard in (0, 1) for group in source_groups(
        root / plan['cohort'] / f'sadi.shard{shard}.jsonl') for r in group}
    report = dict(passed=True, images=0, risk_positive=0, gate_blocked=0, gate_allowed=0,
                  proposals=0, accepted=0, readout_cache_hits=0, no_action_exact_baseline=0)
    for shard in (0, 1):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            caches = {r['sar_ems_statistics']['contrast_readout_cache_hits'] for r in group}
            gradients = {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            assert len(caches) == len(gradients) == 1 and caches == gradients
            report['readout_cache_hits'] += caches.pop()
            for row in group:
                report['images'] += 1
                stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
                assert stats['ffn_promotion_gate'] and stats['require_positive_ffn_gain']
                assert not stats['require_negative_visual_support']
                classes = {d['step']: d for d in stats['class_decisions']}
                decisions = {d['step']: d for d in stats['promotion_decisions']}
                assert len(decisions) == len(stats['promotion_decisions'])
                assert sum(not d['permitted'] for d in decisions.values()) == stats['blocked_by_ffn_promotion']
                assert len(decisions) == stats['raw_probe_positive_calls']
                for step, d in decisions.items():
                    assert not classes[step]['blocked'] and classes[step]['token'] == d['token']
                    assert d['probe_p_real'] < .5
                    assert all(math.isfinite(d[k]) for k in ('probe_p_real', 'visual_support', 'gain'))
                    assert d['permitted'] == (d['gain'] > 0)
                    if not d['permitted']:
                        assert tokens[step] == d['token']
                    report['risk_positive'] += 1
                    report['gate_allowed' if d['permitted'] else 'gate_blocked'] += 1
                for e in stats['recycle_events']:
                    d = decisions[e['step']]
                    assert d['permitted'] and d['token'] == e['token']
                    assert all(d[k] == e[k] for k in ('probe_p_real', 'visual_support', 'gain'))
                    report['proposals'] += 1
                    report['accepted'] += int(e['accepted'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == baseline[row['image_id']]['response_token_ids']
                    report['no_action_exact_baseline'] += 1
    assert report['images'] == plan['images']
    write(root / 'ffn_promotion_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
