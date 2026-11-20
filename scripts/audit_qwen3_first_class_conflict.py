"""Independently check class history, the two signs and cached sign reuse."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from features.coco_object_history import ObjectClassHistory
from scripts.audit_qwen3_probe_recycle_events import audit as physical_audit
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, write


def audit(root):
    from transformers import AutoTokenizer
    physical_audit(root)
    plan = read(root / 'protocol.json')
    assert plan['independent500_status_at_design'] == 'generating'
    own_lexicon = root / 'object_lexicon.json'
    assert sha(own_lexicon) == plan['source_sha256'][str(own_lexicon)]
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(own_lexicon)['token_ids'])
    name, = plan['candidates']
    report = dict(passed=True, images=0, class_decisions=0, class_blocked=0, raw_probe_positive=0,
                  sign_blocked=0, sign_permitted=0, proposals=0, accepted=0,
                  cached_action_readouts=0, equal_gradient_probe_calls=True)
    for shard in (0, 1):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            cache_counts = {r['sar_ems_statistics']['contrast_readout_cache_hits'] for r in group}
            gradient_counts = {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            assert len(cache_counts) == len(gradient_counts) == 1 and cache_counts == gradient_counts
            report['cached_action_readouts'] += cache_counts.pop()
            for row in group:
                report['images'] += 1
                stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
                assert stats['conflict_sign_gate']
                assert stats['require_negative_visual_support'] and stats['require_positive_ffn_gain']
                decisions = {d['step']: d for d in stats['class_decisions']}
                assert len(decisions) == len(stats['class_decisions'])
                for step, d in decisions.items():
                    target = d['token']
                    assert target in classes and target not in tokens[:step]
                    prior = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False))
                    assert d['prior_classes'] == sorted(prior) and d['canonical'] == classes[target]
                    assert d['blocked'] == (classes[target] in prior)
                    if d['blocked']:
                        assert tokens[step] == target
                        report['class_blocked'] += 1
                    report['class_decisions'] += 1
                signs = {d['step']: d for d in stats['conflict_decisions']}
                assert len(signs) == len(stats['conflict_decisions'])
                for step, d in signs.items():
                    assert not decisions[step]['blocked'] and decisions[step]['token'] == d['token']
                    assert d['probe_p_real'] < .5
                    assert all(math.isfinite(d[k]) for k in ('probe_p_real', 'visual_support', 'gain'))
                    assert d['permitted'] == (d['visual_support'] < 0 and d['gain'] > 0)
                    if not d['permitted']:
                        assert tokens[step] == d['token']
                        report['sign_blocked'] += 1
                    else:
                        report['sign_permitted'] += 1
                    report['raw_probe_positive'] += 1
                for event in stats['recycle_events']:
                    d = signs[event['step']]
                    assert d['permitted'] and d['token'] == event['token']
                    assert all(d[k] == event[k] for k in ('probe_p_real', 'visual_support', 'gain'))
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
    assert report['images'] == plan['images']
    write(root / 'first_class_conflict_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
