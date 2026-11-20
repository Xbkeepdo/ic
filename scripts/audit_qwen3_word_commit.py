"""Check greedy-change acceptance, rollback preflight, and prefix-class gates."""

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from features.coco_object_history import ObjectClassHistory
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def audit(root):
    from transformers import AutoTokenizer
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    preflight = read(Path(plan['actuation_preflight']))
    assert preflight['passed'] and preflight['checked_current_KV_layers'] == list(range(20, 36))
    assert all(preflight['cases'][key]['exact_native_logits_tokens_and_current_tail_KV'] for key in ('disabled', 'zero', 'reject'))
    assert preflight['cases']['live']['unchanged_token_rejections'] > 0
    assert preflight['cases']['live']['native_prefix_rows_with_exact_logits_and_current_KV'] > 0
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    source = Path(plan['source'])
    assert (root / 'object_lexicon.json').read_bytes() == (source / 'object_lexicon.json').read_bytes()
    classes = parser.token_classes(tokenizer, read(root / 'object_lexicon.json')['token_ids'])
    baseline = {r['image_id']: r for shard in (0, 1)
                for group in source_groups(root / f'coco100/sadi.shard{shard}.jsonl') for r in group}
    name, = plan['candidates']
    report = dict(passed=True, images=0, proposals=0, accepted=0, rejected_unchanged=0,
                  probability_decreases=0, gradient_probe_calls=0, no_retained_action_matches_sadi=0)
    for shard in (0, 1):
        for group in source_groups(root / f'coco100/{name}.shard{shard}.jsonl'):
            probes = {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            assert len(probes) == 1
            report['gradient_probe_calls'] += probes.pop()
            for row in group:
                report['images'] += 1
                tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
                assert stats['acceptance_requires_token_change'] and stats['additional_acceptance_lm_head_calls'] == 0
                assert stats['maximum_probe_hidden_difference'] == 0
                assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                                                    'diagnostic_reject_all', 'diagnostic_zero_replay'))
                decisions = {d['step']: d for d in stats['class_decisions']}
                assert len(decisions) == len(stats['class_decisions'])
                for step, decision in decisions.items():
                    target = decision['token']
                    assert target in classes and target not in tokens[:step]
                    prior = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False))
                    assert decision['canonical'] == classes[target] and decision['prior_classes'] == sorted(prior)
                    assert decision['blocked'] == (classes[target] in prior)
                events = stats['recycle_events']
                assert len(events) == stats['guided_attention_calls']
                assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                assert sum(e['actual_log_probability_change'] < 0 for e in events) == stats['probability_decrease_proposals']
                assert sum(e['rejected_no_token_change'] for e in events) == stats['rejected_unchanged_token_calls']
                for event in events:
                    step, target = event['step'], event['token']
                    assert not decisions[step]['blocked'] and decisions[step]['token'] == target
                    assert event['source_layer'] == event['target_layer'] == 19
                    assert event['probe_p_real'] < .5 and event['predicted_log_probability_change'] <= 1e-6
                    assert event['probability_decrease_accepted'] == (event['actual_log_probability_change'] < 0)
                    changed = event['proposed_token'] != target
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0 and changed)
                    assert event['rejected_no_token_change'] == (event['actual_log_probability_change'] < 0 and not changed)
                    assert event['generated_token'] == tokens[step]
                    assert tokens[step] == (event['proposed_token'] if event['accepted'] else target)
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                    report['probability_decreases'] += int(event['actual_log_probability_change'] < 0)
                    report['rejected_unchanged'] += int(event['rejected_no_token_change'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == baseline[row['image_id']]['response_token_ids']
                    report['no_retained_action_matches_sadi'] += 1
    assert report['images'] == plan['images']
    write(root / 'word_commit_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
