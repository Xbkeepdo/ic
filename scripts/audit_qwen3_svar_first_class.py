"""Verify the unchanged attention action and class gate against native SVAR."""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import audit_qwen3_probe_first_class as classes
from scripts.audit_qwen3_visual_feedback_mentions import compare, summarize
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def physical_audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    preflight = read(Path(plan['actuation_preflight']))
    assert preflight['passed'] and preflight['baseline_variant'] == 'svar'
    assert preflight['checked_current_KV_layers'] == list(range(20, 36))
    assert all(preflight['cases'][key]['exact_native_logits_tokens_and_current_tail_KV'] for key in ('disabled', 'zero', 'reject'))
    semantic = read(Path(plan['semantic_preflight']))
    assert semantic['passed'] and semantic['exact_E_l'] and semantic['max_abs_difference'] == 0
    baseline = {row['image_id']: row for shard in range(plan['shards'])
                for group in source_groups(root / plan['cohort'] / f'svar.shard{shard}.jsonl') for row in group}
    name, = plan['candidates']
    report = dict(passed=True, primary_baseline='svar', images=0, proposals=0, accepted=0,
                  native_prefix_events=0, no_retained_action_matches_svar=0)
    for shard in range(plan['shards']):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            for row in group:
                report['images'] += 1
                tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
                assert stats['base_variant'] == stats['local_I_before_base_edit'] == 'svar'
                assert stats['intervention_layer'] == 19 and stats['replayed_tail_layers'] == list(range(20, 36))
                assert stats['maximum_probe_hidden_difference'] == 0
                assert not any(stats.get(key) for key in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                                                        'diagnostic_reject_all', 'diagnostic_zero_replay'))
                events = stats['recycle_events']
                assert len(events) == stats['guided_attention_calls']
                assert sum(event['accepted'] for event in events) == stats['accepted_guidance_calls']
                assert len({event['step'] for event in events}) == len(events)
                old = baseline[row['image_id']]['response_token_ids']
                for event in events:
                    step, target = event['step'], event['token']
                    assert event['source_layer'] == event['target_layer'] == 19
                    assert event['probe_p_real'] < .5 and event['predicted_log_probability_change'] <= 1e-6
                    assert event['moved_attention_mass'] > 0 and 0 < event['drift'] <= 1
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                    assert event['generated_token'] == tokens[step]
                    if not event['accepted']:
                        assert tokens[step] == target
                    if old[:step] == tokens[:step]:
                        assert old[step] == target
                        report['native_prefix_events'] += 1
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == old
                    report['no_retained_action_matches_svar'] += 1
    assert report['images'] == plan['images']
    write(root / 'svar_recycle_event_audit.json', report)
    print(json.dumps(report, indent=2))


def audit(root):
    # Reuse the full prefix/alias verifier, supplying a complete SVAR physical
    # verifier rather than its SADI-specific physical comparison.
    with patch.object(classes, 'physical_audit', physical_audit):
        classes.audit(root)
    plan = read(root / 'protocol.json')
    ids = read(root / 'cohort.json')[plan['cohort']]
    draws = np.random.default_rng(43).integers(0, len(ids), (5000, len(ids)))
    name, = plan['candidates']
    report = {}
    for mode in ('sentence', 'legacy'):
        per, = (root / f'evaluations/{mode}_v1/{plan["cohort"]}').glob('chair_per_image*.jsonl')
        rows = [json.loads(line) for line in per.read_bytes().splitlines()]
        assert [row['image_id'] for row in rows] == ids
        baseline = [row['methods']['svar'] for row in rows]
        candidate = [row['methods'][name] for row in rows]
        report[mode] = dict(primary_baseline='svar', base=summarize(baseline), candidate=summarize(candidate),
                            comparison=compare(baseline, candidate, ids, draws))
    write(root / 'svar_primary_mention_audit.json', report)
    print('SVAR_PRIMARY_MENTIONS_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
