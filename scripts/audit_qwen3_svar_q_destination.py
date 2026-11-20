"""Verify the destination-only intervention and every canonical-prefix decision."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import audit_qwen3_probe_first_class as classes
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def physical_audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    check = read(Path(plan['actuation_preflight']))
    assert check['passed'] and check['baseline_variant'] == 'svar'
    assert check['checked_current_KV_layers'] == list(range(20, 36))
    for name in ('disabled', 'zero', 'reject'):
        assert check['cases'][name]['exact_native_logits_tokens_and_current_tail_KV']
    assert check['cases']['reject']['forward_counts'] == check['parent_reject_forward_counts']
    assert check['cases']['live']['proposals'] > 0
    name, = plan['candidates']
    baseline = {r['image_id']: r for shard in (0, 1)
                for group in source_groups(root / plan['cohort'] / f'svar.shard{shard}.jsonl') for r in group}
    report = dict(passed=True, images=0, proposals=0, accepted=0, gradient_calls=0,
                  no_retained_exact_svar=0, native_prefix_events=0, maximum_mass_error=0.)
    for shard in (0, 1):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            calls = {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            captured = {r['sar_ems_statistics']['negative_destination_captures'] for r in group}
            assert len(calls) == len(captured) == 1 and calls == captured
            report['gradient_calls'] += calls.pop()
            for row in group:
                report['images'] += 1
                stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
                old = baseline[row['image_id']]['response_token_ids']
                assert stats['base_variant'] == stats['local_I_before_base_edit'] == 'svar'
                assert stats['intervention_layer'] == 19 and stats['replayed_tail_layers'] == list(range(20, 36))
                assert stats['maximum_probe_hidden_difference'] == 0
                assert stats['ffn_drift_amount_preserved'] and not stats['ffn_destination_weighting']
                assert stats['destination_rule'] == 'per_head_negative_writes'
                assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                                                    'diagnostic_zero_replay', 'diagnostic_reject_all'))
                assert all(stats[k] == 0 for k in ('additional_gradient_probes', 'additional_model_forward_calls',
                                                 'additional_mlp_forward_calls', 'additional_lm_head_calls'))
                events = stats['recycle_events']
                assert len(events) == stats['guided_attention_calls']
                assert len({e['step'] for e in events}) == len(events)
                assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                for event in events:
                    step = event['step']
                    assert event['source_layer'] == event['target_layer'] == 19
                    assert event['probe_p_real'] < .5 and 0 < event['drift'] <= 1
                    assert 'predicted_log_probability_change' not in event
                    assert event['reference_predicted_log_probability_change'] <= 1e-6
                    assert event['first_order_log_probability_change_upper_bound'] == 0
                    assert event['first_order_bound_from_negative_visual_and_positive_history']
                    assert event['maximum_nonvisual_budget_difference'] == 0 and event['mass_conservation_error'] < 1e-6
                    assert abs(event['reference_moved_attention_mass'] - event['moved_attention_mass']) < 1e-5
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                    assert event['generated_token'] == tokens[step]
                    if not event['accepted']:
                        assert tokens[step] == event['token']
                    if old[:step] == tokens[:step]:
                        assert old[step] == event['token']
                        report['native_prefix_events'] += 1
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                    report['maximum_mass_error'] = max(report['maximum_mass_error'], event['mass_conservation_error'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == old
                    report['no_retained_exact_svar'] += 1
    assert report['images'] == plan['images']
    write(root / 'q_destination_runtime_audit.json', report)
    print(report)


def audit(root):
    with patch.object(classes, 'physical_audit', physical_audit):
        classes.audit(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
