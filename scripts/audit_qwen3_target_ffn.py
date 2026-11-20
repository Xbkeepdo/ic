"""Verify target-FFN readouts, exact control provenance and class decisions."""

import argparse
import math
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import audit_qwen3_probe_first_class as classes
from scripts.audit_qwen3_retained_prefix_state import verify_native_state_prefix
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def physical_audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    check = read(Path(plan['actuation_preflight']))
    assert check['passed'] and check['baseline_variant'] == 'sadi'
    assert check['base_layers_inclusive'] == [5, 18]
    assert check['checked_current_KV_layers'] == list(range(20, 36))
    for name in ('disabled', 'zero', 'reject'):
        assert check['cases'][name]['exact_native_logits_tokens_and_current_tail_KV']
    assert check['cases']['reject']['forward_counts'] == check['parent_reject_forward_counts']
    assert check['cases']['live']['proposals'] > 0
    for name, source in plan['additional_copied_controls'].items():
        for shard in (0, 1):
            filename = f'{name}.shard{shard}.jsonl'
            assert (root / plan['cohort'] / filename).read_bytes() == (Path(source) / plan['cohort'] / filename).read_bytes()
    name, = plan['candidates']
    baseline = {r['image_id']: r for shard in (0, 1)
        for group in source_groups(root / plan['cohort'] / f'sadi.shard{shard}.jsonl') for r in group}
    report = dict(passed=True, images=0, proposals=0, accepted=0, gradient_calls=0,
        no_retained_exact_sadi=0, native_prefix_events=0, maximum_preference_l1_change=0.)
    for shard in (0, 1):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            calls = {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            reads = {r['sar_ems_statistics']['target_ffn_preference_reads'] for r in group}
            derivatives = {r['sar_ems_statistics']['ffn_output_gradient_reads'] for r in group}
            assert len(calls) == len(reads) == len(derivatives) == 1 and calls == reads == derivatives
            report['gradient_calls'] += calls.pop()
            for row in group:
                report['images'] += 1
                stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
                old = baseline[row['image_id']]['response_token_ids']
                assert stats['base_variant'] == stats['local_I_before_base_edit'] == 'sadi'
                assert stats['intervention_layer'] == 19 and stats['replayed_tail_layers'] == list(range(20, 36))
                assert stats['maximum_probe_hidden_difference'] == 0
                assert stats['target_conditioned_ffn_preference'] and not stats['ffn_update_edited']
                assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                    'diagnostic_zero_replay', 'diagnostic_reject_all'))
                assert all(stats[k] == 0 for k in ('additional_gradient_backward_calls', 'additional_model_forward_calls',
                    'additional_mlp_forward_calls', 'additional_lm_head_calls'))
                events = stats['recycle_events']
                assert len(events) == stats['guided_attention_calls']
                assert len({e['step'] for e in events}) == len(events)
                assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                for event in events:
                    step = event['step']
                    assert event['source_layer'] == event['target_layer'] == 19
                    assert event['probe_p_real'] < .5 and 0 < event['drift'] <= 1
                    assert event['predicted_log_probability_change'] <= 1e-6
                    assert event['moved_attention_mass'] > 0
                    assert event['target_conditioned_ffn_preference']
                    assert event['ffn_gradient_scope'] == 'source_FFN_output_to_final_log_probability'
                    assert event['ffn_positive_coordinates'] >= 0
                    assert event['ffn_positive_attribution'] >= 0 and event['ffn_negative_attribution'] >= 0
                    assert 0 <= event['reference_ffn_drift'] <= 1
                    assert math.isfinite(event['ffn_preference_l1_change']) and 0 <= event['ffn_preference_l1_change'] <= 2.000001
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                    assert event['generated_token'] == tokens[step]
                    if not event['accepted']:
                        assert tokens[step] == event['token']
                    report['native_prefix_events'] += int(verify_native_state_prefix(old, tokens, events, event))
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                    report['maximum_preference_l1_change'] = max(report['maximum_preference_l1_change'], event['ffn_preference_l1_change'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == old
                    report['no_retained_exact_sadi'] += 1
    assert report['images'] == plan['images']
    write(root / 'target_ffn_runtime_audit.json', report)
    print(report)


def audit(root):
    with patch.object(classes, 'physical_audit', physical_audit):
        classes.audit(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
