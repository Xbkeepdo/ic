"""Check native state only before any earlier retained cache intervention."""

from scripts import audit_qwen3_svar_first_class as reference
from scripts.run_qwen3_signed_feedback import source_groups
from pathlib import Path


def verify_native_state_prefix(native, generated, events, event):
    """Identical emitted tokens do not undo previously changed KV states."""
    step = event['step']
    if any(e['accepted'] and e['step'] < step for e in events):
        return False
    assert native[:step] == generated[:step], 'Token prefix changed before any retained action.'
    assert native[step] == event['token'], 'Native target differs before any retained action.'
    return True


def physical_reference_audit(root):
    # Read/write through reference so its caller's immutable single-arm view
    # and reference_-prefixed outputs remain in force.
    plan = reference.read(root / 'protocol.json')
    assert reference.read(root / 'run_status.json')['status'] == 'complete'
    preflight = reference.read(Path(plan['actuation_preflight']))
    assert preflight['passed'] and preflight['baseline_variant'] == 'svar'
    assert preflight['checked_current_KV_layers'] == list(range(20, 36))
    assert all(preflight['cases'][key]['exact_native_logits_tokens_and_current_tail_KV'] for key in ('disabled', 'zero', 'reject'))
    semantic = reference.read(Path(plan['semantic_preflight']))
    assert semantic['passed'] and semantic['exact_E_l'] and semantic['max_abs_difference'] == 0
    baseline = {row['image_id']: row for shard in range(plan['shards'])
                for group in source_groups(root / plan['cohort'] / f'svar.shard{shard}.jsonl') for row in group}
    name, = plan['candidates']
    report = dict(passed=True, primary_baseline='svar', images=0, proposals=0, accepted=0,
                  native_prefix_events=0, no_retained_action_matches_svar=0,
                  native_state_rule='Native KV is guaranteed only before an earlier retained intervention.')
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
                    report['native_prefix_events'] += int(verify_native_state_prefix(old, tokens, events, event))
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == old
                    report['no_retained_action_matches_svar'] += 1
    assert report['images'] == plan['images']
    reference.write(root / 'svar_recycle_event_audit.json', report)
    print(report)
