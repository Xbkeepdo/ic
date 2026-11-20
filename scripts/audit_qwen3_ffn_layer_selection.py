"""Recompute all layer choices and verify their physical/class invariants."""

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import audit_qwen3_probe_first_class as classes
from scripts.audit_qwen3_retained_prefix_state import verify_native_state_prefix
from scripts.run_qwen3_signed_feedback import read, source_groups, write

LAYERS = (19, 23, 27, 31)


def physical_audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    preflight = read(Path(plan['actuation_preflight']))
    assert preflight['passed'] and preflight['baseline_variant'] == 'sadi'
    assert preflight['base_layers_inclusive'] == [5, 18] and preflight['eligible_layers'] == list(LAYERS)
    for mode in ('disabled', 'zero', 'reject19', 'reject31'):
        assert preflight['cases'][mode]['exact_native_logits_tokens_and_current_tail_KV']
    assert preflight['cases']['zero']['selected_layers'] == list(LAYERS)
    for mode in ('reject19', 'reject31'):
        counts, parent = preflight['cases'][mode]['forward_counts'], preflight['parent_reject_forward_counts']
        assert counts['model'] == parent['model'] and counts['backward'] == parent['backward']
        assert counts['lm_head'] <= parent['lm_head'] and all(counts[f'ffn{i}'] <= parent[f'ffn{i}'] for i in LAYERS)
    for name, source in plan['additional_copied_controls'].items():
        for shard in (0, 1):
            filename = f'{name}.shard{shard}.jsonl'
            assert (root / plan['cohort'] / filename).read_bytes() == (Path(source) / plan['cohort'] / filename).read_bytes()
    name, = plan['candidates']
    baseline = {r['image_id']: r for shard in (0, 1)
        for group in source_groups(root / plan['cohort'] / f'sadi.shard{shard}.jsonl') for r in group}
    report = dict(passed=True, images=0, proposals=0, accepted=0, gradient_calls=0,
        decisions=0, no_positive_score_decisions=0, no_retained_exact_sadi=0, native_prefix_events=0)
    choices, kept = Counter(), Counter()
    for shard in (0, 1):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            calls = {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            assert len(calls) == 1
            report['gradient_calls'] += calls.pop()
            checked = {r['sar_ems_statistics']['native_selected_attention_checks'] for r in group}
            assert len(checked) == 1
            assert checked.pop() == sum(e['selected_layer'] != 19 for r in group for e in r['sar_ems_statistics']['recycle_events'])
            for row in group:
                report['images'] += 1
                stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
                assert stats['base_variant'] == stats['local_I_before_base_edit'] == 'sadi'
                assert stats['eligible_layers'] == stats['gradient_sources'] == list(LAYERS)
                assert stats['selection_mode'] == 'adaptive' and stats['diagnostic_force_layer'] is None
                assert stats['intervention_layer'] == 'selected_per_event' and stats['replayed_tail_layers'] == list(range(20, 36))
                assert stats['maximum_probe_hidden_difference'] == 0 and stats['one_backward_for_all_layers']
                assert stats['extra_model_forward_calls'] == stats['extra_backward_calls'] == 0 and not stats['ffn_update_edited']
                assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                    'diagnostic_zero_replay', 'diagnostic_reject_all'))
                decisions = {d['step']: d for d in stats['layer_decisions']}
                assert len(decisions) == len(stats['layer_decisions'])
                for decision in decisions.values():
                    options = {int(i): item for i, item in decision['candidates'].items()}
                    assert set(options) == set(LAYERS)
                    for item in options.values():
                        assert all(math.isfinite(item[key]) for key in ('score', 'ffn_target_push', 'negative_visual_fraction', 'drift', 'predicted_log_probability_change'))
                        assert 0 <= item['negative_visual_fraction'] <= 1 and 0 <= item['drift'] <= 1
                        expected = max(item['ffn_target_push'], 0)*item['negative_visual_fraction']*item['drift']*item['valid_recycle']
                        assert abs(expected-item['score']) <= 1e-6*max(1, abs(expected))
                        assert item['predicted_log_probability_change'] <= 1e-6
                    chosen = max(LAYERS, key=lambda i: options[i]['score'])
                    if options[chosen]['score'] <= 0:
                        chosen = None
                        report['no_positive_score_decisions'] += 1
                    assert chosen == decision['selected_layer']
                    report['decisions'] += 1
                events = stats['recycle_events']
                assert len(events) == stats['guided_attention_calls']
                assert len({e['step'] for e in events}) == len(events)
                assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                assert {step for step, d in decisions.items() if d['selected_layer'] is not None} == {e['step'] for e in events}
                old = baseline[row['image_id']]['response_token_ids']
                for event in events:
                    step, layer = event['step'], event['selected_layer']
                    assert event['source_layer'] == event['target_layer'] == layer == decisions[step]['selected_layer']
                    assert event['token'] == decisions[step]['token']
                    assert event['probe_p_real'] < .5 and event['selection_score'] > 0 and event['ffn_target_push'] > 0
                    assert event['predicted_log_probability_change'] <= 1e-6 and event['moved_attention_mass'] > 0
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                    assert event['generated_token'] == tokens[step]
                    if not event['accepted']:
                        assert tokens[step] == event['token']
                    report['native_prefix_events'] += int(verify_native_state_prefix(old, tokens, events, event))
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                    choices[layer] += 1
                    kept[layer] += int(event['accepted'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == old
                    report['no_retained_exact_sadi'] += 1
    assert report['images'] == plan['images']
    report.update(selected_layers=dict(choices), accepted_layers=dict(kept))
    write(root / 'ffn_layer_selection_runtime_audit.json', report)
    print(report)


def audit(root):
    with patch.object(classes, 'physical_audit', physical_audit):
        classes.audit(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
