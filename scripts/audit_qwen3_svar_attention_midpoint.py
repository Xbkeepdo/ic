"""Check conservative proposal metadata, native rollback and every prefix class."""

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
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, write


def audit(root):
    from transformers import AutoTokenizer
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    for filename, expected in plan['source_sha256'].items():
        assert sha(Path(filename)) == expected, filename
    preflight = read(Path(plan['actuation_preflight']))
    assert preflight['passed'] and preflight['baseline_variant'] == 'svar'
    assert preflight['checked_current_KV_layers'] == list(range(20, 36))
    assert plan['geometry_modes'] == ['half', 'geometric']
    for mode in plan['geometry_modes']:
        for kind in ('disabled', 'zero', 'reject'):
            case = preflight['cases'][mode + '_' + kind]
            assert case['exact_native_logits_tokens_and_current_tail_KV']
        assert preflight['cases'][mode + '_live']['proposals'] > 0
        assert all(preflight['cases'][mode + '_reject']['forward_counts'][key] <= count
                   for key, count in preflight['parent_reject_forward_counts'].items())
    source = Path(plan['source'])
    assert (root / 'object_lexicon.json').read_bytes() == (source / 'object_lexicon.json').read_bytes()
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root / 'object_lexicon.json')['token_ids'])
    baseline = {r['image_id']: r for shard in (0, 1) for group in source_groups(
        root / plan['cohort'] / f'svar.shard{shard}.jsonl') for r in group}
    results = {}
    for name, mode in zip(plan['candidates'], plan['geometry_modes'], strict=True):
        report = dict(images=0, proposals=0, accepted=0, no_retained_exact_svar=0,
            gradient_probe_calls=0, geometry_calls=0, class_decisions=0, class_blocked=0,
            reference_moved_mass=0., actual_moved_mass=0., maximum_mass_error=0.,
            maximum_roundoff_correction=0., first_order_record_is_bound_not_prediction=True)
        for shard in (0, 1):
            for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
                counts = []
                for field in ('gradient_probe_calls', 'geometry_calls'):
                    values = {r['sar_ems_statistics'][field] for r in group}
                    assert len(values) == 1
                    counts.append(values.pop())
                assert counts[1] <= counts[0]
                report['gradient_probe_calls'] += counts[0]
                report['geometry_calls'] += counts[1]
                for row in group:
                    report['images'] += 1
                    tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
                    assert stats['base_variant'] == stats['local_I_before_base_edit'] == 'svar'
                    assert stats['attention_geometry'] == mode and stats['midpoint_weight'] == .5
                    assert stats['unchanged_signed_support'] and stats['prompt_attention_fixed']
                    assert stats['intervention_layer'] == 19 and stats['replayed_tail_layers'] == list(range(20, 36))
                    assert stats['gradient_scope'] == 'final' and stats['maximum_probe_hidden_difference'] == 0
                    assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                                                        'diagnostic_reject_all', 'diagnostic_zero_replay'))
                    assert all(stats[k] == 0 for k in ('additional_gradient_probes', 'additional_model_forward_calls',
                                                     'additional_mlp_forward_calls', 'additional_lm_head_calls'))
                    decisions = {d['step']: d for d in stats['class_decisions']}
                    assert len(decisions) == len(stats['class_decisions'])
                    for step, decision in decisions.items():
                        target = decision['token']
                        assert target in classes and target not in tokens[:step]
                        prior = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False))
                        assert decision['canonical'] == classes[target] and decision['prior_classes'] == sorted(prior)
                        assert decision['blocked'] == (classes[target] in prior)
                        if decision['blocked']:
                            assert tokens[step] == target
                        report['class_decisions'] += 1
                        report['class_blocked'] += int(decision['blocked'])
                    events = stats['recycle_events']
                    assert len(events) == stats['guided_attention_calls'] == stats['measured_log_probability_changes']
                    assert len({e['step'] for e in events}) == len(events)
                    assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                    for event in events:
                        step, target = event['step'], event['token']
                        assert not decisions[step]['blocked'] and decisions[step]['token'] == target
                        assert event['source_layer'] == event['target_layer'] == 19
                        assert event['probe_p_real'] < .5 and 0 < event['drift'] <= 1
                        assert event['attention_geometry'] == mode
                        assert 'predicted_log_probability_change' not in event
                        assert event['reference_predicted_log_probability_change'] <= 1e-6
                        assert event['first_order_log_probability_change_upper_bound'] == 0
                        assert event['first_order_bound_from_unchanged_signed_support']
                        assert 0 <= event['mass_conservation_error'] < 1e-6
                        assert 0 <= event['midpoint_roundoff_correction'] < 1e-6
                        assert event['moved_attention_mass'] > 0 and event['reference_moved_attention_mass'] > 0
                        assert event['budget_max'] > 0 and event['reference_budget_max'] > 0
                        if mode == 'half':
                            assert abs(event['budget_max'] - .5 * event['reference_budget_max']) < 1e-7
                            assert abs(event['moved_attention_mass'] - .5 * event['reference_moved_attention_mass']) < 1e-5
                        assert math.isfinite(event['actual_log_probability_change'])
                        assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                        assert event['generated_token'] == tokens[step]
                        if not event['accepted']:
                            assert tokens[step] == target
                        report['proposals'] += 1
                        report['accepted'] += int(event['accepted'])
                        report['reference_moved_mass'] += event['reference_moved_attention_mass']
                        report['actual_moved_mass'] += event['moved_attention_mass']
                        report['maximum_mass_error'] = max(report['maximum_mass_error'], event['mass_conservation_error'])
                        report['maximum_roundoff_correction'] = max(report['maximum_roundoff_correction'], event['midpoint_roundoff_correction'])
                    if not stats['accepted_guidance_calls']:
                        assert tokens == baseline[row['image_id']]['response_token_ids']
                        report['no_retained_exact_svar'] += 1
        assert report['images'] == plan['images']
        results[name] = report
    result = dict(passed=True, methods=results)
    write(root / 'attention_midpoint_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
