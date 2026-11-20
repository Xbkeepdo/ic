"""Verify fixed-evidence FFN feedback, accepted events, prefix classes and provenance."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

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
    assert all(preflight['cases'][key]['exact_native_logits_tokens_and_current_tail_KV']
               for key in ('disabled', 'zero', 'reject', 'feedback_reject'))
    assert preflight['cases']['feedback_reject']['feedback_rejections'] > 0
    assert preflight['cases']['live']['proposals'] > 0
    source = Path(plan['source'])
    assert (root / 'object_lexicon.json').read_bytes() == (source / 'object_lexicon.json').read_bytes()
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root / 'object_lexicon.json')['token_ids'])
    baseline = {r['image_id']: r for shard in (0, 1) for group in source_groups(
        root / plan['cohort'] / f'svar.shard{shard}.jsonl') for r in group}
    name, = plan['candidates']
    report = dict(passed=True, images=0, batches=0, proposals=0, accepted=0, probability_decreases=0,
        rejected_ffn_alignment=0, no_retained_exact_svar=0, gradient_probe_calls=0,
        fixed_evidence_reads=0, alignment_readouts=0, class_decisions=0, class_blocked=0,
        maximum_recomputed_drift_error=0., minimum_accepted_drift_reduction=None,
        additional_model_forward_calls=0, additional_mlp_forward_calls=0, additional_lm_head_calls=0)
    for shard in (0, 1):
        for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl'):
            report['batches'] += 1
            counts = []
            for field in ('gradient_probe_calls', 'fixed_evidence_reads', 'ffn_alignment_readouts'):
                values = {r['sar_ems_statistics'][field] for r in group}
                assert len(values) == 1
                counts.append(values.pop())
            assert counts[0] == counts[1]
            report['gradient_probe_calls'] += counts[0]
            report['fixed_evidence_reads'] += counts[1]
            report['alignment_readouts'] += counts[2]
            assert len({e['step'] for r in group for e in
                        r['sar_ems_statistics']['recycle_events']}) <= counts[2] <= counts[0]
            for row in group:
                report['images'] += 1
                tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
                assert stats['base_variant'] == stats['local_I_before_base_edit'] == 'svar'
                assert stats['acceptance_requires_ffn_alignment'] and stats['fixed_negative_evidence_before_proposal']
                assert stats['ffn_alignment_layer'] == stats['intervention_layer'] == 19
                assert stats['replayed_tail_layers'] == list(range(20, 36))
                assert stats['maximum_probe_hidden_difference'] == 0
                assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                    'diagnostic_reject_all', 'diagnostic_zero_replay', 'diagnostic_reject_ffn_alignment'))
                assert all(stats[k] == 0 for k in ('additional_model_forward_calls',
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
                assert len(events) == stats['guided_attention_calls']
                assert stats['gradient_scope'] == 'final'
                assert stats['measured_log_probability_changes'] == len(events)
                expected_mean = sum(e['actual_log_probability_change'] for e in events) / max(1, len(events))
                assert abs(stats['mean_log_probability_change'] - expected_mean) < 1e-10
                assert len({e['step'] for e in events}) == len(events)
                assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                # The new acceptance is stricter than probability decrease.
                # Retain the raw probability-decrease count instead of relabeling it.
                raw_decreases = sum(e['actual_log_probability_change'] < 0 for e in events)
                assert raw_decreases == stats['probability_decrease_proposals'] == stats['log_probability_decreases']
                assert sum(e['rejected_ffn_alignment'] for e in events) == stats['rejected_ffn_alignment_calls']
                assert sum(e['ffn_drift_improved'] for e in events) == stats['ffn_drift_improves_in_proposals']
                for event in events:
                    step, target = event['step'], event['token']
                    assert not decisions[step]['blocked'] and decisions[step]['token'] == target
                    assert event['source_layer'] == event['target_layer'] == 19
                    assert event['probe_p_real'] < .5 and event['predicted_log_probability_change'] <= 1e-6
                    assert event['moved_attention_mass'] > 0 and 0 < event['drift'] <= 1
                    assert event['fixed_negative_evidence'] and not event['diagnostic_reject_ffn_alignment']
                    q = np.asarray(event['ffn_negative_evidence_before'], dtype=np.float32)
                    p = np.asarray(event['ffn_preference_after'], dtype=np.float32)
                    assert q.shape == p.shape == (row['visual_token_count'],)
                    assert np.isfinite(q).all() and np.isfinite(p).all() and (q >= 0).all() and (p >= 0).all()
                    assert abs(float(q.sum()) - 1) < 1e-6 and abs(float(p.sum()) - 1) < 1e-6
                    recomputed = float(np.clip(1 - np.sqrt(q * p).sum(), 0, 1))
                    error = abs(recomputed - event['ffn_drift_after'])
                    assert error < 1e-6
                    report['maximum_recomputed_drift_error'] = max(report['maximum_recomputed_drift_error'], error)
                    assert math.isfinite(event['actual_log_probability_change'])
                    decreased = event['actual_log_probability_change'] < 0
                    improved = event['ffn_drift_after'] < event['drift']
                    assert event['probability_decrease_accepted'] == decreased
                    assert event['ffn_drift_improved'] == improved
                    assert event['accepted'] == (decreased and improved)
                    assert event['rejected_ffn_alignment'] == (decreased and not improved)
                    assert event['generated_token'] == tokens[step]
                    if not event['accepted']:
                        assert tokens[step] == target
                    else:
                        delta = event['drift'] - event['ffn_drift_after']
                        old = report['minimum_accepted_drift_reduction']
                        report['minimum_accepted_drift_reduction'] = delta if old is None else min(old, delta)
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                    report['probability_decreases'] += int(decreased)
                    report['rejected_ffn_alignment'] += int(event['rejected_ffn_alignment'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == baseline[row['image_id']]['response_token_ids']
                    report['no_retained_exact_svar'] += 1
    assert report['images'] == plan['images']
    write(root / 'ffn_alignment_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
