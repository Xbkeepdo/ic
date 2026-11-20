"""Verify each feature-blind gate, class history and retained attention event."""

import argparse
import hashlib
import json
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
    assert plan['gate_seeds'] == [43, 44, 45]
    assert plan['calibration']['positives'] == 23 and plan['calibration']['eligible'] == 208
    assert plan['calibration']['rate'] == 23 / 208
    preflight = read(Path(plan['actuation_preflight']))
    assert preflight['passed']
    assert preflight['checked_current_KV_layers'] == list(range(20, 36))
    assert all(preflight['cases'][k]['exact_native_logits_tokens_and_current_tail_KV']
               for k in ('disabled', 'zero', 'reject'))
    source = Path(plan['source'])
    assert (root / 'object_lexicon.json').read_bytes() == (source / 'object_lexicon.json').read_bytes()
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root / 'object_lexicon.json')['token_ids'])
    baseline = {row['image_id']: row for shard in (0, 1)
                for group in source_groups(root / f'coco100/sadi.shard{shard}.jsonl') for row in group}
    results = {}
    for name, seed in zip(plan['candidates'], plan['gate_seeds'], strict=True):
        summary = dict(images=0, eligible=0, hash_positive=0, detector_positive=0,
                       proposals=0, accepted=0, proposals_detector_would_block=0,
                       no_retained_action_matches_sadi=0)
        for shard in (0, 1):
            for group in source_groups(root / f'coco100/{name}.shard{shard}.jsonl'):
                for row in group:
                    summary['images'] += 1
                    tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
                    assert stats['gate_type'] == 'feature_blind_matched_hash'
                    assert not stats['detector_used_for_selection'] and stats['probe_real_threshold'] is None
                    assert stats['gate_seed'] == seed and stats['gate_rate'] == 23 / 208
                    assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                                                        'diagnostic_reject_all', 'diagnostic_zero_replay'))
                    assert stats['intervention_layer'] == 19 and stats['replayed_tail_layers'] == list(range(20, 36))
                    assert stats['maximum_probe_hidden_difference'] == 0
                    classes_by_step = {d['step']: d for d in stats['class_decisions']}
                    assert len(classes_by_step) == len(stats['class_decisions'])
                    for step, decision in classes_by_step.items():
                        target = decision['token']
                        assert target in classes and target not in tokens[:step]
                        prior = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False))
                        assert decision['canonical'] == classes[target]
                        assert decision['prior_classes'] == sorted(prior)
                        assert decision['blocked'] == (classes[target] in prior)
                    hashes = {d['step']: d for d in stats['hash_gate_decisions']}
                    assert len(hashes) == len(stats['hash_gate_decisions']) == stats['probe_calls']
                    for step, decision in hashes.items():
                        target = decision['token']
                        assert not classes_by_step[step]['blocked'] and classes_by_step[step]['token'] == target
                        raw = f'{seed}|{row["image_id"]}|{step}|{target}'.encode('ascii')
                        uniform = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64
                        assert uniform == decision['uniform']
                        assert decision['permitted'] == (uniform < 23 / 208)
                        assert 0 <= decision['detector_p_real'] <= 1
                        summary['eligible'] += 1
                        summary['hash_positive'] += int(decision['permitted'])
                        summary['detector_positive'] += int(decision['detector_p_real'] < .5)
                    assert sum(d['permitted'] for d in hashes.values()) == stats['hash_gate_positive_calls']
                    assert sum(d['detector_p_real'] < .5 for d in hashes.values()) == stats['probe_positive_calls']
                    events = stats['recycle_events']
                    assert len(events) == stats['guided_attention_calls']
                    assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                    for event in events:
                        step, target = event['step'], event['token']
                        decision = hashes[step]
                        assert decision['permitted'] and decision['token'] == target
                        assert event['probe_p_real'] == decision['detector_p_real']
                        assert event['gate_score'] == 0
                        assert event['source_layer'] == event['target_layer'] == 19
                        assert event['predicted_log_probability_change'] <= 1e-6
                        assert event['moved_attention_mass'] > 0 and 0 < event['drift'] <= 1
                        assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                        assert event['generated_token'] == tokens[step]
                        if not event['accepted']:
                            assert tokens[step] == target
                        summary['proposals'] += 1
                        summary['accepted'] += int(event['accepted'])
                        summary['proposals_detector_would_block'] += int(event['probe_p_real'] >= .5)
                    if not stats['accepted_guidance_calls']:
                        assert tokens == baseline[row['image_id']]['response_token_ids']
                        summary['no_retained_action_matches_sadi'] += 1
        assert summary['images'] == plan['images']
        results[name] = summary
    report = dict(passed=True, seeds=plan['gate_seeds'], calibration=plan['calibration'], methods=results)
    write(root / 'matched_hash_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
