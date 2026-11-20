"""Independently check saved rival gates, bounded trials and actual emissions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    preflight = read(root / 'rival_preflight.json')
    assert preflight['passed'] and not preflight['production_force_rival']
    assert all(v['exact_native_logits_tokens_and_current_tail_KV']
               for k, v in preflight['cases'].items() if k != 'live_guided')
    allowed = set(read(root / 'object_lexicon.json')['token_ids'])
    trace = Path(plan['source_trace'])
    labels = {(r['image_id'], r['step']): r['mentions'] for shard in (0, 1)
              for r in map(json.loads, (trace / f'features.shard{shard}.jsonl').read_bytes().splitlines())}
    source = Path(plan['source'])
    baseline = {r['image_id']: r['response_token_ids'] for shard in (0, 1)
                for r in map(json.loads, (source / f'coco100/sadi.shard{shard}.jsonl').read_bytes().splitlines())}
    name, = plan['candidates']
    groups = [group for shard in (0, 1)
              for group in source_groups(root / plan['cohort'] / f'{name}.shard{shard}.jsonl')]
    report = dict(passed=True, images=0, candidates=0, selected_rivals=0, proposals=0,
                  accepted=0, tail_trial_forwards=0, gradient_probes=0, reasons=Counter(),
                  maximum_probe_hidden_difference=0., no_action_captions_equal_baseline=0, changed_events=[])
    for group in groups:
        for key, field in (('gradient_probes', 'gradient_probe_calls'),
                           ('tail_trial_forwards', 'attention_trial_forward_calls')):
            values = {r['sar_ems_statistics'][field] for r in group}
            assert len(values) == 1
            report[key] += values.pop()
        for row in group:
            report['images'] += 1
            stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
            assert stats['grounded_rival'] and not stats['diagnostic_force_rival']
            assert not any(stats.get(k) for k in ('diagnostic_reject_all', 'diagnostic_zero_replay',
                                                'diagnostic_force_probe_gate'))
            assert stats['maximum_probe_hidden_difference'] == 0
            choices = {e['step']: e for e in stats['rival_choices']}
            assert len(choices) == len(stats['rival_choices'])
            report['candidates'] += len(choices)
            for step, choice in choices.items():
                target = choice['target_token']
                assert target in allowed and target not in tokens[:step]
                report['reasons'][choice['reason']] += 1
                if choice['selected']:
                    assert choice['probe_p_real'] < .5 and choice['rival_token'] in allowed
                    assert choice['rival_token'] != target and choice['rival_p_real'] >= .5
                    assert choice['rival_visual_support'] > 0
                    assert choice['rival_visual_support'] > choice['target_visual_support']
                    assert 0 <= choice['native_logit_gap'] <= 2
                    report['selected_rivals'] += 1
            events = stats['recycle_events']
            if not events:
                assert tokens == baseline[row['image_id']]
                report['no_action_captions_equal_baseline'] += 1
            assert len(events) == stats['guided_attention_calls']
            assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
            assert len(events) - stats['accepted_guidance_calls'] == stats['target_match_events']
            for event in events:
                step, target, rival = event['step'], event['token'], event['rival_token']
                assert choices[step]['selected']
                assert choices[step]['target_token'] == target and choices[step]['rival_token'] == rival
                assert target not in tokens[:step] and event['probe_p_real'] < .5
                assert event['predicted_margin_change'] <= 1e-6
                assert event['moved_attention_mass'] > 0 and 0 < event['drift'] <= 1
                trials = event['trials']
                assert 1 <= len(trials) <= 3
                assert [t['multiplier'] for t in trials] == [1., 2., 4.][:len(trials)]
                assert all(not t['accepted'] for t in trials[:-1])
                for trial in trials:
                    assert trial['accepted'] == (trial['predicted_token'] == rival
                                                   and trial['margin'] < event['native_logit_gap'])
                assert event['accepted'] == trials[-1]['accepted']
                assert event['generated_token'] == tokens[step] == (rival if event['accepted'] else target)
                assert abs(event['actual_margin_change'] -
                           (trials[-1]['margin'] - event['native_logit_gap'])) < 1e-6
                report['proposals'] += 1
                if event['accepted']:
                    report['accepted'] += 1
                    matches = baseline[row['image_id']][:step] == tokens[:step]
                    report['changed_events'].append(dict(image_id=row['image_id'], step=step,
                        target_token=target, rival_token=rival, prefix_matches_baseline=matches,
                        baseline_mentions=labels.get((row['image_id'], step)) if matches else None,
                        **{k: event[k] for k in ('probe_p_real', 'rival_probe_p_real', 'visual_support',
                           'rival_visual_support', 'native_logit_gap', 'actual_margin_change')}))
    assert report['images'] == plan['images']
    write(root / 'rival_event_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
