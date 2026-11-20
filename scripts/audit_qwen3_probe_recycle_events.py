"""Check saved detector gates and actual accepted/rejected recycling events."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write


def audit(root):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    preflight_path = root / 'early_preflight.json'
    if not preflight_path.exists():
        preflight_path = Path(plan['actuation_preflight'])
    preflight = read(preflight_path)
    assert preflight['passed']
    assert preflight['checked_current_KV_layers'] == list(range(20, 36))
    assert all(preflight['cases'][k]['exact_native_logits_tokens_and_current_tail_KV']
               for k in ('disabled', 'zero', 'reject'))
    trace = Path(plan['source_trace'])
    labels = {(r['image_id'], r['step']): r['mentions'] for shard in (0, 1)
              for r in map(json.loads, (trace / f'features.shard{shard}.jsonl').read_bytes().splitlines())}
    allowed = set(read(root / 'object_lexicon.json')['token_ids'])
    baseline = {r['image_id']: r for shard in (0, 1)
                for r in map(json.loads, (root / plan['cohort'] / f'sadi.shard{shard}.jsonl').read_bytes().splitlines())}
    report = dict(passed=True, images=0, proposals=0, accepted=0, immediate_changes=[],
                  maximum_probe_hidden_difference=0., native_prefix_events=0,
                  cached_label_scope='Old100 trace positions only; no label is inferred for other image IDs.')
    name, = plan['candidates']
    for shard in (0, 1):
        for row in map(json.loads, (root / plan['cohort'] / f'{name}.shard{shard}.jsonl').read_bytes().splitlines()):
            report['images'] += 1
            stats, tokens = row['sar_ems_statistics'], row['response_token_ids']
            assert stats['intervention_layer'] == 19 and stats['replayed_tail_layers'] == list(range(20, 36))
            assert stats['maximum_probe_hidden_difference'] == 0
            assert not any(stats.get(k) for k in ('diagnostic_reject_all', 'diagnostic_zero_replay',
                                                'diagnostic_force_probe_gate'))
            events = stats['recycle_events']
            assert len(events) == stats['guided_attention_calls']
            assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
            assert len({e['step'] for e in events}) == len(events)
            for event in events:
                step, target = event['step'], event['token']
                assert target in allowed and target not in tokens[:step]
                assert event['source_layer'] == event['target_layer'] == 19
                assert event['probe_p_real'] < .5 and event['predicted_log_probability_change'] <= 1e-6
                assert event['moved_attention_mass'] > 0 and 0 < event['drift'] <= 1
                assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                assert event['generated_token'] == tokens[step]
                if not event['accepted']:
                    assert tokens[step] == target
                old = baseline[row['image_id']]['response_token_ids']
                matches = old[:step] == tokens[:step]
                if matches:
                    assert old[step] == target
                    report['native_prefix_events'] += 1
                report['proposals'] += 1
                report['accepted'] += int(event['accepted'])
                if tokens[step] != target:
                    report['immediate_changes'].append(dict(image_id=row['image_id'], step=step,
                        old_token=target, new_token=tokens[step], old=tokenizer.decode([target]),
                        new=tokenizer.decode(tokens[step:step + 8]),
                        context=tokenizer.decode(tokens[max(0, step - 18):step]),
                        prefix_matches_baseline=matches,
                        baseline_mentions=labels.get((row['image_id'], step)) if matches else None,
                        **{k: event[k] for k in ('probe_p_real', 'visual_support', 'gain', 'drift',
                                                'actual_log_probability_change')}))
    assert report['images'] == plan['images']
    write(root / 'recycle_event_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
