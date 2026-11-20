"""Post-hoc old100 replay acceptance on identical pre-intervention histories."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, output):
    assert not output.exists(), 'Preserve earlier diagnostic outputs'
    plan = read(root/'protocol.json')
    assert plan['images'] == 100 and read(root/'final_provenance.json')['passed']
    assert read(root/'post_audit_status.json')['status'] == 'passed'
    names = ['causal_fixed', 'causal_native', 'causal_refresh']
    rows, hashes = {}, {str(root/'protocol.json'): sha(root/'protocol.json')}
    for name in names:
        rows[name] = {}
        for shard in (0, 1):
            path = root/f'coco100/{name}.shard{shard}.jsonl'
            hashes[str(path)] = sha(path)
            for row in map(json.loads, path.read_bytes().splitlines()):
                assert row['image_id'] not in rows[name]
                rows[name][row['image_id']] = row
    assert all(set(rows[name]) == set(rows[names[0]]) for name in names)
    records = []
    for image in sorted(rows[names[0]]):
        values = {name: rows[name][image] for name in names}
        events = {name: {e['step']: e for e in row['sar_ems_statistics']['full_joint_events']}
                  for name, row in values.items()}
        first = min((event['step'] for table in events.values() for event in table.values()
                     if event['accepted']), default=513)
        common = set.intersection(*(set(table) for table in events.values()))
        for step in sorted(common):
            # Same emitted text does not ensure the same KV history after an
            # accepted hidden-state change. Exclude every step after any keep.
            if step > first:
                continue
            selected = {name: table[step] for name, table in events.items()}
            tokens = {name: row['response_token_ids'][:step] for name, row in values.items()}
            assert all(tokens[name] == tokens[names[0]] for name in names)
            assert len({event['token'] for event in selected.values()}) == 1
            assert len({event['probe_p_real'] for event in selected.values()}) == 1
            classes = {name: next(d['canonical'] for d in row['sar_ems_statistics']['full_class_decisions']
                                  if d['step'] == step) for name, row in values.items()}
            assert len(set(classes.values())) == 1
            measurements = {}
            for name, event in selected.items():
                layers = list(event['layer_details'].values())
                measurements[name] = dict(accepted=event['accepted'],
                    actual_log_probability_change=event['actual_log_probability_change'],
                    immediate_token_changed=event['immediate_token_changed'],
                    proposed_token=event['proposed_token'],
                    mean_coefficient=sum(v['coefficient_mean'] for v in layers)/len(layers),
                    mean_drift=sum(v['drift'] for v in layers)/len(layers),
                    max_coefficient_refresh_change=max(v['coefficient_native_max_difference'] for v in layers))
            records.append(dict(image_id=image, step=step, canonical=next(iter(classes.values())),
                token=selected[names[0]]['token'], p_real=selected[names[0]]['probe_p_real'],
                before_any_prior_accepted_action=True, methods=measurements))
    counts = {name: dict(proposals=len(records), accepted=sum(r['methods'][name]['accepted'] for r in records),
                        same_step_token_changes=sum(r['methods'][name]['accepted'] and
                            r['methods'][name]['immediate_token_changed'] for r in records)) for name in names}
    contrasts = {}
    for name in names[1:]:
        contrasting = [r for r in records if r['methods']['causal_fixed']['accepted'] != r['methods'][name]['accepted']]
        contrasts[name] = dict(fixed_only=sum(r['methods']['causal_fixed']['accepted'] for r in contrasting),
            evidence_only=sum(r['methods'][name]['accepted'] for r in contrasting),
            image_step_pairs=[[r['image_id'], r['step']] for r in contrasting])
    result = dict(passed=True, created=time.time(), source_sha256=sha(Path(__file__)), input_sha256=hashes,
        scope='Post-hoc mechanism analysis of the reused development100 only; no new generation or validation tuning.',
        matched_scope='Same row token prefix and no earlier accepted hidden/KV action in any method; identical target, p_real and class.',
        limitation='Matched replay effects establish differences between operators/budgets, not image truth or mitigation quality.',
        counts=counts, acceptance_contrasts=contrasts, records=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(counts=counts, contrasts=contrasts,
                         repeated_table_case=[r for r in records if r['image_id'] == 243882]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)
