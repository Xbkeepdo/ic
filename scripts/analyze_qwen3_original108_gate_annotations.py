"""Describe original108 gate decisions against COCO annotations on the old100."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, output):
    from transformers import AutoTokenizer
    assert not output.exists(), 'Preserve existing diagnostic outputs'
    plan = read(root/'protocol.json')
    assert plan['images'] == 100 and read(root/'final_provenance.json')['passed']
    summary = read(root/'independent_summary.json')
    metrics_path, = (root/'evaluations/sentence_v1/coco100').glob('metrics.*.json')
    annotation_path = Path(read(metrics_path)['chair_cache'])
    annotations = read(annotation_path)['objects']
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    names = ['causal_fixed', 'causal_native', 'causal_refresh']
    rows = {}; inputs = {str(metrics_path): sha(metrics_path), str(annotation_path): sha(annotation_path),
                         str(root/'protocol.json'): sha(root/'protocol.json')}
    for name in ['sadi', *names]:
        rows[name] = {}
        for shard in (0, 1):
            path = root/f'coco100/{name}.shard{shard}.jsonl'
            inputs[str(path)] = sha(path)
            assert inputs[str(path)] == summary['caption_sha256'][str(path)]
            for row in map(json.loads, path.read_bytes().splitlines()):
                assert row['image_id'] not in rows[name]
                rows[name][row['image_id']] = row
    ids = read(root/'cohort.json')['coco100']
    assert all(set(rows[name]) == set(ids) for name in rows)
    records, exclusions = [], []
    for image in ids:
        values = {name: rows[name][image] for name in names}
        stats = {name: row['sar_ems_statistics'] for name, row in values.items()}
        cut = min((e['step'] for stat in stats.values() for e in stat['full_joint_events'] if e['accepted']), default=513)
        gates = {name: {g['step']: g for g in stat['full_gate_events']} for name, stat in stats.items()}
        common = set.intersection(*(set(table) for table in gates.values()))
        for step in sorted(common):
            if step > cut:
                continue
            selected = {name: table[step] for name, table in gates.items()}
            prefixes = [values[name]['response_token_ids'][:step] for name in names]
            assert all(prefix == prefixes[0] for prefix in prefixes)
            assert prefixes[0] == rows['sadi'][image]['response_token_ids'][:step]
            assert len({g['token'] for g in selected.values()}) == 1
            target = selected[names[0]]['token']
            assert target == rows['sadi'][image]['response_token_ids'][step]
            probabilities = {name: g['p_real'] for name, g in selected.items()}
            if len(set(probabilities.values())) != 1:
                exclusions.append(dict(image_id=image, step=step, reason='Probe probabilities not bitwise equal.', probabilities=probabilities))
                continue
            decisions = [next(d for d in stat['full_class_decisions'] if d['step'] == step) for stat in stats.values()]
            assert len({d['canonical'] for d in decisions}) == 1
            canonical = decisions[0]['canonical']
            gate = selected[names[0]]
            assert all(g['positive'] == gate['positive'] and g['threshold'] == plan['real_threshold'] for g in selected.values())
            base_tokens = rows['sadi'][image]['response_token_ids']
            records.append(dict(image_id=image, step=step, token=target, canonical=canonical,
                p_real=gate['p_real'], triggered=gate['positive'], annotation_supported=canonical in annotations[str(image)],
                same_pre_action_history=True, original_sadi_context=tokenizer.decode(base_tokens[max(0, step-10):step+10],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False),
                methods={name: next((dict(accepted=e['accepted'], actual_log_probability_change=e['actual_log_probability_change'],
                    immediate_token_changed=e['immediate_token_changed']) for e in stats[name]['full_joint_events'] if e['step'] == step), None)
                    for name in names}))
    counts = Counter((r['triggered'], r['annotation_supported']) for r in records)
    table = {str(trigger): {str(support): counts[trigger, support] for support in (False, True)} for trigger in (False, True)}
    unique = Counter((trigger, support) for trigger, support, image, canonical in
                     {(r['triggered'], r['annotation_supported'], r['image_id'], r['canonical']) for r in records})
    result = dict(passed=True, created=time.time(), source_sha256=sha(Path(__file__)), input_sha256=inputs,
        original_gate_threshold=plan['real_threshold'], compared_steps=len(records), excluded_not_exact=len(exclusions),
        event_crosstab_triggered_then_annotation_supported=table,
        image_class_crosstab_triggered_then_annotation_supported={str(t):{str(s):unique[t,s] for s in (False,True)} for t in (False,True)},
        records=records, exclusions=exclusions,
        scope='Post-hoc original development100 only; exclude all steps after any prior accepted hidden/KV action in any method.',
        labels='Annotation-supported means canonical class occurs in CHAIR COCO object annotations; not a physical-truth or contextual-sense judgment.',
        limitation='An annotation-unsupported class can be visible but unannotated, a colour/venue/part word, a negation, or a pictured object. Repeated image-class decisions are not independent. No detection accuracy, calibration, or mitigation success claim.',
        validation_untouched=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(compared_steps=len(records), excluded_not_exact=len(exclusions), counts=table,
        triggered_cases=[{k:r[k] for k in ('image_id','step','canonical','p_real','annotation_supported','original_sadi_context')}
                         for r in records if r['triggered']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)
