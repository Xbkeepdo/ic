"""Summarize completed selected-layer outputs without running the model."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_qwen3_visual_feedback_mentions import compare, summarize
from scripts.run_qwen3_signed_feedback import read, source_groups, write


def analyze(root):
    assert read(root / 'post_audit_status.json')['status'] == 'passed'
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    name, = plan['candidates']
    base, reference = plan['primary_baseline'], plan['primary_mechanism_comparator']
    ids = read(root / 'cohort.json')[plan['cohort']]
    assert len(ids) == plan['images'] == 100
    draws = np.random.default_rng(43).integers(0, len(ids), size=(5000, len(ids)))
    mentions, all_scores = {}, {}
    for mode in ('sentence', 'legacy'):
        path, = (root / f'evaluations/{mode}_v1' / plan['cohort']).glob('chair_per_image*.jsonl')
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert [row['image_id'] for row in rows] == ids
        scores = {method: [row['methods'][method] for row in rows] for method in plan['variants']}
        all_scores[mode] = {row['image_id']: row['methods'] for row in rows}
        mentions[mode] = dict(metrics={m: summarize(v) for m, v in scores.items()},
            pairs={method: compare(scores[method], scores[name], ids, draws)
                for method in (base, reference, 'svar')})
    write(root / 'ffn_layer_selection_mention_audit.json', mentions)
    raw, activity = {}, {}
    for method in plan['variants']:
        groups = [group for shard in (0, 1)
            for group in source_groups(root / plan['cohort'] / f'{method}.shard{shard}.jsonl')]
        raw[method] = {row['image_id']: row for group in groups for row in group}
        assert set(raw[method]) == set(ids)
        events = [event for row in raw[method].values()
            for event in row['sar_ems_statistics'].get('recycle_events', [])]
        accepted = [e for e in events if e['accepted']]
        activity[method] = dict(proposals=len(events), accepted=len(accepted),
            active_images=sum(bool(row['sar_ems_statistics'].get('accepted_guidance_calls')) for row in raw[method].values()),
            moved_attention_mass_accepted=sum(e['moved_attention_mass'] for e in accepted),
            moved_attention_mass_proposed=sum(e['moved_attention_mass'] for e in events),
            mean_drift=sum(e['drift'] for e in events)/max(1, len(events)))
        if method == name:
            decisions = [d for row in raw[method].values() for d in row['sar_ems_statistics']['layer_decisions']]
            selected = Counter(e['selected_layer'] for e in events)
            kept = Counter(e['selected_layer'] for e in accepted)
            activity[method].update(selected_layers={str(i): selected[i] for i in (19,23,27,31)},
                accepted_layers={str(i): kept[i] for i in (19,23,27,31)}, layer_decisions=len(decisions),
                no_positive_layer=sum(d['selected_layer'] is None for d in decisions),
                gradient_calls=sum(group[0]['sar_ems_statistics']['gradient_probe_calls'] for group in groups))
    write(root / 'ffn_layer_selection_activity.json', activity)
    summary = read(root / 'independent_summary.json')['eval']['sentence']
    m, b = summary[name], summary[base]
    pair = mentions['sentence']['pairs'][base]
    numeric = dict(chair_s_lower=m['chair_s'] < b['chair_s'], chair_i_lower=m['chair_i'] < b['chair_i'],
        absolute_false_mentions_lower=m['audit_false_mentions'] < b['audit_false_mentions'],
        f1_loss_within_1_5pp=m['object_f1'] >= b['object_f1']-.015,
        recall_loss_within_2pp=m['object_recall'] >= b['object_recall']-.02,
        leave_largest_ci_no_worse=pair['drop_largest_reduction']['candidate']['chair_i'] <= pair['drop_largest_reduction']['base']['chair_i'])
    write(root / 'ffn_layer_selection_numerical_screen.json', dict(primary_baseline=base,
        criteria=numeric, passed=all(numeric.values()), image_review='pending', advance_to_500=False,
        interpretation='Development screen only; an unchanged or denominator-only result is insufficient.'))
    cases = {}
    for image in ids:
        scores = all_scores['sentence'][image]
        candidate = Counter(map(tuple, scores[name]['hallucinated_objects']))
        if all(candidate == Counter(map(tuple, scores[method]['hallucinated_objects'])) for method in (base, reference)):
            continue
        cases[str(image)] = dict(methods={method: dict(
            generated_text=raw[method][image]['generated_text'], scores=scores[method],
            events=raw[method][image]['sar_ems_statistics'].get('recycle_events', [])) for method in plan['variants']})
    changed = {}
    for image in ids:
        if all(raw[name][image]['response_token_ids'] == raw[method][image]['response_token_ids'] for method in (base,reference)):
            continue
        changed[str(image)] = dict(methods={method:dict(generated_text=raw[method][image]['generated_text'],
            events=raw[method][image]['sar_ems_statistics'].get('recycle_events',[])) for method in plan['variants']},
            changed_against_base=raw[name][image]['response_token_ids'] != raw[base][image]['response_token_ids'],
            changed_against_old_guidance=raw[name][image]['response_token_ids'] != raw[reference][image]['response_token_ids'])
    write(root / 'changed_caption_review_inputs.json', dict(scope='All sequence differences against SADI or old guidance',cases=changed))
    write(root / 'visual_review_inventory.json', dict(
        scope='All images whose candidate false-mention multiset differs from SADI or the old full guidance; inspect language context before visual identities.',
        cases=cases, analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    report = dict(screen=numeric, activity=activity, case_ids=list(cases), changed_caption_ids=list(changed), metrics={})
    for mode, methods in read(root / 'independent_summary.json')['eval'].items():
        report['metrics'][mode] = {method: {k: methods[method][k] for k in
            ('chair_s', 'chair_i', 'object_recall', 'object_f1', 'audit_false_mentions', 'audit_mentions')} for method in plan['variants']}
        report['metrics'][mode]['candidate_pairs'] = {method: methods[name]['paired_against'][method] for method in (base, reference, 'svar')}
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
