"""Summarize the semantic-final study after the independent audits pass."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from scripts.audit_qwen3_visual_feedback_mentions import compare, summarize
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, write


def analyze(root):
    assert read(root / 'post_audit_status.json')['status'] == 'passed'
    plan = read(root / 'protocol.json')
    raw, groups = {}, {}
    for name in plan['variants']:
        groups[name] = [g for shard in (0, 1) for g in source_groups(root / f'coco100/{name}.shard{shard}.jsonl')]
        raw[name] = {r['image_id']: r for group in groups[name] for r in group}
    ids = read(root / 'cohort.json')['coco100']
    draws = np.random.default_rng(43).integers(0, len(ids), size=(5000, len(ids)))
    report, cases, activity = {}, {}, {}
    for mode in ('sentence', 'legacy'):
        path, = (root / f'evaluations/{mode}_v1/coco100').glob('chair_per_image*.jsonl')
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert [row['image_id'] for row in rows] == ids
        scores = {name: [r['methods'][name] for r in rows] for name in plan['variants']}
        report[mode] = dict(metrics={name: summarize(v) for name, v in scores.items()},
            pairs={name: {ref: compare(scores[ref], scores[name], ids, draws)
                          for ref in plan['variants'] if ref != name} for name in plan['candidates']})
        if mode == 'sentence':
            by_image = {r['image_id']: r['methods'] for r in rows}
            for name in plan['candidates']:
                changed = {}
                for image in ids:
                    if raw[name][image]['generated_text'] == raw['sadi'][image]['generated_text']:
                        continue
                    a, b = by_image[image]['sadi'], by_image[image][name]
                    changed[str(image)] = dict(
                        methods={method: dict(generated_text=raw[method][image]['generated_text'],
                            scores=by_image[image][method],
                            events=raw[method][image]['sar_ems_statistics'].get('semantic_events', []))
                            for method in plan['variants']},
                        false_mentions_changed=Counter(map(tuple, a['hallucinated_objects'])) != Counter(map(tuple, b['hallucinated_objects'])))
                cases[name] = changed
    for name in plan['candidates']:
        rows = list(raw[name].values())
        events = [e for r in rows for e in r['sar_ems_statistics']['semantic_events']]
        decisions = [e for r in rows for e in r['sar_ems_statistics']['class_decisions']]
        activity[name] = dict(events=len(events), active_images=sum(bool(r['sar_ems_statistics']['semantic_events']) for r in rows),
            probability_decreases=sum(e['actual_log_probability_change'] < 0 for e in events),
            mean_log_probability_change=sum(e['actual_log_probability_change'] for e in events)/max(1, len(events)),
            class_decisions=len(decisions), class_blocked=sum(d['blocked'] for d in decisions),
            cache_builds=sum(g[0]['sar_ems_statistics']['static_cache_builds'] for g in groups[name]),
            cache_reuses=sum(g[0]['sar_ems_statistics']['static_cache_reuses'] for g in groups[name]),
            semantic_head_calls=sum(g[0]['sar_ems_statistics']['semantic_projection_calls'] for g in groups[name]),
            replay_steps=sum(g[0]['sar_ems_statistics']['replay_steps'] for g in groups[name]),
            changed_text_images=len(cases[name]), false_count_changed_images=sum(v['false_mentions_changed'] for v in cases[name].values()))
    write(root / 'semantic_mentions.json', report)
    write(root / 'semantic_changed_text.json', cases)
    write(root / 'semantic_activity.json', activity)
    write(root / 'semantic_analysis_provenance.json', dict(source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(root / 'protocol.json'), numerical_screen=read(root / 'screening.json'),
        image_review='pending', advance_to_500=False))
    print(json.dumps(dict(activity=activity, numerical_screen=read(root / 'screening.json')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
