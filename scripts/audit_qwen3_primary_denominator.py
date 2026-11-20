"""Audit true-object repetitions relative to the explicitly named primary base."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    base = plan['primary_baseline']
    assert base in plan['variants']
    report = dict(passed=True, primary_baseline=base, eval={})
    for mode in ('sentence', 'legacy'):
        folder = root / f'evaluations/{mode}_v1' / plan['cohort']
        metric_path, = folder.glob('metrics.*.json')
        gt = read(Path(read(metric_path)['chair_cache']))['objects']
        per_path, = folder.glob('chair_per_image*.jsonl')
        rows = [json.loads(line) for line in per_path.read_bytes().splitlines()]
        assert [row['image_id'] for row in rows] == read(root / 'cohort.json')[plan['cohort']]
        unique, counts = {}, {}
        for name in plan['variants']:
            values = [row['methods'][name]['object_recall'] * len(gt[str(row['image_id'])]) for row in rows]
            assert all(abs(value - round(value)) < 1e-8 for value in values)
            unique[name] = list(map(round, values))
            assert all(0 <= value <= row['methods'][name]['real_mentions'] for value, row in zip(unique[name], rows, strict=True))
            total = sum(row['methods'][name]['real_mentions'] for row in rows)
            counts[name] = dict(real_mentions=total, unique_real_image_classes=sum(unique[name]),
                                repeated_real_mentions=total - sum(unique[name]))
        pairs = {}
        for name in (n for n in plan['variants'] if n != base):
            changes = [row['methods'][name]['real_mentions'] - unique[name][i]
                       - row['methods'][base]['real_mentions'] + unique[base][i] for i, row in enumerate(rows)]
            order = sorted(range(len(rows)), key=lambda i: -changes[i])
            dropped, trimmed = set(order[:3]), {}
            for method in (base, name):
                scores = [row['methods'][method] for i, row in enumerate(rows) if i not in dropped]
                trimmed[method] = dict(chair_i=sum(s['hallucinated_mentions'] for s in scores) / max(1, sum(s['object_mentions'] for s in scores)),
                                      chair_s=sum(s['chair_s'] for s in scores) / len(scores),
                                      false_mentions=sum(s['hallucinated_mentions'] for s in scores))
            pairs[name] = dict(largest_real_repetition_increases=[dict(image_id=rows[i]['image_id'], increase=changes[i]) for i in order[:3]],
                               drop_top3_real_repetition_increases=trimmed)
        report['eval'][mode] = dict(counts=counts, pairs=pairs)
    (root / 'primary_denominator_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
