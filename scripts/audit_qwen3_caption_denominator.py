"""Recover real-object class counts and audit CHAIR denominator changes."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def audit(root):
    protocol = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    report = {}
    for mode in ('sentence', 'legacy'):
        folder = root / f'evaluations/{mode}_v1/{protocol["cohort"]}'
        metrics_path, = folder.glob('metrics.*.json')
        metrics = read(metrics_path)
        ground_truth = read(Path(metrics['chair_cache']))['objects']
        per_image, = folder.glob('chair_per_image*.jsonl')
        rows = [json.loads(line) for line in per_image.read_bytes().splitlines()]
        real_unique, counts = {}, {}
        for name in protocol['variants']:
            unique = [row['methods'][name]['object_recall'] * len(ground_truth[str(row['image_id'])]) for row in rows]
            assert all(abs(value - round(value)) < 1e-8 for value in unique)
            real_unique[name] = list(map(round, unique))
            assert all(value <= row['methods'][name]['real_mentions'] for value, row in zip(real_unique[name], rows, strict=True))
            real = sum(row['methods'][name]['real_mentions'] for row in rows)
            counts[name] = dict(real_mentions=real, unique_real_image_classes=sum(real_unique[name]),
                                repeated_real_mentions=real - sum(real_unique[name]))
        pairs = {}
        for name in protocol['candidates']:
            changes = [row['methods'][name]['real_mentions'] - real_unique[name][i]
                       - row['methods']['sadi']['real_mentions'] + real_unique['sadi'][i]
                       for i, row in enumerate(rows)]
            order = sorted(range(len(rows)), key=lambda i: -changes[i])
            dropped = set(order[:3])
            trimmed = {}
            for method in ('sadi', name):
                scores = [row['methods'][method] for i, row in enumerate(rows) if i not in dropped]
                trimmed[method] = dict(chair_i=sum(s['hallucinated_mentions'] for s in scores)
                                              / max(1, sum(s['object_mentions'] for s in scores)),
                                       chair_s=sum(s['chair_s'] for s in scores) / len(scores),
                                       false_mentions=sum(s['hallucinated_mentions'] for s in scores))
            pairs[name] = dict(largest_real_repetition_increases=[dict(image_id=rows[i]['image_id'], increase=changes[i]) for i in order[:3]],
                                drop_top3_real_repetition_increases=trimmed)
        report[mode] = dict(counts=counts, pairs=pairs)
    (root / 'denominator_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['sentence'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
