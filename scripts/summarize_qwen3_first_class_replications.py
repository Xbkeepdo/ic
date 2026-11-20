"""Report both frozen500 cohorts separately and their descriptive1000 pool."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_qwen3_fresh_caption500 import boot, calculate, lines, read, sha

KEYS = ('chair_s', 'chair_i', 'object_recall', 'object_f1')
CANDIDATE = 'sadi_probe_first_class_recycle_19'


def summarize(first, replication, output):
    roots = (first, replication)
    protocols = [read(root / 'protocol.json') for root in roots]
    names = protocols[0]['variants']
    assert names == ['sadi', CANDIDATE, 'svar'] == protocols[1]['variants']
    for key in ('model', 'prompt', 'recipe', 'max_new_tokens', 'max_batch'):
        assert protocols[0][key] == protocols[1][key], key
    for source, expected in protocols[0]['source_sha256'].items():
        assert protocols[1]['source_sha256'][source] == expected, source
    cohorts = [read(root / 'cohort.json')['coco500'] for root in roots]
    assert all(len(ids) == len(set(ids)) == 500 for ids in cohorts)
    assert set(cohorts[0]).isdisjoint(cohorts[1])
    metadata, summaries, scores, hashes = [], [], {}, {}
    for index, root in enumerate(roots):
        assert read(root / 'run_status.json')['status'] == 'complete'
        report = read(root / 'independent_summary.json')
        runtime = read(root / 'first_class_runtime_audit.json')
        mechanism = read(root / 'mechanism_audit.json')[CANDIDATE]
        assert report['status'] == 'passed' and runtime['passed'] and not runtime['partial']
        assert runtime['paired_images'] == mechanism['images'] == 500
        assert mechanism['maximum_probe_hidden_difference'] == 0
        summaries.append(report)
        metadata.append(dict(root=str(root), role='exploratory follow-up' if index == 0 else 'frozen replication',
                             sample_sha256=report['sample_sha256'], runtime=runtime,
                             generation=report['generation']))
        files = [root / name for name in ('protocol.json', 'cohort.json', 'selection.json', 'run_status.json',
                  'independent_summary.json', 'first_class_runtime_audit.json', 'mechanism_audit.json',
                  'object_mention_audit.json', 'denominator_audit.json')]
        for mode in ('sentence', 'legacy'):
            folder = root / f'evaluations/{mode}_v1/coco500'
            per, = folder.glob('chair_per_image*.jsonl')
            metric, = folder.glob('metrics.*.json')
            rows = lines(per)
            assert [row['image_id'] for row in rows] == cohorts[index]
            files.extend((per, metric))
            values = {name: [row['methods'][name] for row in rows] for name in names}
            for name in names:
                computed = calculate(values[name])
                reference = report['evaluators'][mode]['metrics'][name]
                for key, value in computed.items():
                    np.testing.assert_allclose(value, reference[key], rtol=0, atol=1e-12)
            scores[index, mode] = values
        for shard in (0, 1):
            files.extend(root / f'coco500/{name}.shard{shard}.jsonl' for name in names)
        hashes.update({str(path): sha(path) for path in files})
    result = dict(passed=True, summary_script_sha256=sha(Path(__file__)),
        primary='Second500 frozen replication; first500 remains separately reported.',
        pooled_scope='Descriptive only: the first cohort was already inspected when replication was chosen. Pooled intervals are not a separate confirmatory test.',
        bootstrap='5000 paired draws, seed43, independently resample500 images inside each fixed cohort.',
        cohorts=metadata, input_sha256=hashes, evaluators={})
    rng = np.random.default_rng(43)
    draws = np.concatenate((rng.integers(0, 500, (5000, 500)), rng.integers(500, 1000, (5000, 500))), axis=1)
    for mode in ('sentence', 'legacy'):
        pooled = {name: scores[0, mode][name] + scores[1, mode][name] for name in names}
        metrics = {name: calculate(rows) for name, rows in pooled.items()}
        samples = {name: boot(rows, draws) for name, rows in pooled.items()}
        comparisons = {}
        for left, right in combinations(names, 2):
            comparisons[f'{right} vs {left}'] = dict(
                delta={key: metrics[right][key] - metrics[left][key] for key in KEYS},
                delta_ci95={key: np.quantile(samples[right][key] - samples[left][key], [.025, .975]).tolist() for key in KEYS},
                fixed_hallucinated_images=sum(a['chair_s'] > b['chair_s'] for a, b in zip(pooled[left], pooled[right], strict=True)),
                introduced_hallucinated_images=sum(a['chair_s'] < b['chair_s'] for a, b in zip(pooled[left], pooled[right], strict=True)))
        ranking = sorted(range(1000), key=lambda i: pooled[CANDIDATE][i]['hallucinated_mentions'] - pooled['sadi'][i]['hallucinated_mentions'])
        ids = cohorts[0] + cohorts[1]
        result['evaluators'][mode] = dict(
            by_cohort=[report['evaluators'][mode] for report in summaries],
            pooled_metrics=metrics, pooled_comparisons=comparisons,
            largest_reductions=[dict(image_id=ids[i], cohort=i // 500,
                                    base=pooled['sadi'][i]['hallucinated_mentions'],
                                    candidate=pooled[CANDIDATE][i]['hallucinated_mentions']) for i in ranking[:3]],
            pooled_exclude_top3={name: calculate([rows[i] for i in ranking[3:]]) for name, rows in pooled.items()})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    text = ['# Qwen3：原首次类别候选的两批500图', '',
            '第二批固定复验单独作为主要验证；第一批为探索性跟进。1000图合并只作描述性汇总，不能替代第二批结果。', '',
            '同一冻结配方，cap512/maxB16、greedy BF16 eager、SADI5–18及19层回收；两批图片不重叠。sentence为主。', '',
            '| 图片范围 | 方法 | CHAIRs ↓ | CHAIRi ↓ | 召回 ↑ | F1 ↑ | 幻觉／全部提及 |',
            '|---|---|---:|---:|---:|---:|---:|']
    labels = {'sadi': 'SADI', CANDIDATE: 'SADI＋首次类别19层回收', 'svar': 'SVAR'}
    sentence = result['evaluators']['sentence']
    for label, metrics in [('第一批500', summaries[0]['evaluators']['sentence']['metrics']),
                           ('第二批500', summaries[1]['evaluators']['sentence']['metrics']),
                           ('合并1000', sentence['pooled_metrics'])]:
        for name in names:
            row = metrics[name]
            text.append(f"| {label} | {labels[name]} | {row['chair_s']*100:.2f} | {row['chair_i']*100:.2f} | {row['object_recall']*100:.2f} | {row['object_f1']*100:.2f} | {row['hallucinated_mentions']}／{row['object_mentions']} |")
    (output / 'summary.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('TWO_COHORT_SUMMARY', output)
    print(json.dumps(sentence['pooled_metrics']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('replication', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summarize(args.first, args.replication, args.output)
