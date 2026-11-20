"""Independently verify paired fresh-caption metrics and inspect repeated mentions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


KEYS = ('chair_s', 'chair_i', 'object_recall', 'object_f1')


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(rows):
    n = len(rows)
    assert n
    hallucinated = sum(r['hallucinated_mentions'] for r in rows)
    mentions = sum(r['object_mentions'] for r in rows)
    false_classes = sum(len({x[1] for x in r['hallucinated_objects']}) for r in rows)
    return {
        'images': n, 'chair_s': statistics.mean(r['chair_s'] for r in rows),
        'chair_i': hallucinated / max(1, mentions), 'object_mentions': mentions,
        'hallucinated_mentions': hallucinated, 'hallucinated_image_classes': false_classes,
        'repeated_false_mentions': hallucinated - false_classes,
        'object_recall': statistics.mean(r['object_recall'] for r in rows),
        'object_f1': statistics.mean(r['object_f1'] for r in rows),
    }


def boot(rows, draws):
    vectors = {k: np.array([r[k] for r in rows], dtype=np.float64)[draws] for k in (
        'chair_s', 'hallucinated_mentions', 'object_mentions', 'object_recall', 'object_f1')}
    return {'chair_i': vectors['hallucinated_mentions'].sum(1) / np.maximum(1, vectors['object_mentions'].sum(1)),
            **{k: vectors[k].mean(1) for k in ('chair_s', 'object_recall', 'object_f1')}}


def main(root):
    protocol = read(root / 'protocol.json')
    status = read(root / 'run_status.json')
    assert status['status'] == 'complete'
    ids = read(root / 'cohort.json')['coco500']
    selection = read(root / 'selection.json')
    assert sorted(set(ids)) == ids and len(ids) == protocol['image_count'] == 500
    assert set(ids).isdisjoint(selection['excluded_ids'])
    digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    assert digest == protocol['sample_sha256'] == selection['sample_sha256']
    names = protocol['variants']
    for relative, expected in protocol['source_sha256'].items():
        assert sha(root.parents[1] / relative) == expected, relative
    assert sha(Path(protocol['prior_protocol'])) == protocol['prior_protocol_sha256']
    raw = {name: {} for name in names}
    plans = {name: {} for name in names}
    hashes = {}
    for shard in range(2):
        for name in names:
            path = root / f'coco500/{name}.shard{shard}.jsonl'
            hashes[path.name] = sha(path)
            rows = lines(path)
            assert len(rows) == 250 and set(r['image_id'] for r in rows) == set(ids[shard::2])
            offset, groups = 0, []
            while offset < len(rows):
                first = rows[offset]
                size = first['batch_size']
                assert 1 <= size <= protocol['max_batch']
                group = rows[offset:offset + size]
                assert len(group) == size
                for row in group:
                    assert row['variant'] == name and row['image_id'] not in raw[name]
                    assert 1 <= len(row['response_token_ids']) <= protocol['max_new_tokens']
                    assert row['batch_size'] == size and row['visual_token_count'] == first['visual_token_count']
                    assert row['batch_wall_seconds'] == first['batch_wall_seconds']
                    assert np.isclose(row['seconds'] * size, first['batch_wall_seconds'], atol=1e-8)
                    assert row['decoder'] == 'native_qwen3_generate_same_span_batch'
                    raw[name][row['image_id']] = row
                groups.append([r['image_id'] for r in group])
                offset += size
            plans[name][shard] = groups
    assert all(p == plans[names[0]] for p in plans.values())
    assert hashes == read(root / 'batch_audit.json')['caption_sha256']

    draws = np.random.default_rng(43).integers(0, 500, size=(5000, 500))
    candidate = protocol.get('candidate_method') or next(name for name in names if '_alternative_' in name)
    baseline = 'sadi'
    output = {'status': 'passed', 'sample_sha256': digest, 'evaluators': {}, 'generation': {},
              'batch_sizes': dict(Counter(len(group) for groups in plans[names[0]].values() for group in groups)),
              'elapsed_seconds': status['finished'] - status['started'],
              'finished_utc': datetime.fromtimestamp(status['finished'], timezone.utc).isoformat()}

    for mode in ('sentence', 'legacy'):
        path, = (root / f'evaluations/{mode}_v1/coco500').glob('metrics.first500.*.json')
        result = read(path)
        assert hashes == {Path(p).name: h for p, h in result['evaluation_inputs']['caption_sha256'].items()}
        per_image = path.with_name('chair_per_image' + path.name[len('metrics'):-5] + '.jsonl')
        scores = lines(per_image)
        assert [r['image_id'] for r in scores] == ids
        a = {name: [r['methods'][name] for r in scores] for name in names}
        summary = {name: calculate(rows) for name, rows in a.items()}
        samples = {name: boot(rows, draws) for name, rows in a.items()}
        pairs = {}
        for name in names:
            for key in KEYS:
                np.testing.assert_allclose(summary[name][key], result['methods'][name][key], atol=1e-12, rtol=0)
        for left, right in combinations(names, 2):
            pair = result['methods'][right]['paired_against'][left]
            for key in KEYS:
                np.testing.assert_allclose(summary[right][key] - summary[left][key], pair['delta'][key], atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.quantile(samples[right][key] - samples[left][key], [.025, .975]),
                                           pair['delta_ci95'][key], atol=1e-12, rtol=0)
            assert pair['changed_token_sequences'] == sum(raw[right][i]['response_token_ids'] != raw[left][i]['response_token_ids'] for i in ids)
            assert pair['fixed_hallucinated_images'] == sum(y['chair_s'] < x['chair_s'] for x, y in zip(a[left], a[right], strict=True))
            assert pair['introduced_hallucinated_images'] == sum(y['chair_s'] > x['chair_s'] for x, y in zip(a[left], a[right], strict=True))
            pairs[f'{right} vs {left}'] = pair
        rank = sorted(range(len(ids)), key=lambda i: a[candidate][i]['hallucinated_mentions'] - a[baseline][i]['hallucinated_mentions'])
        reduced = [{'image_id': ids[i], 'base': a[baseline][i]['hallucinated_mentions'],
                    'candidate': a[candidate][i]['hallucinated_mentions']} for i in rank[:3]]
        without_top3 = {name: calculate([entries[i] for i in rank[3:]]) for name, entries in a.items()}
        uncapped = [i for i, image in enumerate(ids) if all(len(raw[n][image]['response_token_ids']) < protocol['max_new_tokens']
                                                            for n in (candidate, baseline))]
        pair_uncapped = {name: calculate([entries[i] for i in uncapped]) for name, entries in a.items()}
        halves = [{name: calculate(entries[parity::2]) for name, entries in a.items()} for parity in (0, 1)]
        output['evaluators'][mode] = {'metrics': summary, 'comparisons': pairs, 'largest_improvements': reduced,
                                      'exclude_top3': without_top3, 'candidate_base_uncapped': pair_uncapped,
                                      'by_generation_shard': halves}

    for name, records in raw.items():
        rows = list(records.values())
        stats = [row['sar_ems_statistics'] or {} for row in rows]
        assert all(not s.get('diagnostic_force_guidance') and not s.get('diagnostic_zero_replay')
                   and not s.get('diagnostic_reject_all') for s in stats)
        for stat in stats:
            expected_replay = stat.get('guided_attention_calls', 0) if stat.get('head_specific') else 0
            assert stat.get('local_recompute_events', 0) == expected_replay
            if name == candidate and protocol.get('candidate_method'):
                assert stat['head_specific'] and stat['gradient_scope'] == 'final'
        output['generation'][name] = {'mean_tokens': statistics.mean(len(r['response_token_ids']) for r in rows),
                                      'cap_count': sum(len(r['response_token_ids']) == protocol['max_new_tokens'] for r in rows),
                                      'gpu_seconds': sum(r['seconds'] for r in rows),
                                      'attention_events': sum(s.get('guided_attention_calls', 0) for s in stats),
                                      'target_matches': sum(s.get('target_match_events', 0) for s in stats)}
    path = root / 'independent_summary.json'
    path.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('AUDIT PASS', len(names) * len(ids), 'captions', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
