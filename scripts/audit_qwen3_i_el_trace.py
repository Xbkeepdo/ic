"""Independently check labels, semantic caches and frozen-probe predictions."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    assert read(root / 'status.json')['status'] == 'complete'
    plan = read(root / 'protocol.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(Path(path)) == expected, path
    source = Path(plan['source'])
    per_image, = (source / 'evaluations/sentence_v1/coco100').glob('chair_per_image*.jsonl')
    expected = {row['image_id']: row['methods']['sadi']
                for row in map(json.loads, per_image.read_bytes().splitlines())}
    records, checks, raw = [], [], {}
    caches = {}
    for shard in range(plan['shards']):
        records.extend(map(json.loads, (root / f'features.shard{shard}.jsonl').read_bytes().splitlines()))
        checks.extend(dict(row, shard=shard) for row in read(root / f'batch_checks.shard{shard}.json'))
        for row in map(json.loads, (source / f'coco100/sadi.shard{shard}.jsonl').read_bytes().splitlines()):
            assert row['image_id'] not in raw
            raw[row['image_id']] = row
        for batch, group in enumerate(plan['groups'][shard]):
            file = root / f'semantics/shard{shard}.batch{batch}.npz'
            if not file.exists():
                continue
            with np.load(file, allow_pickle=False) as cache:
                d = {key: cache[key] for key in cache.files}
            assert d['image_ids'].tolist() == group['ids']
            assert d['span'][1] - d['span'][0] == group['visual_token_count']
            assert d['token_ids'].tolist() == sorted(set(d['token_ids'].tolist()))
            assert d['E_l'].shape == (36, len(group['ids']), len(d['token_ids']))
            assert np.isfinite(d['E_l']).all() and (d['E_l'] >= 0).all()
            caches[shard, batch] = d
    assert set(raw) == set(read(root / 'cohort.json')['trace']) == set(expected)
    by_image = defaultdict(list)
    keys = set()
    semantic_columns = 0
    for row in records:
        image, step, token = row['image_id'], row['step'], row['target_token_id']
        assert (image, step) not in keys
        keys.add((image, step))
        assert raw[image]['response_token_ids'][step] == token
        feature = np.asarray(row['features'], dtype=np.float32)
        assert feature.shape == (72,) and np.isfinite(feature).all() and (feature >= 0).all()
        cache = caches[row['shard'], row['batch']]
        slot = cache['image_ids'].tolist().index(image)
        column = cache['token_ids'].tolist().index(token)
        np.testing.assert_array_equal(feature[36:], cache['E_l'][:, slot, column])
        semantic_columns += 36
        by_image[image].append(row)
    total_hallucinations = total_mentions = 0
    for image, reference in expected.items():
        mentions = [m for row in by_image[image] for m in row['mentions']]
        assert len(mentions) == reference['object_mentions']
        assert sum(m['hallucinated'] for m in mentions) == reference['hallucinated_mentions']
        seen = set()
        for row in sorted(by_image[image], key=lambda r: r['step']):
            for mention in row['mentions']:
                assert mention['repeat'] == (mention['canonical'] in seen)
                seen.add(mention['canonical'])
        total_mentions += len(mentions)
        total_hallucinations += sum(m['hallucinated'] for m in mentions)
    with np.load(root / 'predictions.npz', allow_pickle=False) as cache:
        saved = {key: cache[key] for key in cache.files}
    features = np.asarray([r['features'] for r in records], dtype=np.float32)
    np.testing.assert_array_equal(features, saved['features'])
    np.testing.assert_array_equal([r['image_id'] for r in records], saved['image_id'])
    np.testing.assert_array_equal([r['step'] for r in records], saved['step'])
    np.testing.assert_array_equal([r['target_token_id'] for r in records], saved['token_id'])
    labels = np.array([any(m['hallucinated'] for m in r['mentions']) for r in records])
    np.testing.assert_array_equal(labels, saved['hallucinated'])
    probe = Path(plan['probe'])
    state = {k: v.numpy() for k, v in torch.load(probe / 'model.pt', map_location='cpu', weights_only=True).items()}
    with np.load(probe / 'input_scaler.npz', allow_pickle=False) as scaler:
        x = ((features - scaler['mean']) / scaler['scale']).astype(np.float32)
    hidden = np.maximum(0, x @ state['net.0.weight'].T + state['net.0.bias'])
    logits = (hidden @ state['net.3.weight'].T + state['net.3.bias'])[:, 0]
    probability = 1 / (1 + np.exp(-logits.astype(np.float64)))
    difference = float(np.max(np.abs(probability - saved['p_real'])))
    assert difference < 2e-6
    analysis = read(root / 'analysis.json')
    assert abs(roc_auc_score(labels, 1 - saved['p_real']) - analysis['probe']['all_objects']['auroc']) < 1e-12
    assert abs(average_precision_score(labels, 1 - saved['p_real']) - analysis['probe']['all_objects']['hall_aupr']) < 1e-12
    for checked in checks:
        group = plan['groups'][checked['shard']][checked['batch']]
        assert checked['ids'] == group['ids']
        selected = [r for r in records if r['shard'] == checked['shard'] and r['batch'] == checked['batch']]
        assert len(selected) == checked['positions']
        stop = max(r['step'] for r in selected) + 1
        assert sum(min(len(raw[i]['response_token_ids']), stop) for i in group['ids']) == checked['checked_tokens']
    assert sum(r['native_argmax_mismatches'] for r in checks) == analysis['native_argmax_mismatches'] == 0
    assert all(r['native_argmax_matches'] for r in records)
    report = dict(passed=True, protocol_sha256=sha(root / 'protocol.json'), images=len(raw),
        positions=len(records), object_mentions=total_mentions, hallucinated_mentions=total_hallucinations,
        semantic_columns_checked=semantic_columns, semantic_caches=len(caches),
        checked_native_tokens=sum(r['checked_tokens'] for r in checks), numpy_probe_max_difference=difference,
        feature_file_sha256={str(p): sha(p) for p in root.glob('features.shard*.jsonl')},
        semantic_file_sha256={str(p): sha(p) for p in (root / 'semantics').glob('*.npz')})
    if 'source_trace' in plan:
        with np.load(Path(plan['source_trace']) / 'predictions.npz', allow_pickle=False) as cache:
            old = {key: cache[key] for key in cache.files}
        for name in ('image_id', 'step', 'token_id', 'hallucinated'):
            np.testing.assert_array_equal(saved[name], old[name])
        np.testing.assert_array_equal(saved['features'][:, 36:], old['features'][:, 36:])
        np.testing.assert_array_equal(saved['visual_support31'], old['visual_support31'])
        columns = list(range(5)) + list(range(19, 36))
        delta = np.abs(saved['features'][:, columns] - old['features'][:, columns])
        report['nonedited_I_max_difference'] = float(delta.max())
        report['E_l_cache_reused_exactly'] = True
    (root / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
