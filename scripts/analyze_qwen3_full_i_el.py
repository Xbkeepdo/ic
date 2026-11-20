"""Audit and score frozen original I/E_l probes on saved SADI-prefix features."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_online_i_el import FullIElProbe


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(labels, probability, tokens, thresholds):
    report = dict(targets=len(labels), hallucinations=int(labels.sum()), real=int((~labels).sum()))
    if len(set(labels.tolist())) < 2:
        return report
    report.update(auroc=float(roc_auc_score(labels, probability)),
                  hall_aupr=float(average_precision_score(labels, probability)))
    weighted, pairs = 0., 0
    for token in np.unique(tokens):
        mask = tokens == token
        group = labels[mask]
        count = int(group.sum()) * int((~group).sum())
        if count:
            weighted += count * roc_auc_score(group, probability[mask])
            pairs += count
    report.update(within_token_auroc=weighted / pairs if pairs else None, within_token_pairs=pairs,
                  gates={})
    for name, threshold in thresholds.items():
        predicted = probability >= threshold
        true = int((predicted & labels).sum())
        report['gates'][name] = dict(threshold=float(threshold), predictions=int(predicted.sum()),
            hall_recall=float(predicted[labels].mean()), real_false_positive_rate=float(predicted[~labels].mean()),
            hall_precision=true / max(1, int(predicted.sum())))
    return report


def analyze(root):
    torch.set_num_threads(1)
    plan = read(root / 'protocol.json')
    for source, expected in plan['source_sha256'].items():
        assert digest(Path(source)) == expected, source
    preflight = read(root / 'feature_preflight.json')
    assert preflight['passed'] and preflight['exact_baseline_logits_and_tokens']
    probe = FullIElProbe(plan['probe']).eval()
    training_root = Path(plan['probe']).parents[1]
    with np.load(training_root / 'compact_features.npz', allow_pickle=False) as cache:
        old = {k: cache[k] for k in cache.files}
    original_input = np.concatenate((old['I'], old['E_l']), axis=1)
    old_real = probe(torch.from_numpy(original_input)).numpy()
    reference = read(Path(plan['probe']) / 'metrics.json')['auc']
    old_auc = roc_auc_score(old['label'][old['test_mask']], old_real[old['test_mask']])
    assert abs(old_auc - reference) < 1e-6
    train_real = old['train_mask'] & (old['label'] == 1)
    threshold = 1 - float(np.quantile(old_real[train_real], .05))
    thresholds = dict(real_below_0_5=.5, real_below_0_1=.9, old_training_real_5_percent=threshold)
    records, batches = [], []
    for shard in range(plan['shards']):
        records.extend(json.loads(line) for line in (root / f'features.shard{shard}.jsonl').read_bytes().splitlines())
        checked = read(root / f'batch_checks.shard{shard}.json')
        expected = plan['groups'][shard]
        # A batch without CHAIR mentions may legitimately have no replay.
        assert all(row['ids'] == expected[row['batch']]['ids'] for row in checked)
        batches.extend(checked)
    ids = read(root / 'cohort.json')['trace']
    assert set(r['image_id'] for r in records).issubset(ids)
    keys = [(r['image_id'], r['step'], r['target_token_id']) for r in records]
    assert len(keys) == len(set(keys))
    features = np.array([r['features'] for r in records], dtype=np.float32)
    assert features.shape == (len(records), 72) and np.isfinite(features).all() and (features >= 0).all()
    real = probe(torch.from_numpy(features)).numpy()
    labels = np.array([any(m['hallucinated'] for m in r['mentions']) for r in records], dtype=bool)
    mixed = np.array([len({m['hallucinated'] for m in r['mentions']}) > 1 for r in records])
    tokens = np.array([r['target_token_id'] for r in records])
    known = np.array([r['in_train_vocabulary'] for r in records])
    nouns = np.array([r['token_is_noun'] for r in records])
    repeated = np.array([any(m['repeat'] for m in r['mentions']) for r in records])
    v = np.array([r['visual_support31'] for r in records])
    masks = dict(all_objects=~mixed, training_tokens=known & ~mixed,
                 training_nouns=known & nouns & ~mixed,
                 first_mentions=~repeated & ~mixed, repeated_mentions=repeated & ~mixed)
    metrics = {name: statistics(labels[mask], 1 - real[mask], tokens[mask], thresholds)
               for name, mask in masks.items() if mask.any()}
    support = {name: statistics(labels[mask], -v[mask], tokens[mask], dict(V_below_0_35=-.35))
               for name, mask in masks.items() if mask.any()}
    checkpoint = Path(plan['probe']) / 'model.pt'
    report = dict(passed=True, purpose=plan['purpose'], source=str(plan['source']), images=len(ids),
        images_with_objects=len({r['image_id'] for r in records}), object_positions=len(records),
        object_mentions=sum(len(r['mentions']) for r in records), mixed_label_positions=int(mixed.sum()),
        checkpoint_sha256=digest(checkpoint), original_probe_auc=float(old_auc), reference_auc=float(reference),
        checkpoint_real_class=1, feature_order='I[0:36] followed by E_l[0:36]',
        probe=metrics, local_visual_support31=support,
        native_checked_tokens=sum(b['checked_tokens'] for b in batches),
        native_argmax_mismatches=sum(b['native_argmax_mismatches'] for b in batches),
        object_argmax_mismatches=sum(not r['native_argmax_matches'] for r in records),
        semantic_cache_files=len(list((root / 'semantics').glob('*.npz'))),
        gpu_seconds=sum(b['seconds'] for b in batches),
        semantic_seconds=sum(b['semantic_seconds'] for b in batches),
        feature_seconds=sum(b['feature_seconds'] for b in batches),
        notes='Frozen probe; no fitting or threshold choice on SADI labels. All 100 images were previously observed for mitigation development.')
    output = root / 'analysis.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(root / 'predictions.npz', image_id=np.array([r['image_id'] for r in records]),
        step=np.array([r['step'] for r in records]), token_id=tokens, features=features,
        p_real=real, hallucinated=labels, in_training_vocabulary=known, noun=nouns, repeated=repeated, visual_support31=v)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
