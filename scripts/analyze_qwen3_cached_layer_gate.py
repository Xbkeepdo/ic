"""Select two layers inside training-image folds for a compact evidence gate."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from analyze_qwen3_cached_gate import estimator, scores

FAMILIES = {'risk_F': (('risk', 1), ('F_log', -1)), 'I_El': (('I', -1), ('E_l', -1))}


def select_layers(data, mask, family):
    labels = 1 - data['label'][mask]
    tokens = data['target_token_id'][mask]
    features = np.stack([data[key][mask] * sign for key, sign in FAMILIES[family]], axis=-1)
    features = features.reshape(len(labels), -1)
    numerator, pairs = np.zeros(features.shape[1]), 0
    for token in np.unique(tokens):
        group = tokens == token
        y = labels[group].astype(bool)
        n_pos = int(y.sum())
        n_pairs = n_pos * int((~y).sum())
        if n_pairs:
            ranks = rankdata(features[group], axis=0)
            numerator += ranks[y].sum(0) - n_pos * (n_pos + 1) / 2
            pairs += n_pairs
    per_layer = (numerator / pairs).reshape(-1, 2).mean(1)
    layers = np.argsort(-per_layer, kind='stable')[:2].tolist()
    return [(key, layer) for layer in layers for key, _ in FAMILIES[family]], per_layer.tolist()


def matrix(data, columns):
    return np.column_stack([data[key][:, layer] for key, layer in columns])


def run(cache, output):
    output.mkdir(exist_ok=False)
    data = np.load(cache, allow_pickle=False)
    train, test = data['train_mask'], data['test_mask']
    assert not (train & test).any()
    assert not set(data['image_id'][train]).intersection(data['image_id'][test])
    labels = (1 - data['label'][train]).astype(int)
    train_indices = np.flatnonzero(train)
    folds = list(GroupKFold(n_splits=3).split(labels, labels, data['image_id'][train]))
    report = dict(cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                  selection='Within-token feature ranking selects two layers inside each training-image fold. Choose family by nested cross-fit within-token AUROC.',
                  families={})
    probabilities = {}
    for family in FAMILIES:
        oof = np.empty(len(labels))
        fold_columns = []
        for fit, held in folds:
            fit_mask = np.zeros(len(train), dtype=bool)
            fit_mask[train_indices[fit]] = True
            columns, _ = select_layers(data, fit_mask, family)
            features = matrix(data, columns)[train]
            model = estimator().fit(features[fit], labels[fit])
            oof[held] = model.predict_proba(features[held])[:, 1]
            fold_columns.append(columns)
        report['families'][family] = dict(fold_columns=fold_columns,
                                          crossfit=scores(labels, oof, data['target_token_id'][train]))
        probabilities[family] = oof
    chosen = max(FAMILIES, key=lambda family: report['families'][family]['crossfit']['within_token_auroc'])
    columns, ranks = select_layers(data, train, chosen)
    features = matrix(data, columns)
    model = estimator().fit(features[train], labels)
    held = model.predict_proba(features[test])[:, 1]
    held_labels = (1 - data['label'][test]).astype(int)
    threshold = float(np.quantile(probabilities[chosen][labels == 0], .95))
    report.update(chosen=chosen, columns=columns, full_train_layer_ranking=ranks,
                  heldout=scores(held_labels, held, data['target_token_id'][test]),
                  threshold=threshold, heldout_gate=dict(
                      false_positive_rate=float((held[held_labels == 0] >= threshold).mean()),
                      hall_recall=float((held[held_labels == 1] >= threshold).mean())))
    scaler, mlp = model.steps[0][1], model.steps[1][1]
    np.savez(output / 'gate.npz', mean=scaler.mean_, scale=scaler.scale_,
             weight0=mlp.coefs_[0], bias0=mlp.intercepts_[0], weight1=mlp.coefs_[1], bias1=mlp.intercepts_[1],
             feature_names=np.array([key for key, _ in columns]), layers=np.array([layer for _, layer in columns]),
             threshold=np.array(threshold))
    transformed = scaler.transform(features[test])
    hidden = np.maximum(0, transformed @ mlp.coefs_[0] + mlp.intercepts_[0])
    reconstructed = 1 / (1 + np.exp(-(hidden @ mlp.coefs_[1] + mlp.intercepts_[1])))
    np.testing.assert_allclose(reconstructed[:, 0], held, atol=1e-6, rtol=0)
    report['portable_forward_verified'] = True
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        run(args.cache, args.output)
