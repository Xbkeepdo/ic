"""Fit small gates from cached COCO features using training-image cross-validation."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VIEWS = {
    'risk_F31': [('risk', 31), ('F_log', 31)],
    'risk_F18_21': [('risk', 18), ('F_log', 18), ('risk', 21), ('F_log', 21)],
    'I_El31': [('I', 31), ('E_l', 31)],
    'I_El18_21': [('I', 18), ('E_l', 18), ('I', 21), ('E_l', 21)],
}


def scores(labels, probabilities, tokens):
    weighted, pairs = 0., 0
    for token in np.unique(tokens):
        group = labels[tokens == token]
        count = int(group.sum()) * int((1 - group).sum())
        if count:
            weighted += count * roc_auc_score(group, probabilities[tokens == token])
            pairs += count
    return dict(auroc=float(roc_auc_score(labels, probabilities)),
                hall_aupr=float(average_precision_score(labels, probabilities)),
                within_token_auroc=weighted / pairs, within_token_pairs=pairs)


def estimator():
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(128,), random_state=43, batch_size=256,
        learning_rate_init=.001, max_iter=150, n_iter_no_change=20))


def run(cache, output):
    output.mkdir(exist_ok=False)
    data = np.load(cache, allow_pickle=False)
    train, test = data['train_mask'], data['test_mask']
    assert not (train & test).any() and (train | test).all()
    assert not set(data['image_id'][train]).intersection(data['image_id'][test])
    labels = (1 - data['label'][train]).astype(int)
    tokens = data['target_token_id'][train]
    folds = list(GroupKFold(n_splits=3).split(labels, labels, data['image_id'][train]))
    report = dict(cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                  train_targets=int(train.sum()), test_targets=int(test.sum()),
                  selection='Maximum within-token AUROC on 3-fold training-image cross-fit predictions.',
                  seed=43, hidden_units=128, views={})
    predictions = {}
    for name, columns in VIEWS.items():
        features = np.column_stack([data[key][train, layer] for key, layer in columns])
        oof = np.empty(len(labels))
        for fit, held in folds:
            model = estimator().fit(features[fit], labels[fit])
            oof[held] = model.predict_proba(features[held])[:, 1]
        report['views'][name] = dict(columns=columns, crossfit=scores(labels, oof, tokens))
        predictions[name] = oof
    chosen = max(VIEWS, key=lambda name: report['views'][name]['crossfit']['within_token_auroc'])
    columns = VIEWS[chosen]
    features = np.column_stack([data[key][:, layer] for key, layer in columns])
    model = estimator().fit(features[train], labels)
    probabilities = model.predict_proba(features[test])[:, 1]
    held_labels = (1 - data['label'][test]).astype(int)
    threshold = float(np.quantile(predictions[chosen][labels == 0], .95))
    report.update(chosen=chosen, heldout=scores(held_labels, probabilities, data['target_token_id'][test]),
                  threshold=threshold, threshold_rule='95th percentile of cross-fit real scores in training images.',
                  heldout_gate=dict(false_positive_rate=float((probabilities[held_labels == 0] >= threshold).mean()),
                                    hall_recall=float((probabilities[held_labels == 1] >= threshold).mean())))
    scaler, mlp = model.steps[0][1], model.steps[1][1]
    np.savez(output / 'gate.npz', mean=scaler.mean_, scale=scaler.scale_,
             weight0=mlp.coefs_[0], bias0=mlp.intercepts_[0],
             weight1=mlp.coefs_[1], bias1=mlp.intercepts_[1],
             feature_names=np.array([k for k, _ in columns]),
             layers=np.array([layer for _, layer in columns]), threshold=np.array(threshold))
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        run(args.cache, args.output)
