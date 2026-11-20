"""Fit or calibrate gates on reused native SADI captions, without target labels."""

import argparse
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read, write, sha


def prepare(root):
    assert not root.exists()
    producer = REPO / 'outputs/qwen3_native_caption_source_20261120'
    original = REPO.parent / 'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b/risk+I+E_l/seed43'
    frozen = dict(original108=original,
        native4000=REPO / 'outputs/qwen3_native_sadi_source_probe_20261119/seed43',
        balanced4000=REPO / 'outputs/qwen3_token_balanced_source_probe_20261120/seed43')
    config = read(original / 'config.json')
    assert config['hidden_sizes'] == [128] and not config['batch_norm'] and config['drop_last']
    assert config['standardize_input'] and config['checkpoint_selection'] == 'minimum_train_loss'
    assert config['positive_class'] == 'real' and config['num_epochs'] == 100
    files = [Path(__file__), producer / 'protocol.json', producer / 'index.npz',
        REPO / 'scripts/train_torch_probe_feature_sets.py', REPO / 'features/qwen3_risk_i_el_probe.py',
        REPO / 'features/qwen3_online_i_el.py']
    for path in frozen.values():
        files += [path / name for name in ['config.json', 'model.pt', 'input_scaler.npz']]
    root.mkdir()
    write(root / 'protocol.json', dict(fixed_at=time.time(), source_sha256={str(p): sha(p) for p in files},
        producer=str(producer), config=config, frozen_models={name: str(path) for name, path in frozen.items()},
        seeds=[43, 44, 45], primary_seed=43,
        training='Original128-hidden recipe, from scratch; train-only scaler; minimum training loss; no validation/checkpoint selection on current candidates.',
        split='Producer400/100 source images, fixed for all three seeds. Retain only target positions matching the current native argmax.',
        calibration='5th percentile, linear, of native-matching real source-test targets. This source-test split is calibration, not an independent threshold evaluation.',
        frozen_comparators='Old three fixed seed43 models, re-calibrated on the same source real pool; weights and scalers unchanged.',
        candidate_order='Primary newly fitted seed43 and separately declared original108 with native-caption calibration; do not replace seed43 with seed44/45.',
        diagnostic_followup='Before generation, inspect cached209. A candidate needs either known fork coverage and fewer than17 automatically-real triggers, or added image-confirmed unsupported physical targets and no more than17 automatically-real triggers. This is development screening, not final efficacy.',
        no_current_or_reserved_features_or_labels=True))
    print(dict(prepared=True, primary=43, seeds=[43, 44, 45]), flush=True)


def fit(root):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.train_torch_probe_feature_sets import TorchProbeConfig, train_and_evaluate_probe
    plan = read(root / 'protocol.json'); producer = Path(plan['producer'])
    verified = read(producer / 'verification.json')
    assert verified['passed'] and sha(producer / 'features.npz') == verified['feature_sha256']
    assert sha(producer / 'protocol.json') == verified['protocol_sha256']
    for filename, expected in plan['source_sha256'].items(): assert sha(Path(filename)) == expected, filename
    assert not (root / 'started.json').exists()
    index = np.load(producer / 'index.npz', allow_pickle=False)
    with np.load(producer / 'features.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(index['row'], data['row'])
        features = data['features'].copy(); matching = data['native_match'].copy()
    labels = index['label_real']; train = matching & index['train']; test = matching & ~index['train']
    assert features.shape == (len(labels), 108) and np.isfinite(features).all()
    assert not set(map(int, index['image_id'][train])) & set(map(int, index['image_id'][test]))
    assert train.sum() >= plan['config']['batch_size']
    assert set(labels[train]) == set(labels[test]) == {0, 1}
    calibration = test & (labels == 1)
    write(root / 'started.json', dict(started_at=time.time(), device='cpu',
        producer_feature_sha256=verified['feature_sha256'], producer_verification_sha256=sha(producer / 'verification.json'),
        total=len(labels), excluded_native_mismatches=int((~matching).sum()),
        train_targets=int(train.sum()), test_targets=int(test.sum()), calibration_real=int(calibration.sum())))
    torch.set_num_threads(1)
    scores, operating_points, reports = {}, {}, []

    def score(name, path):
        model = FullRiskIElProbe(path).eval()
        with torch.inference_mode(): probability = model(torch.from_numpy(features)).numpy()
        assert probability.shape == (len(labels),) and np.isfinite(probability).all()
        threshold = float(np.quantile(probability[calibration], .05, method='linear'))
        scores[name] = probability
        operating_points[name] = dict(model=str(path), threshold=threshold,
            source_real_targets=int(calibration.sum()), calibration_only=True,
            auc=float(roc_auc_score(labels[test], probability[test])),
            real_aupr=float(average_precision_score(labels[test], probability[test])))
        return probability

    for name, filename in plan['frozen_models'].items(): score(name, Path(filename))
    for seed in plan['seeds']:
        destination = root / f'seed{seed}'; assert not destination.exists()
        options = dict(plan['config'], seed=seed)
        options['hidden_sizes'] = tuple(options['hidden_sizes'])
        options['threshold_reporting'] = tuple(options['threshold_reporting'])
        metrics = train_and_evaluate_probe(X_train=features[train], y_train=labels[train],
            X_val=np.empty((0, 108), np.float32), y_val=np.empty(0, np.int32),
            X_test=features[test], y_test=labels[test], config=TorchProbeConfig(**options),
            device=torch.device('cpu'), output_dir=str(destination))
        write(destination / 'metrics.json', metrics)
        probability = score(f'seed{seed}', destination)
        assert abs(float(roc_auc_score(labels[test], probability[test]))-metrics['auc']) < 1e-6
        write(destination / 'calibration.json', operating_points[f'seed{seed}'])
        reports.append(dict(seed=seed, **operating_points[f'seed{seed}'],
                            best_epoch=metrics['best_epoch'], epochs_ran=metrics['epochs_ran']))
    np.savez_compressed(root / 'source_predictions.npz', row=index['row'], label=labels,
                        train=train, test=test, native_match=matching, **scores)
    write(root / 'calibration.json', operating_points)
    for filename, expected in plan['source_sha256'].items(): assert sha(Path(filename)) == expected, filename
    assert not torch.cuda.is_initialized()
    write(root / 'verification.json', dict(passed=True, protocol_sha256=sha(root / 'protocol.json'),
        producer_feature_sha256=verified['feature_sha256'], source_predictions_sha256=sha(root / 'source_predictions.npz'),
        primary_seed=43, source_only=True, source_labels_and_split_verified=True,
        no_current_or_reserved_features_or_labels=True, cuda_initialized=False, reports=reports))
    print(dict(passed=True, operating_points=operating_points), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        else: fit(args.root)
    except Exception:
        if args.root.exists() and not (args.root / 'failure.json').exists():
            write(args.root / 'failure.json', dict(failed_at=time.time(), traceback=traceback.format_exc()))
        raise
