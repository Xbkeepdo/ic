"""Check native-caption gates on existing development features, without replay."""

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
    fit = REPO / 'outputs/qwen3_native_caption_probe_recovery_20261120'
    trace = REPO / 'outputs/qwen3_full_risk_f_trace_20261118'
    static = REPO / 'outputs/qwen3_static_semantic_gate_transfer_20261119'
    reserved = REPO / 'outputs/qwen3_mass_matched_visual_reserved500_20261119'
    assert read(trace / 'final_provenance.json')['passed']
    assert read(static / 'final_provenance.json')['passed']
    assert not (reserved / 'consumed.json').exists()
    paths = [Path(__file__), fit / 'protocol.json',
        *[trace / leaf for leaf in ['labelled_positions.json', 'unlabelled_positions.json', 'final_provenance.json']],
        static / 'predictions.json', static / 'final_provenance.json',
        REPO / 'features/qwen3_risk_i_el_probe.py', REPO / 'features/qwen3_online_i_el.py',
        REPO / 'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json',
        *[reserved / leaf for leaf in ['cohort.json', 'reservation.json', 'batch_plan.json']]]
    root.mkdir()
    write(root / 'protocol.json', dict(fixed_at=time.time(), source_sha256={str(p): sha(p) for p in paths},
        fit=str(fit), trace=str(trace), static=str(static),
        source_fit_protocol_sha256=sha(fit / 'protocol.json'),
        primary='seed43', candidates=['seed43', 'original108'],
        additional_reports=['seed44', 'seed45', 'native4000', 'balanced4000'],
        thresholds='Use source-native-caption calibration already sealed by the fit; no tuning on current labels.',
        scope='Existing209 slots,191 automatically labelled and18 unaligned; reused100-image development set, not independent validation.',
        screen=read(fit / 'protocol.json')['diagnostic_followup'],
        images='Added automatic triggers require original-image review before any generation pilot.',
        no_fitting=True, no_new_captions=True, no_automatic500=True))
    print(dict(prepared=True, candidates=['seed43', 'original108']), flush=True)


def run(root):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    plan = read(root / 'protocol.json'); fit = Path(plan['fit'])
    assert not (root / 'resolved_models.json').exists()
    for filename, expected in plan['source_sha256'].items(): assert sha(Path(filename)) == expected, filename
    verified = read(fit / 'verification.json'); assert verified['passed']
    assert verified['protocol_sha256'] == plan['source_fit_protocol_sha256']
    training = read(fit / 'protocol.json'); producer = Path(training['producer'])
    calibration = read(fit / 'calibration.json')
    paths = [fit / name for name in ['verification.json', 'calibration.json', 'source_predictions.npz']]
    paths += [producer / name for name in ['verification.json', 'features.npz', 'index.npz']]
    for value in calibration.values():
        paths += [Path(value['model']) / name for name in ['model.pt', 'input_scaler.npz', 'config.json']]
    hashes = {str(p): sha(p) for p in paths}
    write(root / 'resolved_models.json', dict(resolved_at=time.time(), source_sha256=hashes,
        target_scores_not_read_yet=True, candidate_names=plan['candidates']))
    source = np.load(producer / 'features.npz', allow_pickle=False)
    source_scores = np.load(fit / 'source_predictions.npz', allow_pickle=False)
    assert sha(producer / 'features.npz') == verified['producer_feature_sha256']
    assert sha(fit / 'source_predictions.npz') == verified['source_predictions_sha256']
    rows = read(Path(plan['trace']) / 'labelled_positions.json') + read(Path(plan['trace']) / 'unlabelled_positions.json')
    features = np.asarray([r['native_features'][:36] + r['native_I'] + r['E_l'] for r in rows], np.float32)
    assert features.shape == (209, 108) and np.isfinite(features).all()
    assert not set(r['image_id'] for r in rows) & set(read(producer / 'protocol.json')['images'])
    keys = [(r['image_id'], r['step'], r['token_id']) for r in rows]; assert len(set(keys)) == 209
    torch.set_num_threads(1)
    scores, thresholds, replays = {}, {}, {}
    for name, details in calibration.items():
        model = FullRiskIElProbe(details['model']).eval()
        with torch.inference_mode():
            replay = model(torch.from_numpy(source['features'])).numpy()
            scores[name] = model(torch.from_numpy(features)).numpy()
        error = float(np.max(np.abs(replay-source_scores[name]))); assert error < 1e-6
        mask = source_scores['test'] & (source_scores['label'] == 1)
        threshold = float(np.quantile(replay[mask], .05, method='linear'))
        assert abs(threshold-details['threshold']) < 1e-9
        thresholds[name] = threshold; replays[name] = error
    scores['reference'] = scores['original108']
    thresholds['reference'] = read(REPO / 'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json')['thresholds']['native']
    old = {(r['image_id'], r['step'], r['token_id']): r for r in read(Path(plan['static']) / 'predictions.json')}
    error = max(abs(float(scores['reference'][i])-old[key]['original108_p_real']) for i, key in enumerate(keys))
    assert error < 1e-6
    predictions = []
    for i, (row, key) in enumerate(zip(rows, keys, strict=True)):
        predictions.append(dict(image_id=row['image_id'], step=row['step'], token_id=row['token_id'],
            canonical=row['canonical'], context=row.get('context', old[key].get('context', '')),
            automatic_hallucination=row.get('automatic_hallucination'),
            scores={name: float(value[i]) for name, value in scores.items()},
            triggered={name: bool(value[i] < thresholds[name]) for name, value in scores.items()}))
    known = np.asarray([r['automatic_hallucination'] is not None for r in predictions])
    hall = np.asarray([r['automatic_hallucination'] is True for r in predictions])[known]
    tokens = np.asarray([r['token_id'] for r in predictions])[known]
    assert known.sum() == 191 and hall.sum() == 29
    counts, metrics = {}, {}
    for name, value in scores.items():
        triggered = value < thresholds[name]; p = value[known]
        counts[name] = dict(triggered=int(triggered.sum()), automatic_false_triggered=int((triggered[known] & hall).sum()),
            automatic_real_triggered=int((triggered[known] & ~hall).sum()), unaligned_triggered=int(triggered[~known].sum()))
        pair_count, weighted = 0, 0.
        for token in np.unique(tokens):
            mask = tokens == token; pairs = int(hall[mask].sum()) * int((~hall[mask]).sum())
            if pairs: pair_count += pairs; weighted += pairs * roc_auc_score(~hall[mask], p[mask])
        metrics[name] = dict(real_auc=float(roc_auc_score(~hall, p)), real_aupr=float(average_precision_score(~hall, p)),
            same_token_auc=weighted/pair_count, same_token_pairs=pair_count)
    assert counts['reference']['triggered'] == 38 and counts['reference']['automatic_real_triggered'] == 17
    fork, = [r for r in predictions if (r['image_id'], r['step'], r['token_id']) == (353411, 97, 22435)]
    screens = {name: bool(fork['triggered'][name] and counts[name]['automatic_real_triggered'] < 17) for name in plan['candidates']}
    added = {name: [r for r in predictions if r['triggered'][name] and not r['triggered']['reference']] for name in plan['candidates']}
    output = dict(passed=True, protocol_sha256=sha(root / 'protocol.json'), model_manifest_sha256=sha(root / 'resolved_models.json'),
        source_replay_max_errors=replays, original_target_score_max_error=error,
        thresholds=thresholds, counts=counts, metrics=metrics, known_fork=fork,
        fork_retention_and_real_fp_screens=screens, added_triggers=added,
        no_new_generation=True, no_fitting=True, cuda_initialized=torch.cuda.is_initialized(), no_mitigation_claim=True)
    assert not output['cuda_initialized']
    for filename, expected in {**plan['source_sha256'], **hashes}.items(): assert sha(Path(filename)) == expected, filename
    write(root / 'predictions.json', predictions); write(root / 'analysis.json', output)
    print({key: value for key, value in output.items() if key not in ['added_triggers', 'known_fork']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    try:
        prepare(args.root) if args.prepare else run(args.root)
    except Exception:
        if args.root.exists() and not (args.root / 'failure.json').exists():
            write(args.root / 'failure.json', dict(failed_at=time.time(), traceback=traceback.format_exc()))
        raise
