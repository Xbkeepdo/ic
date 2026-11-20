"""Score token-and-label-balanced source probes on cached development positions.

This program never fits a model or changes a threshold using development
labels. Preparation is separate from scoring so the recipe is recorded
before any newly trained probe sees the cached development features.
"""

import argparse
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def prepare(root):
    assert not root.exists(),'Preserve an existing preparation or diagnosis.'
    fit=REPO/'outputs/qwen3_token_balanced_source_probe_20261120'
    training=read(fit/'protocol.json')
    assert training['primary_seed_fixed']==43 and training['seeds']==[43,44,45]
    assert training['training_objective']=='BCE with token-and-label-balanced replacement sampling.'
    trace=REPO/'outputs/qwen3_full_risk_f_trace_20261118'
    static=REPO/'outputs/qwen3_static_semantic_gate_transfer_20261119'
    base=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
    reserved=REPO/'outputs/qwen3_mass_matched_visual_reserved500_20261119'
    assert read(trace/'final_provenance.json')['passed']
    assert read(static/'final_provenance.json')['passed']
    assert read(base/'final_provenance.json')['passed']
    assert not (reserved/'consumed.json').exists()
    original=Path(training['original_probe'])
    paths=[Path(__file__),REPO/'scripts/verify_qwen3_teacher_sadi_capture.py',
        REPO/'features/qwen3_risk_i_el_probe.py',REPO/'features/qwen3_online_i_el.py',
        fit/'protocol.json',trace/'protocol.json',trace/'labelled_positions.json',
        trace/'unlabelled_positions.json',trace/'final_provenance.json',
        static/'predictions.json',static/'final_provenance.json',
        base/'protocol.json',base/'cohort.json',base/'final_provenance.json',
        reserved/'cohort.json',reserved/'reservation.json',reserved/'batch_plan.json',
        Path(training['original_source'])/'compact_features.npz',
        original/'model.pt',original/'input_scaler.npz',original/'metrics.json',
        REPO/'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json',
        REPO/'docs/archive/QWEN3_ORIGINAL108_GATE_ANNOTATIONS_20261119.md']
    hashes={str(path):sha(path) for path in paths}
    root.mkdir()
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,
        trained_source_probe=str(fit),source_fit_protocol_sha256=sha(fit/'protocol.json'),
        original_probe=str(original),trace=str(trace),static=str(static),
        seeds=[43,44,45],primary_seed=43,feature_order=['risk0:36','I0:36','E_l0:36'],
        threshold='Use each already frozen source-real5% calibration; do not select by current labels.',
        scope='Same209 native SADI slots:191 automatic labels and18 unaligned. Cached old development100, not an independent test.',
        statistics='Real-positive AUC/AUPR, hall-positive AUPR, same-token pair AUC, exact trigger counts, all added/removed triggers.',
        initial_pilot_retention_screen='Primary43 still covers previously image-and-instance-verified353411 fork, and automatic-real triggers are fewer than original108.',
        interpretation='This is a low-cost coverage/preservation screen, not proof of generation improvement. Added triggers require image review; a different follow-up must be recorded separately, without retuning the model or threshold.',
        no_development_fit=True,no_neural_generation=True,no_automatic500=True,
        reserved='Existing1903 reservation stays unchanged and unconsumed.'))
    print(dict(prepared=True,root=str(root),primary_seed=43),flush=True)


def run(root):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score,roc_auc_score
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    plan=read(root/'protocol.json')
    assert not (root/'resolved_models.json').exists(),'No implicit repeat of target scoring.'
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    fit=Path(plan['trained_source_probe']);fitted=read(fit/'verification.json')
    assert fitted['passed'] and fitted['primary_seed']==43 and fitted['no_current_or_reserved_data']
    assert read(fit/'sampling_verification.json')['passed']
    assert fitted['protocol_sha256']==plan['source_fit_protocol_sha256']
    training=read(fit/'protocol.json')
    source=Path(training['feature_producer']);producer=read(source/'verification.json')
    assert producer['passed'] and producer['feature_sha256']==fitted['producer_feature_sha256']
    assert sha(source/'features.npz')==producer['feature_sha256']
    paths=[fit/'verification.json',fit/'source_predictions.npz',source/'verification.json',source/'features.npz']
    for seed in plan['seeds']:
        paths.extend(fit/f'seed{seed}'/name for name in ['model.pt','input_scaler.npz','config.json','metrics.json','calibration.json'])
    hashes={str(path):sha(path) for path in paths}
    write(root/'resolved_models.json',dict(resolved_at=time.time(),source_sha256=hashes,
        new_source_model_target_scores_not_read_yet=True,primary_seed=43))
    torch.set_num_threads(1)
    old=np.load(Path(training['original_source'])/'compact_features.npz',allow_pickle=False)
    calibration=old['test_mask']&(old['label']==1);assert calibration.sum()==1848
    with np.load(source/'features.npz',allow_pickle=False) as handle:
        np.testing.assert_array_equal(handle['row'],np.arange(len(old['label'])))
        source_features=handle['features'].copy()
    stored=np.load(fit/'source_predictions.npz',allow_pickle=False)
    assert sha(fit/'source_predictions.npz')==fitted['source_predictions_sha256']
    np.testing.assert_array_equal(stored['label'],old['label'])
    models={};thresholds={};replay={}
    for seed in plan['seeds']:
        directory=fit/f'seed{seed}';model=FullRiskIElProbe(directory).eval()
        assert read(directory/'config.json')['seed']==seed
        scores=model(torch.from_numpy(source_features)).numpy()
        error=float(np.max(np.abs(scores-stored[f'seed{seed}'])));assert error<1e-6
        threshold=float(np.quantile(scores[calibration],.05,method='linear'))
        assert abs(threshold-read(directory/'calibration.json')['threshold'])<1e-9
        models[f'native_seed{seed}']=model;thresholds[f'native_seed{seed}']=threshold
        replay[str(seed)]=dict(source_probability_max_error=error,threshold=threshold)
    trace=Path(plan['trace']);rows=read(trace/'labelled_positions.json')+read(trace/'unlabelled_positions.json')
    assert len(rows)==209
    keys=[(r['image_id'],r['step'],r['token_id']) for r in rows];assert len(set(keys))==209
    x=np.asarray([r['native_features'][:36]+r['native_I']+r['E_l'] for r in rows],np.float32)
    assert x.shape==(209,108) and np.isfinite(x).all()
    original=FullRiskIElProbe(plan['original_probe']).eval()
    old_threshold=read(REPO/'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json')['thresholds']['native']
    thresholds['original108']=old_threshold;models['original108']=original
    scores={name:model(torch.from_numpy(x)).numpy() for name,model in models.items()}
    assert all(np.isfinite(value).all() and ((value>=0)&(value<=1)).all() for value in scores.values())
    reference={(r['image_id'],r['step'],r['token_id']):r for r in read(Path(plan['static'])/'predictions.json')}
    assert set(reference)==set(keys)
    control_error=max(abs(float(scores['original108'][i])-reference[key]['original108_p_real']) for i,key in enumerate(keys))
    assert control_error<1e-6
    predictions=[]
    for i,(row,key) in enumerate(zip(rows,keys,strict=True)):
        prior=reference[key]
        item=dict(image_id=row['image_id'],step=row['step'],token_id=row['token_id'],canonical=row['canonical'],
            context=row.get('context',prior.get('context','')),
            automatic_hallucination=row.get('automatic_hallucination'),scores={},triggered={})
        for name in models:
            item['scores'][name]=float(scores[name][i]);item['triggered'][name]=bool(scores[name][i]<thresholds[name])
        predictions.append(item)
    labelled=np.asarray([r['automatic_hallucination'] is not None for r in predictions])
    hall=np.asarray([r['automatic_hallucination'] is True for r in predictions])[labelled]
    tokens=np.asarray([r['token_id'] for r in predictions])[labelled]
    assert labelled.sum()==191 and (~labelled).sum()==18
    counts={};metrics={}
    for name in models:
        triggered=np.asarray([r['triggered'][name] for r in predictions]);label_triggered=triggered[labelled]
        counts[name]=dict(positions=209,triggered=int(triggered.sum()),
            automatic_false_triggered=int((label_triggered&hall).sum()),
            automatic_real_triggered=int((label_triggered&~hall).sum()),unaligned_triggered=int(triggered[~labelled].sum()))
        p=scores[name][labelled];weighted=0.;pairs=0
        for token in np.unique(tokens):
            mask=tokens==token;number=int(hall[mask].sum())*int((~hall[mask]).sum())
            if number:
                weighted+=number*roc_auc_score(~hall[mask],p[mask]);pairs+=number
        metrics[name]=dict(real_auc=float(roc_auc_score(~hall,p)),real_aupr=float(average_precision_score(~hall,p)),
            hall_aupr=float(average_precision_score(hall,1-p)),same_token_auc=weighted/pairs if pairs else None,
            same_token_pairs=pairs,automatic_label_scope_only=True)
    assert counts['original108']['triggered']==38 and counts['original108']['automatic_real_triggered']==17
    primary='native_seed43'
    fork=[r for r in predictions if r['image_id']==353411 and r['step']==97 and r['token_id']==22435]
    assert len(fork)==1
    pilot_screen=bool(fork[0]['triggered'][primary] and counts[primary]['automatic_real_triggered']<counts['original108']['automatic_real_triggered'])
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),models_sha256=sha(root/'resolved_models.json'),
        thresholds=thresholds,source_model_replays=replay,original_target_score_max_error=control_error,
        counts=counts,metrics=metrics,primary_seed=43,initial_pilot_retention_screen=pilot_screen,
        verified_fork_reference=fork[0],no_development_fitting=True,no_neural_generation=True,
        no_cuda_initialized=not torch.cuda.is_initialized(),no_mitigation_claim=True,
        added_triggers=[r for r in predictions if r['triggered'][primary] and not r['triggered']['original108']],
        removed_triggers=[r for r in predictions if not r['triggered'][primary] and r['triggered']['original108']])
    assert result['no_cuda_initialized']
    for name,expected in {**plan['source_sha256'],**hashes}.items():assert sha(Path(name))==expected,name
    write(root/'predictions.json',predictions);write(root/'analysis.json',result)
    print({k:v for k,v in result.items() if k not in ['added_triggers','removed_triggers']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try:
        prepare(args.root) if args.prepare else run(args.root)
    except Exception:
        if args.root.exists() and not (args.root/'failure.json').exists():
            write(args.root/'failure.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
