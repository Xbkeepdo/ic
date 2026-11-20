"""Two source-only, frozen-weight adaptation controls for the original gate.

Control1 recalibrates only the source operating point. Control2 also replaces
the input moments with native-SADI source-train moments. Both reuse the same
audited source trace; neither trains a network or reads new images.
"""

import argparse
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def prepare(root):
    import numpy as np
    assert not root.exists()
    fit=REPO/'outputs/qwen3_native_sadi_source_probe_20261119'
    fit_plan=read(fit/'protocol.json')
    source=Path(fit_plan['original_source']);original=Path(fit_plan['original_probe'])
    producer=Path(fit_plan['feature_producer'])
    trace=REPO/'outputs/qwen3_full_risk_f_trace_20261118'
    static=REPO/'outputs/qwen3_static_semantic_gate_transfer_20261119'
    assert read(trace/'final_provenance.json')['passed'] and read(static/'final_provenance.json')['passed']
    paths=[Path(__file__),REPO/'scripts/verify_qwen3_teacher_sadi_capture.py',
        REPO/'features/qwen3_risk_i_el_probe.py',REPO/'features/qwen3_online_i_el.py',
        fit/'protocol.json',producer/'protocol.json',source/'protocol.json',source/'compact_features.npz',
        original/'model.pt',original/'input_scaler.npz',original/'config.json',original/'metrics.json',
        trace/'labelled_positions.json',trace/'unlabelled_positions.json',trace/'final_provenance.json',
        static/'predictions.json',static/'final_provenance.json',
        REPO/'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json',
        REPO/'docs/archive/QWEN3_ORIGINAL108_GATE_ANNOTATIONS_20261119.md']
    hashes={str(path):sha(path) for path in paths}
    # Replay against a real saved standardizer, not just a synthetic identity.
    with np.load(source/'compact_features.npz',allow_pickle=False) as old:
        raw=np.concatenate([old['risk'],old['I'],old['E_l']],axis=1).astype(np.float64)
        train=old['train_mask'];assert train.sum()==9384
        raw_mean=raw[train].mean(0);raw_std=raw[train].std(0)
        raw_scale=np.where(raw_std>1e-12,raw_std,1.)
    with np.load(original/'input_scaler.npz',allow_pickle=False) as saved:
        np.testing.assert_allclose(raw_mean,saved['mean'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(raw_scale,saved['scale'],rtol=1e-12,atol=1e-12)
        old_mean_error=float(np.max(np.abs(raw_mean-saved['mean'])))
        old_scale_error=float(np.max(np.abs(raw_scale-saved['scale'])))
    root.mkdir();(root/'source').mkdir()
    write(root/'normalization_recipe_replay.json',dict(passed=True,source_train_targets=int(train.sum()),
        saved_original_mean_max_error=old_mean_error,saved_original_scale_max_error=old_scale_error,
        no_model_calls=True,no_target_features_read=True))
    for path in paths:
        if path.is_relative_to(REPO) and path.suffix=='.py':
            target=root/'source'/path.relative_to(REPO)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,
        producer=str(producer),producer_protocol_sha256=sha(producer/'protocol.json'),
        original_source=str(source),original_probe=str(original),trace=str(trace),static=str(static),
        original_seed_fixed=43,feature_order=['risk0:36','I0:36','E_l0:36'],
        controls=dict(calibration_only='Original weights and scaler, source-native real5% threshold.',
            native_moments='Original weights; mean/std from source-native TRAIN mask only; source-native real5% threshold.'),
        moments='Float64 population mean/std over9384 source-train targets; std<=1e-12 maps to1, exactly the original trainer convention.',
        quantile='Source test_mask AND real label1848, q=.05, linear. Calibration subset, not independent threshold evaluation.',
        independent_check='Native-moment output must equal the original gate on (x-mu_native)/sd_native*sd_raw+mu_raw to1e-6; copied checkpoint bytes must equal original.',
        initial_pilot_retention_screen='Previously verified353411 fork remains covered and automatic-real triggers decrease vs original108, assessed separately for both controls.',
        scope='Same cached209 native SADI development slots; report all variants and added/removed triggers, not a generation result.',
        no_neural_refitting=True,no_new_generation=True,no_current_threshold_or_moment_fit=True,
        no_automatic500=True,separate_from_fixed_primary_native_retrained_seed43=True))
    print(dict(prepared=True,root=str(root),controls=['calibration_only','native_moments']),flush=True)


def run(root):
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score,roc_auc_score
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    plan=read(root/'protocol.json')
    assert not (root/'resolved_source.json').exists(),'Do not implicitly repeat source calibration or target scoring.'
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    producer=Path(plan['producer']);verified=read(producer/'verification.json')
    assert verified['passed'] and verified['complete_source_row_identity']
    assert verified['protocol_sha256']==plan['producer_protocol_sha256']
    assert sha(producer/'features.npz')==verified['feature_sha256']
    write(root/'resolved_source.json',dict(resolved_at=time.time(),feature_sha256=verified['feature_sha256'],
        verification_sha256=sha(producer/'verification.json'),protocol_sha256=sha(root/'protocol.json')))
    torch.set_num_threads(1)
    old=np.load(Path(plan['original_source'])/'compact_features.npz',allow_pickle=False)
    with np.load(producer/'features.npz',allow_pickle=False) as handle:
        np.testing.assert_array_equal(handle['row'],np.arange(len(old['label'])))
        source_x=handle['features'].copy()
    train=old['train_mask'];test=old['test_mask'];labels=old['label'];calibration=test&(labels==1)
    assert source_x.shape==(11751,108) and np.isfinite(source_x).all()
    assert train.sum()==9384 and test.sum()==2367 and calibration.sum()==1848
    assert np.logical_xor(train,test).all()
    assert not set(map(int,old['image_id'][train]))&set(map(int,old['image_id'][test]))
    native_train=np.asarray(source_x[train],dtype=np.float64)
    mean=native_train.mean(0);std=native_train.std(0);scale=np.where(std>1e-12,std,1.)
    assert mean.shape==scale.shape==(108,) and np.isfinite(mean).all() and np.isfinite(scale).all()
    original=FullRiskIElProbe(plan['original_probe']).eval()
    moment=FullRiskIElProbe(plan['original_probe']).eval()
    assert original.mean.dtype==original.scale.dtype==torch.float64
    raw_mean=original.mean.numpy().copy();raw_scale=original.scale.numpy().copy()
    moment.mean.copy_(torch.from_numpy(mean));moment.scale.copy_(torch.from_numpy(scale))
    directory=root/'native_moments';directory.mkdir()
    shutil.copyfile(Path(plan['original_probe'])/'model.pt',directory/'model.pt')
    assert sha(directory/'model.pt')==sha(Path(plan['original_probe'])/'model.pt')
    np.savez(directory/'input_scaler.npz',mean=mean,scale=scale)
    shutil.copyfile(Path(plan['original_probe'])/'config.json',directory/'original_training_config.json')
    # This algebraic replay uses the original unchanged scaler and network,
    # independently of replacing buffers in the moment-adapted instance.
    transformed=(source_x.astype(np.float64)-mean)/scale*raw_scale+raw_mean
    replay=original(torch.from_numpy(transformed)).numpy()
    moment_source=moment(torch.from_numpy(source_x)).numpy()
    error=float(np.max(np.abs(replay-moment_source)));assert error<1e-6
    persisted=FullRiskIElProbe(directory).eval()
    np.testing.assert_array_equal(persisted(torch.from_numpy(source_x)).numpy(),moment_source)
    moment=persisted
    frozen_source=original(torch.from_numpy(source_x)).numpy()
    raw_x=np.concatenate([old['risk'],old['I'],old['E_l']],axis=1).astype(np.float32)
    raw_score=original(torch.from_numpy(raw_x)).numpy()
    original_auc=float(roc_auc_score(labels[test],raw_score[test]))
    assert abs(original_auc-read(Path(plan['original_probe'])/'metrics.json')['auc'])<1e-6
    thresholds=dict(original108=float(np.quantile(raw_score[calibration],.05,method='linear')),
        calibration_only=float(np.quantile(frozen_source[calibration],.05,method='linear')),
        native_moments=float(np.quantile(moment_source[calibration],.05,method='linear')))
    reference_threshold=read(REPO/'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json')['thresholds']['native']
    assert abs(thresholds['original108']-reference_threshold)<1e-6
    source_metrics={name:dict(real_auc=float(roc_auc_score(labels[test],p[test])),
        real_aupr=float(average_precision_score(labels[test],p[test])))
        for name,p in [('frozen_on_native',frozen_source),('native_moments',moment_source)]}
    write(root/'source_calibration.json',dict(thresholds=thresholds,source_real_targets=int(calibration.sum()),
        independent_moment_mapping_max_error=error,network_checkpoint_sha256=sha(directory/'model.pt'),
        new_scaler_sha256=sha(directory/'input_scaler.npz'),source_ranking_metrics=source_metrics,
        scaler_fit_source_train_targets=int(train.sum()),no_target_calibration=True))
    write(directory/'adaptation.json',dict(network='Unmodified original raw-source seed43 checkpoint.',
        source_moments='Native-SADI source train9384 targets only.',
        threshold=thresholds['native_moments'],quantile=.05,calibration_only=True,
        no_neural_training=True,source_feature_sha256=verified['feature_sha256']))
    trace=Path(plan['trace']);rows=read(trace/'labelled_positions.json')+read(trace/'unlabelled_positions.json')
    keys=[(r['image_id'],r['step'],r['token_id']) for r in rows]
    assert len(keys)==len(set(keys))==209
    reference={(r['image_id'],r['step'],r['token_id']):r for r in read(Path(plan['static'])/'predictions.json')}
    assert set(reference)==set(keys)
    x=np.asarray([r['native_features'][:36]+r['native_I']+r['E_l'] for r in rows],np.float32)
    assert x.shape==(209,108) and np.isfinite(x).all()
    frozen_target=original(torch.from_numpy(x)).numpy()
    scores=dict(original108=frozen_target,calibration_only=frozen_target,
                native_moments=moment(torch.from_numpy(x)).numpy())
    assert all(np.isfinite(p).all() and ((p>=0)&(p<=1)).all() for p in scores.values())
    baseline_error=max(abs(float(frozen_target[i])-reference[key]['original108_p_real']) for i,key in enumerate(keys))
    assert baseline_error<1e-6
    predictions=[]
    for i,(row,key) in enumerate(zip(rows,keys,strict=True)):
        predictions.append(dict(image_id=row['image_id'],step=row['step'],token_id=row['token_id'],
            canonical=row['canonical'],automatic_hallucination=row.get('automatic_hallucination'),
            context=row.get('context',reference[key].get('context','')),
            scores={name:float(p[i]) for name,p in scores.items()},
            triggered={name:bool(p[i]<thresholds[name]) for name,p in scores.items()}))
    selected=np.asarray([r['automatic_hallucination'] is not None for r in predictions])
    hall=np.asarray([r['automatic_hallucination'] is True for r in predictions])[selected]
    assert selected.sum()==191 and (~selected).sum()==18
    counts={};metrics={}
    for name,p in scores.items():
        gate=np.asarray([r['triggered'][name] for r in predictions]);marked=gate[selected]
        counts[name]=dict(triggered=int(gate.sum()),automatic_false_triggered=int((marked&hall).sum()),
            automatic_real_triggered=int((marked&~hall).sum()),unaligned_triggered=int(gate[~selected].sum()))
        metrics[name]=dict(real_auc=float(roc_auc_score(~hall,p[selected])),
            real_aupr=float(average_precision_score(~hall,p[selected])),hall_aupr=float(average_precision_score(hall,1-p[selected])))
    assert counts['original108']['triggered']==38 and counts['original108']['automatic_real_triggered']==17
    assert metrics['calibration_only']==metrics['original108']
    fork=[r for r in predictions if r['image_id']==353411 and r['step']==97 and r['token_id']==22435]
    assert len(fork)==1
    screens={name:bool(fork[0]['triggered'][name] and counts[name]['automatic_real_triggered']<17)
             for name in ['calibration_only','native_moments']}
    source_outputs=dict(row=np.arange(len(labels)),frozen_on_native=frozen_source,native_moments=moment_source,
                        original_on_raw=raw_score)
    np.savez_compressed(root/'source_predictions.npz',**source_outputs)
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),thresholds=thresholds,
        counts=counts,metrics=metrics,initial_pilot_retention_screens=screens,verified_fork_reference=fork[0],
        independent_moment_mapping_max_error=error,original_target_max_error=baseline_error,
        checkpoint_bytes_identical=True,train_only_moments=True,no_current_calibration=True,
        no_neural_training=True,no_new_generation=True,no_cuda_initialized=not torch.cuda.is_initialized(),
        no_mitigation_claim=True,source_predictions_sha256=sha(root/'source_predictions.npz'),
        transitions={name:dict(added=[r for r in predictions if r['triggered'][name] and not r['triggered']['original108']],
            removed=[r for r in predictions if not r['triggered'][name] and r['triggered']['original108']])
            for name in ['calibration_only','native_moments']})
    assert result['no_cuda_initialized']
    write(root/'predictions.json',predictions);write(root/'analysis.json',result)
    print({k:v for k,v in result.items() if k!='transitions'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    try:
        prepare(args.root) if args.prepare else run(args.root)
    except Exception:
        if args.root.exists() and not (args.root/'failure.json').exists():
            write(args.root/'failure.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
