"""Fit the original probe recipe on audited native-SADI source features.

Preparation fixes the recipe before fitting. Current development or reserved
features and labels are not read by this script. Seed43 stays the primary
generation gate; seeds44/45 are reported without choosing a best seed.
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
    assert not root.exists()
    source=REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'
    original=source/'risk+I+E_l/seed43'
    producer=REPO/'outputs/qwen3_native_sadi_source_features_20261119'
    producer_plan=read(producer/'protocol.json')
    assert producer_plan['full_source'] and producer_plan['source_labels_and_original_image_split_unchanged']
    config=read(original/'config.json')
    assert config['hidden_sizes']==[128] and config['batch_norm'] is False
    assert config['standardize_input'] and config['drop_last']
    assert config['checkpoint_selection']=='minimum_train_loss'
    assert config['positive_class']=='real' and config['split_protocol']=='strict_82_no_validation'
    paths=[Path(__file__),REPO/'scripts/verify_qwen3_teacher_sadi_capture.py',
           REPO/'scripts/train_torch_probe_feature_sets.py',
           REPO/'features/qwen3_risk_i_el_probe.py',REPO/'features/qwen3_online_i_el.py',
           source/'protocol.json',source/'compact_features.npz',
           original/'config.json',original/'model.pt',original/'input_scaler.npz',original/'metrics.json',
           producer/'protocol.json',
           REPO/'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json']
    hashes={str(path):sha(path) for path in paths}
    root.mkdir();(root/'source').mkdir()
    for path in paths:
        if path.is_relative_to(REPO) and path.suffix=='.py':
            target=root/'source'/path.relative_to(REPO)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,
        feature_producer=str(producer),producer_protocol_sha256=sha(producer/'protocol.json'),
        original_source=str(source),original_probe=str(original),config=config,seeds=[43,44,45],
        primary_seed_fixed=43,feature_order=['risk0:36','I0:36','E_l0:36'],
        split='Original source3200/800 images,9384/2367 targets, no validation. Do not redivide images.',
        inference='Complete native SADI5--18 under forced archived raw source captions.',
        scaler='Train-only, unchanged original recipe.',
        selection='Minimum train loss; source test is never a checkpoint monitor.',
        operating_point='5th percentile of source test_mask real scores, matching the original gate calibration. This subset is calibration, not an independent threshold test.',
        threshold_reports='Keep original train-F1/fixed0.5 classification reports separately from mitigation operating point.',
        no_current_or_reserved_features_or_labels=True,
        next='Separate fixed-seed43 development diagnosis before any new generation; no automatic500 from a source classification score.'))
    print(dict(prepared=True,root=str(root),seeds=[43,44,45],primary=43),flush=True)


def fit(root,device):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.train_torch_probe_feature_sets import TorchProbeConfig,train_and_evaluate_probe
    plan=read(root/'protocol.json')
    assert not (root/'started.json').exists(),'No implicit retraining or checkpoint replacement.'
    for filename,expected in plan['source_sha256'].items():assert sha(Path(filename))==expected,filename
    producer=Path(plan['feature_producer']);verification=read(producer/'verification.json')
    assert verification['passed'] and verification['complete_source_row_identity']
    assert verification['protocol_sha256']==plan['producer_protocol_sha256']
    assert sha(producer/'features.npz')==verification['feature_sha256']
    old=np.load(Path(plan['original_source'])/'compact_features.npz',allow_pickle=False)
    with np.load(producer/'features.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['row'],np.arange(len(old['label'])))
        features=archive['features'].copy()
    assert features.shape==(11751,108) and features.dtype==np.float32 and np.isfinite(features).all()
    train=old['train_mask'];test=old['test_mask'];labels=old['label']
    assert train.sum()==9384 and test.sum()==2367 and np.logical_xor(train,test).all()
    assert not set(map(int,old['image_id'][train]))&set(map(int,old['image_id'][test]))
    assert labels[train].sum()==7278 and labels[test].sum()==1848
    write(root/'started.json',dict(started_at=time.time(),device=device,
        feature_sha256=verification['feature_sha256'],feature_verification_sha256=sha(producer/'verification.json'),
        original_rows_and_labels_verified=True,training_only_scaler=True))
    torch.set_num_threads(1)
    raw_features=np.concatenate([old['risk'],old['I'],old['E_l']],axis=1).astype(np.float32)
    original_probe=FullRiskIElProbe(plan['original_probe']).eval()
    old_score=original_probe(torch.from_numpy(raw_features)).numpy()
    original_auc=float(roc_auc_score(labels[test],old_score[test]))
    assert abs(original_auc-read(Path(plan['original_probe'])/'metrics.json')['auc'])<1e-6
    calibration=test&(labels==1)
    original_threshold=float(np.quantile(old_score[calibration],.05,method='linear'))
    reference=read(REPO/'outputs/qwen3_semantic_consensus_gate_transfer_20261119/calibration.json')
    assert abs(original_threshold-reference['thresholds']['native'])<1e-6
    scores={};reports=[]
    for seed in plan['seeds']:
        destination=root/f'seed{seed}';assert not destination.exists()
        config=dict(plan['config']);config['seed']=seed
        config['hidden_sizes']=tuple(config['hidden_sizes'])
        config['threshold_reporting']=tuple(config['threshold_reporting'])
        metrics=train_and_evaluate_probe(X_train=features[train],y_train=labels[train],
            X_val=np.empty((0,108),np.float32),y_val=np.empty(0,np.int32),
            X_test=features[test],y_test=labels[test],config=TorchProbeConfig(**config),
            device=torch.device(device),output_dir=str(destination))
        write(destination/'metrics.json',metrics)
        model=FullRiskIElProbe(destination).eval()
        prediction=model(torch.from_numpy(features)).numpy()
        assert abs(float(roc_auc_score(labels[test],prediction[test]))-metrics['auc'])<1e-6
        threshold=float(np.quantile(prediction[calibration],.05,method='linear'))
        write(destination/'calibration.json',dict(threshold=threshold,quantile=.05,method='linear',
            source_real_targets=int(calibration.sum()),calibration_only=True,
            no_development_labels=True,feature_sha256=verification['feature_sha256']))
        scores[f'seed{seed}']=prediction
        reports.append(dict(seed=seed,auc=metrics['auc'],real_aupr=metrics['aupr'],
            train_f1_threshold=metrics['decision_threshold'],mitigation_real_threshold=threshold,
            epochs_ran=metrics['epochs_ran'],best_epoch=metrics['best_epoch']))
    np.savez_compressed(root/'source_predictions.npz',row=np.arange(len(labels)),label=labels,
                        original_raw_score=old_score,**scores)
    for filename,expected in plan['source_sha256'].items():assert sha(Path(filename))==expected,filename
    write(root/'verification.json',dict(passed=True,protocol_sha256=sha(root/'protocol.json'),
        producer_feature_sha256=verification['feature_sha256'],source_predictions_sha256=sha(root/'source_predictions.npz'),
        reports=reports,primary_seed=43,original_source_auc_replayed=original_auc,
        original_gate_threshold_replayed=original_threshold,source_labels_and_split_verified=True,
        no_current_or_reserved_data=True,no_mitigation_claim=True))
    print(reports,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root)
        else:fit(args.root,args.device)
    except Exception:
        if args.root.exists() and not (args.root/'failure.json').exists():
            write(args.root/'failure.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
