"""Calibrate one fixed score fusion on old data; inspect cached SADI slots."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_semantic_consensus_probe import FullSemanticConsensusProbe, consensus_real_score


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda:handle.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False);handle.write('\n')


def run(root):
    assert not root.exists(), 'Inspect an existing diagnostic rather than rerunning it.'
    trace=REPO/'outputs/qwen3_full_risk_f_trace_20261118'
    static=REPO/'outputs/qwen3_static_semantic_gate_transfer_20261119'
    base=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
    train=REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'
    old=read(base/'protocol.json');native=Path(old['probe']);semantic=train/'E_l/seed43'
    paths=[Path(__file__),REPO/'features/qwen3_semantic_consensus_probe.py',
        REPO/'features/qwen3_risk_i_el_probe.py',REPO/'features/qwen3_static_semantic_probe.py',
        REPO/'features/qwen3_online_i_el.py',trace/'labelled_positions.json',trace/'unlabelled_positions.json',
        trace/'final_provenance.json',static/'protocol.json',static/'predictions.json',
        static/'final_provenance.json',base/'protocol.json',base/'cohort.json',train/'compact_features.npz',train/'protocol.json']
    paths.extend(directory/name for directory in [native,semantic] for name in ['model.pt','input_scaler.npz','metrics.json'])
    inputs={str(p):sha(p) for p in paths}
    root.mkdir()
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=inputs,
        before_inspecting_new_target_scores=True,
        hypothesis='A semantic-only anchor may reduce attention-induced false alarms without a hard semantic veto.',
        formula='sigmoid(0.5*(logit(clip(p_SAM,1e-6,1-1e-6))+logit(clip(p_El,1e-6,1-1e-6))))',
        weights=[.5,.5],source_quantile=.05,quantile_method='linear',
        calibration='Archived source test_mask AND real label; now used as calibration, not independent threshold evaluation.',
        controls='Original SAM-V2 threshold, and E_l-only at the same source5% quantile; report original E_l F1 threshold separately.',
        no_current_label_threshold_fit=True,no_neural_generation=True,no_model_refitting=True,
        scope='Same209 cached native SADI slots,191 aligned automatic labels and18 unaligned; old development100.',
        pilot_rule='A pilot may test real-content retention if the verified fork/spoon error stays covered and aligned-real triggers decrease vs original108. No success or500 claim from diagnosis.',
        validation='Reserved1903 unchanged and unconsumed.'))
    torch.set_num_threads(1)
    assert read(trace/'final_provenance.json')['passed'] and read(static/'final_provenance.json')['passed']
    model=FullSemanticConsensusProbe(native,semantic).eval()
    with np.load(train/'compact_features.npz',allow_pickle=False) as handle:
        old_data={k:handle[k].copy() for k in handle.files}
    original=np.concatenate([old_data['risk'],old_data['I'],old_data['E_l']],axis=1).astype(np.float32)
    assert original.shape[1]==108
    with torch.inference_mode():
        x=torch.from_numpy(original)
        p_native=model.native(x).numpy()
        p_semantic=model.semantic(x[:,72:]).numpy()
        p_fused=model(x).numpy()
    test=old_data['test_mask'];cal=test&(old_data['label']==1)
    assert cal.sum()==1848
    ids=read(base/'cohort.json')['coco100']
    assert not set(ids)&set(map(int,old_data['image_id']))
    thresholds={name:float(np.quantile(score[cal],.05,method='linear'))
        for name,score in [('native',p_native),('semantic',p_semantic),('consensus',p_fused)]}
    assert abs(thresholds['native']-old['real_threshold'])<1e-6
    original_native_threshold=float(old['real_threshold'])
    reference_aucs={name:float(roc_auc_score(old_data['label'][test],score[test]))
                    for name,score in [('native',p_native),('semantic',p_semantic)]}
    assert abs(reference_aucs['native']-read(native/'metrics.json')['auc'])<1e-6
    assert abs(reference_aucs['semantic']-read(semantic/'metrics.json')['auc'])<1e-6
    # Independent odds-product calculation checks the fusion, including
    # clipping, without using the Torch logit/sigmoid implementation.
    a=np.clip(p_native.astype(np.float64),1e-6,1-1e-6)
    b=np.clip(p_semantic.astype(np.float64),1e-6,1-1e-6)
    yes=np.sqrt(a*b);no=np.sqrt((1-a)*(1-b));independent=yes/(yes+no)
    error=float(np.max(np.abs(independent-p_fused)));assert error<1e-12
    equal=torch.tensor([0.,.01,.2,.5,.8,.99,1.],dtype=torch.float64)
    assert torch.allclose(consensus_real_score(equal,equal),equal.clamp(1e-6,1-1e-6),atol=1e-14,rtol=0)
    assert float(consensus_real_score(torch.tensor([.2]),torch.tensor([.8])))==float(consensus_real_score(torch.tensor([.8]),torch.tensor([.2])))
    calibration=dict(passed=True,thresholds=thresholds,source_real_targets=int(cal.sum()),
        source_calibration_images=len(set(map(int,old_data['image_id'][cal]))),source_control_original_threshold=original_native_threshold,
        original_semantic_F1_threshold=read(semantic/'metrics.json')['decision_threshold'],
        source_aucs_before_threshold_calibration=reference_aucs,independent_fusion_max_error=error,
        no_target_label_fitting=True)
    write(root/'calibration.json',calibration)
    labelled=read(trace/'labelled_positions.json');unlabelled=read(trace/'unlabelled_positions.json')
    rows=labelled+unlabelled;assert len(rows)==209
    features=np.asarray([r['native_features'][:36]+r['native_I']+r['E_l'] for r in rows],np.float32)
    old_predictions={(r['image_id'],r['step'],r['token_id']):r for r in read(static/'predictions.json')}
    with torch.inference_mode():
        x=torch.from_numpy(features)
        target_native=model.native(x).numpy();target_semantic=model.semantic(x[:,72:]).numpy();target_fused=model(x).numpy()
    predictions=[]
    for i,r in enumerate(rows):
        key=r['image_id'],r['step'],r['token_id'];previous=old_predictions[key]
        assert abs(target_native[i]-previous['original108_p_real'])<1e-6
        assert abs(target_semantic[i]-previous['semantic_p_real'])<1e-6
        predictions.append(dict(image_id=r['image_id'],step=r['step'],token_id=r['token_id'],canonical=r['canonical'],
            context=r.get('context',previous['context']),automatic_hallucination=r.get('automatic_hallucination'),
            native_score=float(target_native[i]),semantic_score=float(target_semantic[i]),consensus_score=float(target_fused[i]),
            native_positive=bool(target_native[i]<original_native_threshold),
            semantic_positive=bool(target_semantic[i]<thresholds['semantic']),
            consensus_positive=bool(target_fused[i]<thresholds['consensus'])))
    def summarize(selected):
        return {mode:dict(positions=len(selected),triggered=sum(r[mode+'_positive'] for r in selected),
            automatic_false_triggered=sum(r[mode+'_positive'] and r['automatic_hallucination'] is True for r in selected),
            automatic_real_triggered=sum(r[mode+'_positive'] and r['automatic_hallucination'] is False for r in selected),
            unlabelled_triggered=sum(r[mode+'_positive'] and r['automatic_hallucination'] is None for r in selected))
            for mode in ['native','semantic','consensus']}
    totals=summarize(predictions)
    fork=[r for r in predictions if r['image_id']==353411 and r['canonical']=='fork'];assert len(fork)==1
    potential=bool(fork[0]['consensus_positive'] and totals['consensus']['automatic_real_triggered']<totals['native']['automatic_real_triggered'])
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),thresholds=thresholds,counts=totals,
        verified_fork_reference=fork[0],pilot_retention_screen=potential,
        added_triggers=[r for r in predictions if r['consensus_positive'] and not r['native_positive']],
        removed_triggers=[r for r in predictions if not r['consensus_positive'] and r['native_positive']],
        no_cuda_initialization=not torch.cuda.is_initialized(),no_new_generation=True,
        no_mitigation_claim=True,threshold_calibration_uses_source_only=True)
    assert result['no_cuda_initialization']
    for path,value in inputs.items():assert sha(Path(path))==value,path
    write(root/'predictions.json',predictions);write(root/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['added_triggers','removed_triggers']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
