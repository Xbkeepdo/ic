"""Explore old-domain heldout-real5th-percentile gates without new generation."""

import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from features.qwen3_online_i_el import FullIElProbe
from features.qwen3_risk_i_el_probe import FullRiskIElProbe
from scripts.trace_qwen3_full_risk_f import OUT as SOURCE
from scripts.verify_qwen3_full_risk_f import TRAIN
from scripts.run_qwen3_signed_feedback import read,sha,write

OUT=SOURCE/'source_calibration'
FAMILIES={'risk_F':('risk+F_log',('risk','F_log'),FullIElProbe),
          'I_El':('I+E_l',('I','E_l'),FullIElProbe),
          'SAM_V2':('risk+I+E_l',('risk','I','E_l'),FullRiskIElProbe)}


def run():
    torch.set_num_threads(1)
    assert read(SOURCE/'analysis.json')['passed'] and read(SOURCE/'cached_sam_v2/analysis.json')['passed']
    assert not OUT.exists()
    files=[Path(__file__),REPO/'features/qwen3_risk_i_el_probe.py',REPO/'features/qwen3_online_i_el.py',
        TRAIN/'compact_features.npz',SOURCE/'protocol.json',SOURCE/'labelled_positions.json',SOURCE/'cached_sam_v2/scores.json',
        *(SOURCE/f'features.shard{i}.jsonl' for i in (0,1)),
        *(TRAIN/family/'seed43'/f for family,_,_ in FAMILIES.values() for f in ('model.pt','input_scaler.npz'))]
    OUT.mkdir()
    write(OUT/'protocol.json',dict(source=str(SOURCE),fixed_at=time.time(),source_sha256={str(p):sha(p) for p in files},
        alpha=.05,rule='Each unchanged model: fifth percentile of real probabilities on OLD source-domain heldout-real targets; strict p_real<threshold.',
        scope='The old test split is used as source calibration here, not as an independent threshold evaluation. Its images are disjoint from the100 target images.',
        timing='Rule frozen before calculating its thresholds and applying them to target scores.',
        fitting='No model refit or threshold optimization on target labels. No new GPU/model generation/features.',
        variants=['risk_F','I_El','SAM_V2'],views=['native','prelocal'],
        screening='<=14 automatically-real positives and at least one confirmed unsupported object missed by old I/El; development screening only.'))
    calibration={}
    with np.load(TRAIN/'compact_features.npz',allow_pickle=False) as data,torch.inference_mode():
        mask=data['test_mask']&(data['label']==1)
        assert mask.sum()>1000
        assert not set(data['image_id'][mask]).intersection(read(SOURCE/'protocol.json')['ids'])
        for key,(directory,columns,loader) in FAMILIES.items():
            model=loader(TRAIN/directory/'seed43').eval()
            vectors=np.concatenate([data[column][mask] for column in columns],axis=1)
            probabilities=model(torch.from_numpy(vectors)).numpy()
            calibration[key]=dict(threshold=float(np.quantile(probabilities,.05)),source_real_targets=int(mask.sum()),
                source_real_images=len(set(data['image_id'][mask].tolist())),source_only=True,
                source_fraction_below=float((probabilities<np.quantile(probabilities,.05)).mean()))
    # Persist the source-only thresholds BEFORE loading target probabilities/labels.
    write(OUT/'calibration.json',calibration)
    records=[r for i in (0,1) for r in map(json.loads,(SOURCE/f'features.shard{i}.jsonl').read_bytes().splitlines())]
    key=lambda r:(r['image_id'],r['step'],r['token_id'])
    sam={key(r):r for r in read(SOURCE/'cached_sam_v2/scores.json')}
    labels={key(r):r for r in read(SOURCE/'labelled_positions.json')}
    assert len(records)==len(sam)==209 and len(labels)==191
    metrics={};additional={};unlabelled={}
    for family in FAMILIES:
        threshold=calibration[family]['threshold']
        for view in ('native','pre'):
            name=family+'_'+view
            def probability(r):
                if family=='SAM_V2':return sam[key(r)][view+'_p_real']
                return r[view+('_i_el_p_real' if family=='I_El' else '_p_real')]
            subset=[r for r in records if key(r) in labels]
            y=np.array([labels[key(r)]['automatic_hallucination'] for r in subset])
            selected=np.array([probability(r)<threshold for r in subset])
            metrics[name]=dict(threshold=threshold,selected=int(selected.sum()),automatic_false_selected=int((selected&y).sum()),
                automatic_real_selected=int((selected&~y).sum()),recall=float((selected&y).sum()/29),real_fpr=float((selected&~y).sum()/162))
            additional[name]=[dict(image_id=r['image_id'],step=r['step'],token_id=r['token_id'],p_real=probability(r),
                **{k:labels[key(r)][k] for k in ('canonical','token_text','context')}) for r in subset
                if probability(r)<threshold and labels[key(r)]['automatic_hallucination'] and not labels[key(r)]['old_i_el_positive']]
            unlabelled[name]=[dict(image_id=r['image_id'],step=r['step'],token_id=r['token_id'],p_real=probability(r))
                for r in records if key(r) not in labels and probability(r)<threshold]
    result=dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),calibration=calibration,metrics=metrics,
        additional_false=additional,unlabelled_positives=unlabelled,new_generation=0,new_feature_extraction=0,new_training=0,
        source_domain_quantile_is_not_a_target_FPR_guarantee=True,image_review='pending',advance_to_mitigation=[])
    write(OUT/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('additional_false','unlabelled_positives')},indent=2))
    print('ADDITIONAL',json.dumps(additional,indent=2))


if __name__=='__main__':run()
