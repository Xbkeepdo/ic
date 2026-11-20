"""Evaluate original SAM-V2 on already-read native/local-pre SADI features."""

import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
from features.qwen3_risk_i_el_probe import FullRiskIElProbe
from scripts.trace_qwen3_full_risk_f import OUT as SOURCE
from scripts.verify_qwen3_full_risk_f import TRAIN
from scripts.run_qwen3_signed_feedback import read,sha,write

OUT=SOURCE/'cached_sam_v2'
MODEL=TRAIN/'risk+I+E_l/seed43'


def run():
    torch.set_num_threads(1)
    assert read(SOURCE/'status.json')['status']=='complete' and read(SOURCE/'analysis.json')['passed']
    assert not OUT.exists()
    files=[Path(__file__),REPO/'features/qwen3_risk_i_el_probe.py',REPO/'features/qwen3_online_i_el.py',
        SOURCE/'protocol.json',SOURCE/'analysis.json',SOURCE/'labelled_positions.json',SOURCE/'unlabelled_positions.json',
        MODEL/'model.pt',MODEL/'input_scaler.npz',TRAIN/'compact_features.npz',
        *(SOURCE/f'features.shard{i}.jsonl' for i in (0,1))]
    hashes={str(p):sha(p) for p in files}
    assert hashes[str(MODEL/'model.pt')]=='8ddd3f9edbc5a7cf8d0d570569562561c5b3eb5c2723601b52ab552e0dbd9760'
    assert hashes[str(MODEL/'input_scaler.npz')]=='d550bb0f4b51fa8734d6550e1d44b22316a17ace6e8d255446bc6a43c8d24f55'
    OUT.mkdir()
    write(OUT/'protocol.json',dict(source=str(SOURCE),fixed_at=time.time(),model=str(MODEL),source_sha256=hashes,
        input_order='risk0..35,I0..35,E_l0..35',threshold_p_real=.5,
        variants=['native','prelocal'],method='Frozen original108-input SAM-V2; no fit, no GPU model call, no new threshold.',
        scope='Previously chosen209 native positions;191 automatic labels,18 unlabelled retained.',
        prospective_screen='At least one visually confirmed unsupported object missed by old I/El, <=14 automatically-real positives on191 positions. No retuning.'))
    model=FullRiskIElProbe(MODEL).eval()
    with np.load(TRAIN/'compact_features.npz',allow_pickle=False) as data,torch.inference_mode():
        mask=data['test_mask'];features=np.concatenate([data[k][mask] for k in ('risk','I','E_l')],axis=1)
        probabilities=model(torch.from_numpy(features)).numpy()
        auc=float(roc_auc_score(data['label'][mask],probabilities))
        assert abs(auc-.9110604392396301)<1e-12,auc
        write(OUT/'cpu_verification.json',dict(passed=True,old_test_targets=2367,real_positive_auroc=auc,
            input_order=['risk0..35','I0..35','E_l0..35'],checkpoint_sha256=sha(MODEL/'model.pt')))
    records=[r for shard in (0,1) for r in map(json.loads,(SOURCE/f'features.shard{shard}.jsonl').read_bytes().splitlines())]
    key=lambda r:(r['image_id'],r['step'],r['token_id'])
    labels={key(r):r for r in read(SOURCE/'labelled_positions.json')}
    assert len(records)==209 and len(labels)==191
    scored=[dict(image_id=r['image_id'],step=r['step'],token_id=r['token_id'],canonical=r['canonical']) for r in records]
    metrics={};additional={};unlabelled={};checks={}
    for name in ('native','pre'):
        vectors=np.concatenate([np.array([r[f'{name}_features'][:36] for r in records],np.float32),
            np.array([r[f'{name}_I'] for r in records],np.float32),np.array([r['E_l'] for r in records],np.float32)],axis=1)
        with torch.inference_mode():probabilities=model(torch.from_numpy(vectors)).numpy()
        p={name:buffer.detach().cpu().numpy() for name,buffer in model.named_buffers()}
        standardized=((vectors.astype(p['mean'].dtype)-p['mean'])/p['scale']).astype(np.float32)
        manual=expit(np.maximum(0,standardized@p['weight0'].T+p['bias0'])@p['weight1'].T+p['bias1'])[:,0]
        np.testing.assert_allclose(probabilities,manual,rtol=0,atol=5e-6)
        assert np.array_equal(probabilities<.5,manual<.5)
        checks[name]=float(np.max(np.abs(probabilities-manual)))
        for record,probability in zip(scored,probabilities,strict=True):record[f'{name}_p_real']=float(probability)
        labelled=[dict(r,**{k:labels[key(r)][k] for k in ('automatic_hallucination','old_i_el_positive','old_p_real','mention_canonical','token_text','context')})
                  for r in scored if key(r) in labels]
        y=np.array([r['automatic_hallucination'] for r in labelled]);probs=np.array([r[f'{name}_p_real'] for r in labelled]);selected=probs<.5
        metrics[name]=dict(auroc=float(roc_auc_score(1-y,probs)),real_aupr=float(average_precision_score(1-y,probs)),
            hall_aupr=float(average_precision_score(y,1-probs)),selected=int(selected.sum()),automatic_false_selected=int((selected&y).sum()),
            automatic_real_selected=int((selected&~y).sum()),recall=float((selected&y).sum()/29),real_fpr=float((selected&~y).sum()/162))
        additional[name]=[r for r in labelled if r[f'{name}_p_real']<.5 and r['automatic_hallucination'] and not r['old_i_el_positive']]
        unlabelled[name]=[r for r in scored if key(r) not in labels and r[f'{name}_p_real']<.5]
    write(OUT/'scores.json',scored)
    report=dict(passed=True,protocol_sha256=sha(OUT/'protocol.json'),recorded=209,labelled=191,automatic_false=29,
        metrics=metrics,additional_false=additional,unlabelled_positives=unlabelled,independent_numpy_max_errors=checks,
        image_review='pending',advance_to_mitigation=[],new_gpu_model_calls=0,new_feature_extraction=0,no_training=True)
    write(OUT/'analysis.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('additional_false','unlabelled_positives')},indent=2))
    print('ADDITIONAL',json.dumps(additional,indent=2))


if __name__=='__main__':run()
