"""Independently replay full detector scores and audit paired readout transfer."""

import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from features.qwen3_online_i_el import FullIElProbe
from scripts.run_qwen3_signed_feedback import read,sha,write


def analyze(root):
    torch.set_num_threads(1)
    plan=read(root/'protocol.json');source=Path(plan['source'])
    assert read(root/'status.json')['status']=='complete'
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    key=lambda r:(r['image_id'],r['step'],r['token_id'])
    old={key(r):r for shard in (0,1) for r in map(json.loads,(source/f'features.shard{shard}.jsonl').read_bytes().splitlines())}
    labelled={key(r):r for r in read(source/'labelled_positions_v2.json')}
    records=[];times=[]
    for shard in (0,1):
        assert read(root/f'worker{shard}.json')['status']=='complete'
        assert read(root/f'preflight.shard{shard}.json')['passed']
        records.extend(map(json.loads,(root/f'features.shard{shard}.jsonl').read_bytes().splitlines()))
        times.extend(read(p) for p in root.glob(f'timing.shard{shard}.batch*.json'))
    assert len(records)==len({key(r) for r in records})==len(old)==209
    assert {key(r) for r in records}==set(old) and len(labelled)==191
    for r in records:
        assert r['canonical']==old[key(r)]['canonical']
        assert len(r['native_features'])==len(r['pre_features'])==72
        assert len(r['native_I'])==len(r['pre_I'])==len(r['E_l'])==36
        outside=[i for i in range(72) if i%36 not in range(5,19)]
        np.testing.assert_array_equal(np.array(r['native_features'])[outside],np.array(r['pre_features'])[outside])
        np.testing.assert_array_equal(np.array(r['native_features'])[[27,63,28,64]],old[key(r)]['features'])
        assert r['compact_p_hall']==old[key(r)]['p_hall']
    score_checks={}
    with torch.inference_mode():
        risk=FullIElProbe(plan['full_model']).eval();iel=FullIElProbe(plan['i_el_control']).eval()
        for name in ('native','pre'):
            features=np.asarray([r[f'{name}_features'] for r in records],np.float32)
            expected=risk(torch.from_numpy(features)).numpy()
            observed=np.asarray([r[f'{name}_p_real'] for r in records])
            np.testing.assert_allclose(expected,observed,atol=1e-5,rtol=0)
            assert np.array_equal(expected<.5,observed<.5)
            score_checks['risk_'+name]=float(np.max(np.abs(expected-observed)))
            features=np.concatenate([np.array([r[f'{name}_I'] for r in records],np.float32),
                                     np.array([r['E_l'] for r in records],np.float32)],axis=1)
            expected=iel(torch.from_numpy(features)).numpy()
            observed=np.asarray([r[f'{name}_i_el_p_real'] for r in records])
            np.testing.assert_allclose(expected,observed,atol=1e-5,rtol=0)
            assert np.array_equal(expected<.5,observed<.5)
            score_checks['iel_'+name]=float(np.max(np.abs(expected-observed)))
    joined=[dict(r,automatic_hallucination=labelled[key(r)]['automatic_hallucination'],
        old_i_el_positive=labelled[key(r)]['old_i_el_positive'],old_p_real=labelled[key(r)]['old_p_real'],
        mention_canonical=labelled[key(r)]['mention_canonical'],token_text=labelled[key(r)]['token_text'],
        context=labelled[key(r)]['context']) for r in records if key(r) in labelled]
    unlabelled=[r for r in records if key(r) not in labelled]
    difference=np.abs(np.array([r['pre_i_el_p_real'] for r in joined])-np.array([r['old_p_real'] for r in joined]))
    assert difference.max()<1e-3
    assert all((r['pre_i_el_p_real']<.5)==r['old_i_el_positive'] for r in joined)
    labels=np.array([r['automatic_hallucination'] for r in joined]);assert labels.sum()==29
    primary={};additional={};lost={}
    for name,column in [('risk_native','native_p_real'),('risk_prelocal','pre_p_real'),
                         ('iel_native','native_i_el_p_real'),('iel_prelocal','pre_i_el_p_real'),('old_iel','old_p_real')]:
        probabilities=np.array([r[column] for r in joined]);selected=probabilities<.5
        primary[name]=dict(auroc=float(roc_auc_score(1-labels,probabilities)),
            real_aupr=float(average_precision_score(1-labels,probabilities)),
            hall_aupr=float(average_precision_score(labels,1-probabilities)),
            selected=int(selected.sum()),automatic_false_selected=int((selected&labels).sum()),
            automatic_real_selected=int((selected&~labels).sum()),
            recall_within_eligible=float((selected&labels).sum()/29),real_fpr=float((selected&~labels).sum()/162))
        if name.startswith('risk_'):
            additional[name]=[r for r in joined if r[column]<.5 and r['automatic_hallucination'] and not r['old_i_el_positive']]
            lost[name]=[r for r in joined if r[column]>=.5 and r['automatic_hallucination'] and r['old_i_el_positive']]
    report=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),recorded=209,eligible_labelled=191,
        automatic_false=29,automatic_real=162,unlabelled=18,primary=primary,
        additional_false=additional,lost_old_false=lost,score_replay_max_errors=score_checks,
        pre_iel_vs_old_cached_max_probability_difference=float(difference.max()),
        pre_iel_vs_old_all191_gate_flags_equal=True,
        timing=dict(batches=len(times),images=sum(len(v['image_ids']) for v in times),
            gpu_wall_seconds=sum(v['wall_seconds'] for v in times),feature_seconds=sum(v['feature_seconds'] for v in times),
            static_seconds=sum(v['static_seconds'] for v in times),semantic_head_calls=sum(v['semantic_head_calls'] for v in times),
            risk_solver_calls=sum(v['risk_solver_calls'] for v in times),
            maximum_memory_allocated=max(v['maximum_memory_allocated'] for v in times),
            maximum_E_l_cache_relative_error=max(v['E_l_max_relative'] for v in times)),
        image_review='pending',advance_to_mitigation=[],interpretation='Original frozen models, no refit. Automatic labels and token-prefix ambiguity retained. Prelocal is not an unmodified base model.')
    assert report['timing']['batches']==22 and report['timing']['images']==100
    assert report['timing']['risk_solver_calls']==50*209
    assert all(v['exact_native_full_tokens'] and v['compact_features_exact'] for v in times)
    write(root/'labelled_positions.json',joined);write(root/'unlabelled_positions.json',unlabelled)
    write(root/'analysis.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('additional_false','lost_old_false')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
