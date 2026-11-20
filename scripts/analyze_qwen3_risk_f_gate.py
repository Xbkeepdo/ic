"""Compare frozen risk/F and old I/E_l gates on identical native positions."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scripts.run_qwen3_signed_feedback import read,sha,write


def analyze(root):
    plan=read(root/'protocol.json')
    assert read(root/'status.json')['status']=='complete'
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    positions=read(REPO/'outputs/qwen3_native_gate_coverage_20261118/positions.json')
    key=lambda r:(r['image_id'],r['step'],r['token_id'])
    eligible={key(r):r for r in positions if r['filters']['absent_class']}
    assert len(eligible)==191
    records=[];times=[]
    for shard in (0,1):
        assert read(root/f'worker{shard}.json')['status']=='complete'
        assert read(root/f'preflight.shard{shard}.json')['passed']
        records.extend(map(json.loads,(root/f'features.shard{shard}.jsonl').read_bytes().splitlines()))
        times.extend(read(path) for path in root.glob(f'timing.shard{shard}.batch*.json'))
    indexed={key(r):r for r in records}
    assert len(indexed)==len(records) and set(eligible).issubset(indexed)
    values=[];unlabelled=[]
    for record in records:
        assert len(record['features'])==4 and np.isfinite(record['features']).all()
        assert record['positive']==(record['p_hall']>=plan['threshold'])
        if key(record) in eligible:
            label=eligible[key(record)]
            assert record['canonical']==label['canonical']
            values.append(dict(record,automatic_hallucination=label['hallucinated'],
                old_i_el_positive=label['filters']['cached_probe_below_half'],
                old_p_real=label['p_real'],token_text=label['token_text'],context=label['context']))
        else:unlabelled.append(record)
    labels=np.array([r['automatic_hallucination'] for r in values])
    assert labels.sum()==29
    selected=np.array([r['positive'] for r in values]);old=np.array([r['old_i_el_positive'] for r in values])
    scores=np.array([r['p_hall'] for r in values])
    def stats(pred):
        return dict(selected=int(pred.sum()),automatic_false_selected=int((pred&labels).sum()),
            automatic_real_selected=int((pred&~labels).sum()),
            recall_within_eligible=float((pred&labels).sum()/labels.sum()),
            false_positive_rate_within_eligible=float((pred&~labels).sum()/(~labels).sum()))
    report=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),recorded=len(records),
        eligible_labelled=len(values),automatic_false=29,automatic_real=162,unlabelled=len(unlabelled),
        threshold=plan['threshold'],no_refit_no_threshold_tuning=True,
        scores=dict(auroc=float(roc_auc_score(labels,scores)),
                    hall_aupr=float(average_precision_score(labels,scores))),
        risk_F=stats(selected),old_i_el=stats(old),
        additional_false=[r for r in values if r['positive'] and r['automatic_hallucination'] and not r['old_i_el_positive']],
        lost_old_false=[r for r in values if not r['positive'] and r['automatic_hallucination'] and r['old_i_el_positive']],
        union_false_selected=int(((selected|old)&labels).sum()),
        timing=dict(batches=len(times),images=sum(len(v['image_ids']) for v in times),
            exact_native_full_tokens=all(v['exact_native_full_tokens'] for v in times),
            gpu_wall_seconds=sum(v['wall_seconds'] for v in times),
            feature_seconds=sum(v['feature_seconds'] for v in times),
            static_seconds=sum(v['static_seconds'] for v in times),
            semantic_head_calls=sum(v['semantic_head_calls'] for v in times)),
        proceed_to_mitigation=False,image_review='pending',
        interpretation='Automatic CHAIR labels on191 eligible native positions only, not altered trajectories or all words.')
    assert report['old_i_el']['automatic_false_selected']==11 and report['old_i_el']['automatic_real_selected']==7
    assert report['timing']['images']==100 and report['timing']['batches']==22 and report['timing']['exact_native_full_tokens']
    write(root/'labelled_positions.json',values)
    write(root/'unlabelled_positions.json',unlabelled)
    write(root/'analysis.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
