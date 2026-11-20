"""Keep token-class eligibility distinct from completed-mention CHAIR labels.

The frozen v1 analysis incorrectly equated these for skateboarder/Carrots.
All191 positions remain in the primary result; this revision independently
revalidates each online token/history decision and every portable gate score.
"""

import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from features.coco_object_history import ObjectClassHistory
from features.qwen3_online_risk_f import CompactRiskFGate
from scripts.run_qwen3_signed_feedback import read,sha,write,source_groups


def analyze(root):
    from transformers import AutoTokenizer
    plan=read(root/'protocol.json')
    original_status=read(root/'status.json')
    assert original_status['status']=='failed' and 'analyze_qwen3_risk_f_gate.py' in original_status['error']
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    positions=read(REPO/'outputs/qwen3_native_gate_coverage_20261118/positions.json')
    key=lambda r:(r['image_id'],r['step'],r['token_id'])
    eligible={key(r):r for r in positions if r['filters']['absent_class']}
    assert len(eligible)==191
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(tokenizer,read(root/'object_lexicon.json')['token_ids'])
    native={};records=[];times=[]
    for shard in (0,1):
        assert read(root/f'worker{shard}.json')['status']=='complete'
        assert read(root/f'preflight.shard{shard}.json')['passed']
        records.extend(map(json.loads,(root/f'features.shard{shard}.jsonl').read_bytes().splitlines()))
        times.extend(read(path) for path in root.glob(f'timing.shard{shard}.batch*.json'))
        for group in source_groups(Path(plan['source'])/f'coco100/sadi.shard{shard}.jsonl'):
            for row in group:native[row['image_id']]=row['response_token_ids']
    indexed={key(r):r for r in records}
    assert len(indexed)==len(records) and set(eligible).issubset(indexed)
    for r in records:
        tokens=native[r['image_id']];step=r['step'];target=r['token_id']
        assert step>0 and tokens[step]==target and target not in tokens[:step]
        assert target in classes and r['canonical']==classes[target]
        prior=parser.classes(tokenizer.decode(tokens[:step],skip_special_tokens=True,clean_up_tokenization_spaces=False))
        assert classes[target] not in prior
        assert len(r['features'])==4 and np.isfinite(r['features']).all()
        assert r['positive']==(r['p_hall']>=plan['threshold'])
    gate=CompactRiskFGate(plan['gate'])
    probabilities=gate.predict([r['features'] for r in records])
    observed=np.array([r['p_hall'] for r in records])
    np.testing.assert_allclose(probabilities,observed,atol=1e-6,rtol=0)
    assert np.array_equal(probabilities>=gate.threshold,[r['positive'] for r in records])
    labelled=[];unlabelled=[];mismatches=[]
    for r in records:
        if key(r) not in eligible:
            unlabelled.append(r);continue
        label=eligible[key(r)]
        item=dict(r,mention_canonical=label['canonical'],automatic_hallucination=label['hallucinated'],
            old_i_el_positive=label['filters']['cached_probe_below_half'],old_p_real=label['p_real'],
            token_text=label['token_text'],context=label['context'])
        labelled.append(item)
        if r['canonical']!=label['canonical']:mismatches.append(item)
    assert {key(r) for r in mismatches}=={(2179,45,97982),(311553,118,3261)}
    def summarize(values):
        labels=np.array([r['automatic_hallucination'] for r in values])
        selected=np.array([r['positive'] for r in values]);old=np.array([r['old_i_el_positive'] for r in values])
        def stats(pred):
            return dict(selected=int(pred.sum()),automatic_false_selected=int((pred&labels).sum()),
                automatic_real_selected=int((pred&~labels).sum()),recall_within_eligible=float((pred&labels).sum()/labels.sum()),
                false_positive_rate_within_eligible=float((pred&~labels).sum()/(~labels).sum()))
        return dict(positions=len(values),automatic_false=int(labels.sum()),automatic_real=int((~labels).sum()),
            auroc=float(roc_auc_score(labels,[r['p_hall'] for r in values])),
            hall_aupr=float(average_precision_score(labels,[r['p_hall'] for r in values])),
            risk_F=stats(selected),old_i_el=stats(old),union_false_selected=int(((selected|old)&labels).sum()))
    primary=summarize(labelled)
    assert primary['positions']==191 and primary['automatic_false']==29
    assert primary['old_i_el']['automatic_false_selected']==11 and primary['old_i_el']['automatic_real_selected']==7
    report=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),recorded=len(records),unlabelled=len(unlabelled),
        primary=primary,exclude_two_token_prefix_mismatches=summarize([r for r in labelled if r not in mismatches]),
        threshold=plan['threshold'],no_refit_no_threshold_tuning=True,
        additional_false=[r for r in labelled if r['positive'] and r['automatic_hallucination'] and not r['old_i_el_positive']],
        lost_old_false=[r for r in labelled if not r['positive'] and r['automatic_hallucination'] and r['old_i_el_positive']],
        token_vs_completed_mention_mismatches=mismatches,
        gate_replay_max_difference=float(np.max(np.abs(probabilities-observed))),
        timing=dict(batches=len(times),images=sum(len(v['image_ids']) for v in times),
            exact_native_full_tokens=all(v['exact_native_full_tokens'] for v in times),
            gpu_wall_seconds=sum(v['wall_seconds'] for v in times),feature_seconds=sum(v['feature_seconds'] for v in times),
            static_seconds=sum(v['static_seconds'] for v in times),semantic_head_calls=sum(v['semantic_head_calls'] for v in times)),
        proceed_to_mitigation=False,image_review='pending',
        interpretation='Main191 positions unchanged. Query-token alias class determines online eligibility; completed-mention CHAIR class determines the automatic label. Prefix ambiguity is also reported separately.')
    assert report['timing']['images']==100 and report['timing']['batches']==22 and report['timing']['exact_native_full_tokens']
    write(root/'labelled_positions_v2.json',labelled)
    write(root/'unlabelled_positions_v2.json',unlabelled)
    write(root/'analysis_v2.json',report)
    write(root/'analysis_recovery.json',dict(passed=True,source=str(Path(__file__)),source_sha256=sha(Path(__file__)),
        original_failed_status_sha256=sha(root/'status.json'),protocol_sha256=sha(root/'protocol.json'),
        reason='Token prefix alias and completed mention may differ;191 primary positions and all scores preserved. All209 actual token/history decisions and frozen gate scores independently rechecked.'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
