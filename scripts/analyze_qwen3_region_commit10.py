"""Audit retained same-token proposals, frozen controls, then full captions."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.qwen3_caption_groups_with_cache import source_groups


def prepare(root):
    assert not (root/'analysis_request.json').exists() and not (root/'evaluations').exists()
    files=[Path(__file__),REPO/'scripts/evaluate_qwen3_region_caption_chair.py',REPO/'scripts/evaluate_caption_chair.py',
        REPO/'scripts/evaluate_generation_time_evidence_drift_llava.py',REPO/'scripts/evaluate_qwen3_next200_sentence_tokenized.py',
        REPO/'coco-labeling/coco_chair.py',REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py']
    write(root/'analysis_request.json',dict(fixed_at=time.time(),protocol_sha256=sha(root/'protocol.json'),
        source_sha256={str(p):sha(p) for p in files},before_inspecting_caption_metrics=True,
        primary='Full cohort sentence Cs/Ci/false counts/recall/F1, paired5000 seed43; legacy secondary.',
        quality='Frozen generic quality analysis, all-arm repetitions and denominator diagnostics; full captions and images required.',
        advancement='No500 decision from old10. If physically useful complete-caption changes beyond unverified/shifted, original100 next; no blanket demand for immediate word correction.'))


def check(root):
    plan=read(root/'protocol.json');request=read(root/'analysis_request.json')
    assert request['protocol_sha256']==sha(root/'protocol.json')
    for name,wanted in {**plan['source_sha256'],**request['source_sha256']}.items():assert sha(REPO/name)==wanted,name
    return plan,request


def run(root):
    from features.qwen3_region_commit import should_commit
    from features.qwen3_region_guidance import prefix_sha
    plan,request=check(root)
    assert read(root/'pipeline_complete.json')['passed'] and not (root/'analysis_provenance.json').exists()
    assert all(read(root/f'worker{s}.json')['passed'] for s in range(plan['shards']))
    ids=read(root/'cohort.json')[plan['cohort']];raw={};groups={};activity={}
    for name in plan['variants']:
        raw[name]={};groups[name]=[];batch_stats=[]
        for shard in range(plan['shards']):
            batches=source_groups(root/f'coco10/{name}.shard{shard}.jsonl')
            groups[name].append([(tuple(r['image_id'] for r in g),g[0]['visual_token_count']) for g in batches])
            batch_stats.extend(g[0]['sar_ems_statistics'] for g in batches)
            for batch in batches:
                for row in batch:
                    assert row['image_id'] not in raw[name];raw[name][row['image_id']]=row
        assert set(raw[name])==set(ids) and groups[name]==groups['sadi']
        if name not in plan['candidates']:continue
        events=[]
        for image in ids:
            row=raw[name][image];tokens=row['response_token_ids'];s=row['sar_ems_statistics']
            assert s['commit_mode']==name and s['policy']=='unverified'
            assert not s['disabled'] and not s['diagnostic_zero_strength'] and not s['diagnostic_reject_all']
            assert s['unchanged_base_layers']==list(range(5,19)) and s['intervention_layers']==list(range(20,36))
            assert s['crop_encoder_calls_in_batch']==s['fine_prefill_calls_in_batch']==0
            gates={(g['step'],g['token']):g for g in s['gates']}
            for e in s['full_events']:
                assert e['commit_mode']==name and 0<=e['drift']<=1 and e['strength']==5*e['drift']
                assert e['reliability']==dict(reliability_deficit=1.,skipped=True)
                assert e['prefix_sha256']==prefix_sha(tokens[:e['step']])
                assert gates[e['step'],e['token']]['positive']
                assert gates[e['step'],e['token']]['p_real']<.418672114610672
                expected=should_commit(name,e['before_logp'],e['after_logp'],e['token'],e['proposed_token'])
                assert e['accepted']==expected and e['committed_layers']==(list(range(20,36)) if expected else [])
                assert tokens[e['step']]==(e['proposed_token'] if expected else e['token'])
                assert e['same_token_commit']==(expected and e['proposed_token']==e['token'])
                assert e['visual_mass_relative_error_fp32']<1e-5
                if name!='shifted':assert e['rect']==e['applied_rect']
                events.append(e)
            assert s['accepted']==sum(e['accepted'] for e in s['full_events'])
        activity[name]=dict(proposals=len(events),committed=sum(e['accepted'] for e in events),
            same_token_commits=sum(e['same_token_commit'] for e in events),
            nonzero_same_token_commits=sum(e['same_token_commit'] and e['proposed_current_kv_max_difference']>0 for e in events),
            changed_vs_sadi=sum(raw[name][i]['response_token_ids']!=raw['sadi'][i]['response_token_ids'] for i in ids),
            changed_vs_old_unverified=sum(raw[name][i]['response_token_ids']!=raw['unverified'][i]['response_token_ids'] for i in ids),
            encoder_calls=sum(s['actual_encoder_calls_in_batch'] for s in batch_stats),
            seconds=sum(raw[name][i]['seconds'] for i in ids))
    for entry in plan['historical_controls']:
        source=Path(entry['source']);assert sha(source)==entry['source_sha256']
        expected,=[g for g in source_groups(source) if [r['image_id'] for r in g]==entry['ids']]
        actual,=source_groups(root/f'coco10/unverified.shard{entry["shard"]}.jsonl')
        assert expected==actual and sha(root/f'coco10/unverified.shard{entry["shard"]}.jsonl')==entry['materialized_sha256']
    write(root/'mechanism_audit.json',dict(passed=True,activity=activity,historical_controls_preserved=True))
    for mode in ['sentence','legacy']:
        with (root/f'evaluate_{mode}.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_qwen3_region_caption_chair.py','--generation-root',str(root),
                '--cohort','coco10','--limit','10','--variants',*plan['variants'],'--evaluator',mode],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'run_status.json',dict(status='complete',phase='generation_and_evaluation_complete',finished=time.time()))
    with (root/'independent_audit.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/audit_qwen3_caption_with_native_cache_v2.py',str(root)],
            cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(root/'independent_summary.json')['passed'];check(root)
    write(root/'analysis_provenance.json',dict(passed=True,finished=time.time(),protocol_sha256=sha(root/'protocol.json'),
        request_sha256=sha(root/'analysis_request.json'),physical_review='pending',direct_advance_to500=False))
    print(activity)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare(args.root) if args.prepare else run(args.root)
