"""Audit original next-layer expansion with exact cached controls."""

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
        primary='Full100 sentence Cs/Ci/false counts/mean image recall/F1. Legacy secondary; all-pair5000 paired bootstrap seed43.',
        references=['sadi','sadi_sar_ems_next_19_20','svar'],
        secondary='Frozen generic quality analysis, nominal strength versus actual activity, source target versus emitted-token matches.',
        review='All changed false-class counts, fixed/introduced images, recall losses, repetition concentrations and original photographs before claims of physical improvement.',
        advance='Only a useful full100 candidate with real gains beyond old-next, without repetition/alias explanation, enters reserved49-group500. No validation tuning.'))


def check(root):
    plan=read(root/'protocol.json');request=read(root/'analysis_request.json')
    assert request['protocol_sha256']==sha(root/'protocol.json')
    for name,wanted in {**plan['source_sha256'],**request['source_sha256']}.items():assert sha(REPO/name)==wanted,name
    return plan,request


def run(root):
    import json
    plan,request=check(root)
    assert read(root/'pipeline_complete.json')['passed'] and not (root/'analysis_provenance.json').exists()
    assert all(read(root/f'preflight{s}.json')['passed'] and read(root/f'worker{s}.json')['passed'] for s in [0,1])
    ids=read(root/'cohort.json')['coco100'];nouns=set(read(root/'wordnet_nouns.json')['token_ids'])
    raw={};grouping={};activity={}
    for name in plan['variants']:
        raw[name]={};grouping[name]=[];batches=[]
        for shard in range(plan['shards']):
            groups=source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            grouping[name].append([(tuple(r['image_id'] for r in g),g[0]['visual_token_count']) for g in groups])
            batches.extend(g[0]['sar_ems_statistics'] for g in groups)
            for group in groups:
                for row in group:
                    assert row['image_id'] not in raw[name];raw[name][row['image_id']]=row
        assert set(raw[name])==set(ids) and grouping[name]==grouping['sadi']
        if name not in plan['candidates']:continue
        events=[]
        for image in ids:
            row=raw[name][image];s=row['sar_ems_statistics'];tokens=row['response_token_ids']
            assert s['source_layers']==list(range(18,24)) and s['target_layers']==list(range(19,25))
            assert s['unchanged_base_layers']==list(range(5,19)) and s['strength']==plan['strengths'][name]
            assert s['compact_semantic_cache'] and not s['disabled'] and not s['diagnostic_zero_budget'] and not s['diagnostic_force_guidance']
            assert s['local_recompute_events']==0 and s['all_applied_during_original_single_forward']
            assert s['guided_attention_calls']==len(s['full_events'])
            assert s['executed_events']==len(s['full_events'])+s['unreported_terminal_events']
            for e in s['full_events']:
                assert 0<=e['step']<len(tokens) and e['token'] in nouns
                assert e['target_layer']==e['source_layer']+1 and e['target_layer'] in range(19,25)
                assert e['gain']>0 and 0<=e['drift']<=1 and 0<e['budget_max']<=s['strength']+1e-6
            events.extend(s['full_events'])
        activity[name]=dict(caption_events=len(events),terminal_events=sum(r['sar_ems_statistics']['unreported_terminal_events'] for r in raw[name].values()),
            per_target={i:sum(e['target_layer']==i for e in events) for i in range(19,25)},
            source_target_matches_emitted=sum(raw[name][image]['response_token_ids'][e['step']]==e['token']
                for image in ids for e in raw[name][image]['sar_ems_statistics']['full_events']),
            new_term_builds=sum(s['new_term_builds'] for s in batches),shared_term_hits=sum(s['shared_term_hits'] for s in batches),
            formal_encoder_calls=sum(s['original_encoder_calls'] for s in batches),formal_vision_hits=sum(s['vision_cache_hits'] for s in batches),
            changed_vs_sadi=sum(raw[name][i]['response_token_ids']!=raw['sadi'][i]['response_token_ids'] for i in ids),
            changed_vs_next2=sum(raw[name][i]['response_token_ids']!=raw['sadi_sar_ems_next_19_20'][i]['response_token_ids'] for i in ids))
    for entry in plan['other_controls']:
        source=Path(entry['source']);assert sha(source)==entry['source_sha256']
        assert (root/f'coco100/{entry["variant"]}.shard{entry["shard"]}.jsonl').read_bytes()==source.read_bytes()
    write(root/'mechanism_audit.json',dict(passed=True,activity=activity,nominal_matching_only=True,
        provisional_source_tokens_are_not_final_targets=True,controls_preserved=True))
    for mode in ['sentence','legacy']:
        with (root/f'evaluate_{mode}.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_qwen3_region_caption_chair.py','--generation-root',str(root),
                '--cohort','coco100','--limit','100','--variants',*plan['variants'],'--evaluator',mode],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'run_status.json',dict(status='complete',phase='generation_and_evaluation_complete',finished=time.time()))
    with (root/'independent_audit.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/audit_qwen3_caption_with_native_cache_v2.py',str(root)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(root/'independent_summary.json')['passed'];check(root)
    write(root/'analysis_provenance.json',dict(passed=True,finished=time.time(),protocol_sha256=sha(root/'protocol.json'),
        request_sha256=sha(root/'analysis_request.json'),physical_review='pending',direct_advance_to500=False))
    print(json.dumps(dict(activity=activity),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare(args.root) if args.prepare else run(args.root)
