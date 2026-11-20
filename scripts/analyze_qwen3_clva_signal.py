"""Audit CLVA/SAR feedback using its direct-commit semantics, then captions."""

import argparse
import json
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
        REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',REPO/'coco-labeling/coco_chair.py']
    write(root/'analysis_request.json',dict(fixed_at=time.time(),protocol_sha256=sha(root/'protocol.json'),
        source_sha256={str(p):sha(p) for p in files},before_inspecting_caption_metrics=True,
        primary='Full cohort sentence Cs/Ci, false counts, mean image recall/F1;5000 paired bootstrap seed43. Legacy secondary.',
        refs=['sadi','svar','clva','clva_fixed'],
        review='Every changed caption and all fixed/introduced hallucinated images, false-class/count changes and recall losses; physical gains may be direct or indirect.',
        semantics='Committed feedback rows are execution counts, not successful corrections; no posterior probability-decrease acceptance condition.',
        followup='10 is implementation screening; proceed to original100 after correctness unless output degenerates. Independent500 only after useful full100 effects and review.'))


def check(root):
    plan=read(root/'protocol.json');request=read(root/'analysis_request.json')
    assert request['protocol_sha256']==sha(root/'protocol.json')
    for n,v in {**plan['source_sha256'],**request['source_sha256']}.items():assert sha(REPO/n)==v,n
    return plan,request


def run(root):
    plan,request=check(root)
    assert read(root/'pipeline_complete.json')['passed'] and not (root/'analysis_provenance.json').exists()
    assert all(read(root/f'worker{s}.json')['passed'] for s in range(plan['shards']))
    ids=read(root/'cohort.json')[plan['cohort']];allowed=set(read(root/'object_lexicon.json')['token_ids'])
    raw={};grouping={};activity={}
    for name in plan['variants']:
        raw[name]={};grouping[name]=[];batches=[]
        for shard in range(plan['shards']):
            groups=source_groups(root/f'{plan["cohort"]}/{name}.shard{shard}.jsonl')
            grouping[name].append([(tuple(r['image_id'] for r in g),g[0]['visual_token_count']) for g in groups])
            for group in groups:
                batches.append(group[0]['sar_ems_statistics'])
                for row in group:
                    assert row['image_id'] not in raw[name];raw[name][row['image_id']]=row
        assert set(raw[name])==set(ids) and grouping[name]==grouping['sadi']
        if name in ['sadi','svar']:continue
        events=[]
        for image in ids:
            row=raw[name][image];tokens=row['response_token_ids'];s=row['sar_ems_statistics']
            assert s['mode']==name and not s['diagnostic_disabled'] and not s['diagnostic_zero_guidance'] and not s['diagnostic_reject_all']
            assert s['unchanged_base_layers']==list(range(5,19)) and s['action_layers']==list(range(19,36))
            assert s['source_positive_layer']==16 and s['source_negative_layer']==0
            assert s['raw_clva_layer_steps']%17==0 and s['no_trained_detector'] and s['no_history_repeat_filter']
            for e in s['full_events']:
                assert e['mode']==name and e['token'] in allowed and 0<=e['D_native']<=1
                assert e['D_applied']==e['D_native'] and abs(e['alpha']-14*(1+e['D_applied']))<3e-6
                if name=='clva_fixed':assert e['D_native']==.5
                if name!='clva_sar':assert e['positive_mask_changed_positions']==0
                assert e['committed'] and e['committed_layers']==list(range(19,36))
                assert tokens[e['step']]==e['emitted_token']==e['proposed_token']
                assert e['token_changed']==(e['token']!=e['emitted_token'])
                events.append(e)
            assert s['accepted']==len(s['full_events'])
        if name=='clva':assert not events
        activity[name]=dict(feedback_rows=len(events),immediate_token_changes=sum(e['token_changed'] for e in events),
            changed_vs_sadi=sum(raw[name][i]['response_token_ids']!=raw['sadi'][i]['response_token_ids'] for i in ids),
            raw_anchor_query_rows=sum(raw[name][i]['sar_ems_statistics']['raw_clva_layer_steps']//17 for i in ids),
            extra_tail_passes=sum(s['feedback_calls_in_batch'] for s in batches),
            original_encoder_calls=sum(s['encoder_calls_in_batch'] for s in batches),vision_cache_hits=sum(s['vision_cache_hits'] for s in batches),
            static_head_calls=sum(s['static_head_calls'] for s in batches),static_hits=sum(s['shared_static_hits'] for s in batches),
            mean_D=sum(e['D_native'] for e in events)/max(1,len(events)),
            mean_alpha=sum(e['alpha'] for e in events)/max(1,len(events)),
            changed_positive_masks=sum(e['positive_mask_changed_positions']>0 for e in events),
            seconds=sum(raw[name][i]['seconds'] for i in ids))
    write(root/'mechanism_audit.json',dict(passed=True,activity=activity,semantics='Execution counts, not improvement counts.'))
    for mode in ['sentence','legacy']:
        with (root/f'evaluate_{mode}.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_qwen3_region_caption_chair.py','--generation-root',str(root),
                '--cohort',plan['cohort'],'--limit',str(plan['images']),'--variants',*plan['variants'],'--evaluator',mode],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'run_status.json',dict(status='complete',phase='generation_and_evaluation_complete',finished=time.time()))
    with (root/'independent_audit.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/audit_qwen3_caption_with_native_cache_v2.py',str(root)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=read(root/'independent_summary.json');assert summary['passed']
    path,=(root/f'evaluations/sentence_v1/{plan["cohort"]}').glob('chair_per_image*.jsonl')
    scored=[json.loads(line) for line in path.read_bytes().splitlines()];assert [r['image_id'] for r in scored]==ids
    fields=['chair_s','chair_i','hallucinated_mentions','hallucinated_objects','object_mentions','object_recall','object_f1']
    cases={};review={}
    for name in plan['candidates']:
        cases[name]={};queues={ref:dict(fixed=[],introduced=[],false_mentions_changed=[],recall_losses=[]) for ref in ['sadi','clva']}
        for image,score in zip(ids,scored,strict=True):
            if any(raw[name][image]['response_token_ids']!=raw[ref][image]['response_token_ids'] for ref in ['sadi','clva']):
                cases[name][image]=dict(texts={n:raw[n][image]['generated_text'] for n in plan['variants']},
                    scores={n:{k:score['methods'][n][k] for k in fields} for n in plan['variants']},
                    feedback_events=raw[name][image]['sar_ems_statistics']['full_events'])
            for ref,q in queues.items():
                a,b=score['methods'][ref],score['methods'][name]
                if a['chair_s']>b['chair_s']:q['fixed'].append(image)
                if a['chair_s']<b['chair_s']:q['introduced'].append(image)
                if a['hallucinated_mentions']!=b['hallucinated_mentions']:q['false_mentions_changed'].append(image)
                if a['object_recall']>b['object_recall']:q['recall_losses'].append(image)
        review[name]=queues
    write(root/'changed_cases.json',cases);write(root/'image_review_queue.json',review);write(root/'activity.json',activity)
    check(root)
    write(root/'analysis_provenance.json',dict(passed=True,analysis_request_sha256=sha(root/'analysis_request.json'),
        protocol_sha256=sha(root/'protocol.json'),finished=time.time(),image_review='pending',direct_advance_to500=False))
    print(json.dumps(dict(activity=activity,sentence={n:{k:r[k] for k in ['chair_s','chair_i','object_recall','object_f1','audit_false_mentions','audit_mentions']}
        for n,r in summary['eval']['sentence'].items()},review=review),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root)
