"""Generic caption and mechanism audit for support-guided attention variants."""

import argparse
import hashlib
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
    paths=[Path(__file__),REPO/'scripts/evaluate_qwen3_region_caption_chair.py',REPO/'scripts/evaluate_caption_chair.py',
        REPO/'scripts/evaluate_generation_time_evidence_drift_llava.py',REPO/'scripts/evaluate_qwen3_next200_sentence_tokenized.py',
        REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',REPO/'coco-labeling/coco_chair.py']
    write(root/'analysis_request.json',dict(fixed_at=time.time(),protocol_sha256=sha(root/'protocol.json'),
        source_sha256={str(p):sha(p) for p in paths},before_inspecting_new_caption_metrics=True,
        metrics='Sentence primary, legacy secondary; unchanged independent numeric audit and5000 paired bootstrap seed43.',
        review='Every changed caption and every Cs fix/introduction, false-mention change or recall loss; inspect originals before physical correction claim.',
        decision='Implementation/development cohort only. Never advance directly from10 to500; useful whole-caption effects must pass original100 first.'))


def check(root):
    plan=read(root/'protocol.json');request=read(root/'analysis_request.json')
    assert request['protocol_sha256']==sha(root/'protocol.json')
    for n,v in {**plan['source_sha256'],**request['source_sha256']}.items():assert sha(REPO/n)==v,n
    return plan,request


def run(root):
    import numpy as np
    plan,request=check(root);assert read(root/'pipeline_complete.json')['passed']
    assert all(read(root/f'worker{s}.json')['passed'] for s in range(plan['shards']))
    assert not (root/'analysis_provenance.json').exists()
    ids=read(root/'cohort.json')[plan['cohort']];raw={};activity={};grouping={}
    for name in plan['variants']:
        raw[name]={};grouping[name]=[]
        for shard in range(plan['shards']):
            groups=source_groups(root/f'{plan["cohort"]}/{name}.shard{shard}.jsonl')
            grouping[name].append([(tuple(r['image_id'] for r in g),g[0]['visual_token_count']) for g in groups])
            for group in groups:
                for r in group:assert r['image_id'] not in raw[name];raw[name][r['image_id']]=r
        assert set(raw[name])==set(ids) and grouping[name]==grouping['sadi']
    activity_names=[n for n in plan['variants'] if n not in ['sadi','svar']]
    for name in activity_names:
        events=[];alignments=[]
        for image in ids:
            r=raw[name][image];s=r['sar_ems_statistics'];tokens=r['response_token_ids']
            assert not s['disabled'] and not s['diagnostic_zero_strength'] and not s['diagnostic_reject_all']
            assert s['all_proposals_evaluated_before_commits'] and s['crop_encoder_calls_in_batch']==s['fine_prefill_calls_in_batch']==0
            for gate in s['gates']:
                assert gate['positive']==(gate['p_real']<.418672114610672)
                assert gate['prefix_sha256']==hashlib.sha256(np.asarray(tokens[:gate['step']],dtype=np.int64).tobytes()).hexdigest()
            for e in s['full_events']:
                c=e['rival_choice'];target,rival=c['target'],c['rival'];lift=c['lift']
                candidates=[x for x in lift if x!=target]
                assert rival==max(candidates,key=lambda x:(c['native_class_mass'][x]*lift[x],x))
                confidence=max(0.,lift[rival]-lift[target])/max(1e-30,lift[rival]+lift[target])
                assert abs(e['support_confidence']-confidence)<1e-10
                assert abs(e['strength']-5*e['drift']*confidence)<1e-10
                assert e['visual_mass_relative_error_fp32']<1e-5
                assert tokens[e['step']]==(e['proposed_token'] if e['accepted'] else e['token'])
                assert e['accepted']==(e['proposed_token']!=e['token'] and e['after_logp']<e['before_logp'])
                if e['accepted']:assert e['committed_layers']==list(range(20,36)) and e['proposed_current_kv_max_difference']>0
                events.append(e)
                if 'ffn_alignment' in e:
                    a=e['ffn_alignment'];assert a['attempts']<=4 and a['autograd_calls']==a['extra_mlp_or_model_calls']==0
                    if a['applied']:assert a['after']<a['before'] and a['relative_norm_change']<.01
                    alignments.append(a)
        activity[name]=dict(proposals=len(events),accepted=sum(e['accepted'] for e in events),
            changed_vs_sadi=sum(raw[name][i]['response_token_ids']!=raw['sadi'][i]['response_token_ids'] for i in ids),
            ffn_alignment_attempts=len(alignments),ffn_applied=sum(a['applied'] for a in alignments),
            max_ffn_relative_norm_change=max([a['relative_norm_change'] for a in alignments],default=0),
            seconds=sum(raw[name][i]['seconds'] for i in ids))
    write(root/'mechanism_audit.json',dict(passed=True,activity=activity))
    for mode in ['sentence','legacy']:
        with (root/f'evaluate_{mode}.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_qwen3_region_caption_chair.py','--generation-root',str(root),
                '--cohort',plan['cohort'],'--limit',str(plan['images']),'--variants',*plan['variants'],'--evaluator',mode],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'run_status.json',dict(status='complete',phase='generation_evaluation_complete',finished_at=time.time()))
    with (root/'independent_audit.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/audit_qwen3_caption_with_native_cache_v2.py',str(root)],cwd=REPO,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(root/'independent_summary.json')['passed']
    per,=(root/f'evaluations/sentence_v1/{plan["cohort"]}').glob('chair_per_image*.jsonl')
    scores=[json.loads(line) for line in per.read_bytes().splitlines()];assert [r['image_id'] for r in scores]==ids
    keep=['chair_s','hallucinated_mentions','hallucinated_objects','object_mentions','object_recall','object_f1']
    cases={}
    for name in activity_names:
        cases[name]={}
        for image,scored in zip(ids,scores,strict=True):
            if raw[name][image]['response_token_ids']==raw['sadi'][image]['response_token_ids']:continue
            cases[name][image]=dict(texts={n:raw[n][image]['generated_text'] for n in plan['variants']},
                scores={n:{k:scored['methods'][n][k] for k in keep} for n in plan['variants']},
                accepted_events=[e for e in raw[name][image]['sar_ems_statistics']['full_events'] if e['accepted']])
    write(root/'changed_cases.json',cases);write(root/'activity.json',activity)
    check(root)
    write(root/'analysis_provenance.json',dict(passed=True,protocol_sha256=sha(root/'protocol.json'),
        request_sha256=sha(root/'analysis_request.json'),finished_at=time.time(),image_review='pending',direct_advance_to500=False))
    print(json.dumps(dict(activity=activity,changed_images={n:list(v) for n,v in cases.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root)
