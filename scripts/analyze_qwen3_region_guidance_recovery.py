"""Audit live region guidance and compute paired full-caption results."""

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
from scripts.audit_qwen3_fresh_caption500 import calculate,boot
from scripts.audit_qwen3_visual_feedback_mentions import compare


def prepare(root):
    assert not (root/'analysis_recovery_request.json').exists()
    assert not list((root/'evaluations').glob('**/metrics*.json'))
    assert 'KeyError: \'guided_steps\'' in (root/'evaluation_sentence.log').read_text()
    files=[Path(__file__),REPO/'scripts/evaluate_qwen3_region_caption_chair.py']
    write(root/'analysis_recovery_request.json',dict(fixed_at=time.time(),
        original_request_sha256=sha(root/'analysis_request.json'),
        source_sha256={str(p):sha(p) for p in files},
        reason='Legacy summary expects guided_steps; the new controller has explicit proposals and accepted counts.',
        change='Adapt activity summary only. Preserve CHAIR parser, per-image scoring, aggregation, bootstrap, raw captions and all generation settings.',
        original_failure_preserved=True,before_any_new_caption_metrics=True))


def check(root):
    request=read(root/'analysis_request.json');plan=read(root/'protocol.json')
    recovery=read(root/'analysis_recovery_request.json')
    assert recovery['original_request_sha256']==sha(root/'analysis_request.json')
    for n,v in recovery['source_sha256'].items():assert sha(Path(n))==v,n
    assert request['protocol_sha256']==sha(root/'protocol.json')
    assert request['before_inspecting_new_caption_metrics']
    for n,v in {**plan['source_sha256'],**request['source_sha256']}.items():assert sha(REPO/n)==v,n
    return plan,request


def run(root):
    import numpy as np
    plan,request=check(root)
    assert read(root/'pipeline_complete.json')['passed']
    assert all(read(root/f'worker{s}.json')['passed'] for s in [0,1])
    assert not (root/'analysis_provenance.json').exists()
    ids=read(root/'cohort.json')[plan['cohort']];assert len(ids)==100
    raw={};groups={};activity={};caption_hashes={}
    for name in plan['variants']:
        groups[name]=[];raw[name]={}
        for shard in [0,1]:
            path=root/f'coco100/{name}.shard{shard}.jsonl'
            caption_hashes[str(path)]=sha(path)
            if name in plan['copied_controls']:assert path.read_bytes()==(Path(plan['source'])/f'coco100/{path.name}').read_bytes()
            part=source_groups(path);groups[name].append([(tuple(r['image_id'] for r in g),g[0]['visual_token_count']) for g in part])
            for group in part:
                for row in group:
                    image=row['image_id'];assert image not in raw[name]
                    assert row['variant']==name and 1<=len(row['response_token_ids'])<=512
                    raw[name][image]=row
        assert set(raw[name])==set(ids)
        assert groups[name]==groups['sadi']
    for name in plan['candidates']:
        events=[];gates=[];batches=[]
        for shard in [0,1]:
            for group in source_groups(root/f'coco100/{name}.shard{shard}.jsonl'):batches.append(group[0]['sar_ems_statistics'])
        for image in ids:
            row=raw[name][image];tokens=row['response_token_ids'];s=row['sar_ems_statistics']
            assert not s['disabled'] and not s['diagnostic_zero_strength'] and not s['diagnostic_reject_all']
            assert s['all_proposals_evaluated_before_commits']
            assert s['intervention_layers']==list(range(20,36)) and s['unchanged_base_layers']==list(range(5,19))
            for gate in s['gates']:
                assert len(gate['features'])==108 and np.isfinite(gate['features']).all()
                assert gate['positive']==(gate['p_real']<.418672114610672)
                import hashlib
                assert gate['prefix_sha256']==hashlib.sha256(np.asarray(tokens[:gate['step']],dtype=np.int64).tobytes()).hexdigest()
                gates.append(gate)
            for e in s['full_events']:
                assert e['policy']==name and e['visual_mass_relative_error_fp32']<1e-5
                assert any(g['positive'] and g['step']==e['step'] and g['token']==e['token'] for g in s['gates'])
                expected=5*e['drift']*(1 if name=='unverified' else e['reliability']['reliability_deficit'])
                assert abs(e['strength']-expected)<1e-10
                assert 0<=e['drift']<=1 and 0<=e['reliability']['reliability_deficit']<=1
                assert tokens[e['step']]==(e['proposed_token'] if e['accepted'] else e['token'])
                assert e['accepted']==(e['proposed_token']!=e['token'] and e['after_logp']<e['before_logp'])
                assert e['committed_layers']==(list(range(20,36)) if e['accepted'] else [])
                if e['accepted']:assert e['proposed_current_kv_max_difference']>0
                events.append(e)
            assert s['accepted']==sum(e['accepted'] for e in s['full_events'])
        if name=='unverified':assert all(s['crop_encoder_calls_in_batch']==s['fine_prefill_calls_in_batch']==0 for s in batches)
        assert all(s['actual_encoder_calls_in_batch']==s['crop_encoder_calls_in_batch']+1-s['original_vision_cache_hits'] for s in batches)
        activity[name]=dict(proposals=len(events),accepted=sum(e['accepted'] for e in events),gate_targets=len(gates),
            active_images=sum(raw[name][i]['sar_ems_statistics']['accepted']>0 for i in ids),
            changed_captions=sum(raw[name][i]['response_token_ids']!=raw['sadi'][i]['response_token_ids'] for i in ids),
            original_encoder_calls=sum(1-s['original_vision_cache_hits'] for s in batches),
            crop_encoder_calls=sum(s['crop_encoder_calls_in_batch'] for s in batches),
            fine_prefill_calls=sum(s['fine_prefill_calls_in_batch'] for s in batches),
            fine_probability_cache_hits=sum(s['fine_probability_cache_hits'] for s in batches),
            native_static_cache_hits=sum(s['native_static_cache_hits'] for s in batches),
            seconds=sum(raw[name][i]['seconds'] for i in ids),
            max_visual_mass_relative_error=max([e['visual_mass_relative_error_fp32'] for e in events],default=0))
    proof=dict(passed=True,activity=activity,caption_sha256=caption_hashes)
    assert read(root/'mechanism_audit.json')==proof
    # Original captions are immutable. The two evaluator versions write to separate roots.
    for mode in ['sentence','legacy']:
        with (root/f'evaluation_{mode}_recovery.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_qwen3_region_caption_chair.py','--generation-root',str(root),
                '--cohort','coco100','--limit','100','--variants',*plan['variants'],'--evaluator',mode],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    draws=np.random.default_rng(43).integers(0,100,size=(5000,100));summary={};paired={};review={};uncapped={};screen={}
    changed={name:{} for name in plan['candidates']}
    for mode in ['sentence','legacy']:
        folder=root/f'evaluations/{mode}_v1/coco100';per,=folder.glob('chair_per_image*.jsonl');metric,=folder.glob('metrics*.json')
        rows=[json.loads(line) for line in per.read_bytes().splitlines()];assert [r['image_id'] for r in rows]==ids
        values={name:[r['methods'][name] for r in rows] for name in plan['variants']}
        summary[mode]={name:calculate(records) for name,records in values.items()}
        metrics=read(metric);assert metrics['evaluation_inputs']['caption_sha256']==caption_hashes
        samples={name:boot(records,draws) for name,records in values.items()}
        for name,record in summary[mode].items():
            for key in ['chair_s','chair_i','object_recall','object_f1']:
                np.testing.assert_allclose(record[key],metrics['methods'][name][key],rtol=0,atol=1e-12)
        paired[mode]={}
        for name in plan['candidates']:
            paired[mode][name]={}
            for reference in plan['variants']:
                if name==reference:continue
                a,b=summary[mode][reference],summary[mode][name]
                detail=compare(values[reference],values[name],ids,draws)
                detail.update(delta={k:b[k]-a[k] for k in samples[name]},
                    delta_ci95={k:np.quantile(samples[name][k]-samples[reference][k],[.025,.975]).tolist() for k in samples[name]})
                paired[mode][name][reference]=detail
            if mode!='sentence':continue
            a,b=summary[mode]['sadi'],summary[mode][name];detail=paired[mode][name]['sadi'];drop=detail['drop_largest_reduction']
            checks=dict(chair_s_lower=b['chair_s']<a['chair_s'],chair_i_lower=b['chair_i']<a['chair_i'],
                false_mentions_lower=b['hallucinated_mentions']<a['hallucinated_mentions'],
                f1_loss_at_most_1_5pp=b['object_f1']>=a['object_f1']-.015,
                recall_loss_at_most_2pp=b['object_recall']>=a['object_recall']-.02,
                leave_largest_ci_no_worse=drop['candidate']['chair_i']<=drop['base']['chair_i'])
            screen[name]=dict(checks=checks,numeric_pass=all(checks.values()),image_review='pending')
            queues=dict(fixed=[],introduced=[],recall_losses=[])
            for image,left,right in zip(ids,values['sadi'],values[name],strict=True):
                if left['chair_s']>right['chair_s']:queues['fixed'].append(image)
                if left['chair_s']<right['chair_s']:queues['introduced'].append(image)
                if left['object_recall']>right['object_recall']:queues['recall_losses'].append(image)
                if raw[name][image]['response_token_ids']!=raw['sadi'][image]['response_token_ids']:
                    changed[name][image]=dict(scores={'sadi':left,name:right},
                        captions={n:raw[n][image]['generated_text'] for n in plan['variants']},
                        events=raw[name][image]['sar_ems_statistics']['full_events'])
            review[name]=dict(queues=queues,image_ids=sorted(set().union(*map(set,queues.values()))),image_review='pending')
        subset=[j for j,i in enumerate(ids) if all(len(raw[n][i]['response_token_ids'])<512 for n in plan['variants'])]
        uncapped[mode]=dict(ids=[ids[j] for j in subset],output_defined_secondary=True,
            methods={n:calculate([records[j] for j in subset]) for n,records in values.items()} if subset else {})
    check(root)
    assert all(sha(Path(p))==v for p,v in caption_hashes.items())
    for name,data in [('summary.json',summary),('paired.json',paired),('activity.json',activity),
        ('changed_cases.json',changed),('image_review_queue.json',review),('screening.json',screen),('uncapped.json',uncapped)]:write(root/name,data)
    write(root/'analysis_provenance.json',dict(passed=True,analysis_request_sha256=sha(root/'analysis_request.json'),
        protocol_sha256=sha(root/'protocol.json'),caption_sha256=caption_hashes,finished_at=time.time(),image_review='pending',
        analysis_recovery_request_sha256=sha(root/'analysis_recovery_request.json')))
    write(root/'run_status.json',dict(status='complete',finished_at=time.time()))
    print(json.dumps(dict(summary=summary['sentence'],screening=screen,activity=activity),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root)
