"""Descriptive paired analysis for the frozen bounded-tilt validation500."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.audit_qwen3_visual_feedback_mentions import compare
from scripts.audit_qwen3_fresh_caption500 import calculate
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.qwen3_caption_groups_with_cache import source_groups


BRIEF_FIELDS=('step','source_layer','target_layer','token','proposed_token','emitted_token',
    'accepted','terminal_eos','probe_p_real','native_top2_gap','original_target_log_probability',
    'predicted_log_probability_change','actual_log_probability_change','immediate_token_changed')


def brief_event(event,tokens,tokenizer):
    predicted='predicted_log_probability_change'
    value={key:event[key] for key in BRIEF_FIELDS if key!=predicted}
    if event['signal_guided']:
        value[predicted]=event[predicted]
    else:
        assert predicted not in event or event[predicted] is None
        value[predicted]=None
    value['signal_guided']=event['signal_guided']
    value['gradient_prediction_available']=event['signal_guided']
    decode=lambda sequence:tokenizer.decode(sequence,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    value.update(target_text=decode([event['token']]),proposed_text=decode([event['proposed_token']]),
        prefix_context=decode(tokens[max(0,event['step']-12):event['step']]),
        generated_context=decode(tokens[max(0,event['step']-4):event['step']+5]))
    return value


def analyze(root):
    from transformers import AutoTokenizer
    plan=read(root/'protocol.json');cohort=plan['cohort'];ids=read(root/'cohort.json')[cohort]
    assert read(root/'run_status.json')['status']=='complete'
    assert read(root/'mechanism_audit.json')['passed'] and read(root/'independent_summary.json')['passed']
    analysis=read(root/'analysis_protocol.json')
    assert analysis['generation_protocol_sha256']==sha(root/'protocol.json')
    assert analysis['before_inspecting_validation_metrics']
    for n,v in analysis['source_sha256'].items():assert sha(Path(n))==v,n
    assert not (root/'analysis_provenance.json').exists()
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    raw={};groups={};inputs={str(root/'protocol.json'):sha(root/'protocol.json')}
    for name in plan['variants']:
        paths=[root/f'{cohort}/{name}.shard{s}.jsonl' for s in (0,1)]
        groups[name]=[g for p in paths for g in source_groups(p)]
        raw[name]={r['image_id']:r for group in groups[name] for r in group}
        assert set(raw[name])==set(ids)
        inputs.update({str(p):sha(p) for p in paths})
    draws=np.random.default_rng(43).integers(0,len(ids),size=(5000,len(ids)))
    summaries={};case_records={};review={};sensitivity={}
    for mode in ('sentence','legacy'):
        per_path,=(root/f'evaluations/{mode}_v1/{cohort}').glob('chair_per_image*.jsonl')
        inputs[str(per_path)]=sha(per_path)
        rows=[json.loads(line) for line in per_path.read_bytes().splitlines()]
        assert [r['image_id'] for r in rows]==ids
        scores={name:[r['methods'][name] for r in rows] for name in plan['variants']}
        summaries[mode]={name:{ref:compare(scores[ref],scores[name],ids,draws)
            for ref in plan['variants'] if ref!=name} for name in plan['candidates']}
        uncapped=[index for index,image in enumerate(ids)
            if all(len(raw[name][image]['response_token_ids'])<plan['max_new_tokens'] for name in plan['variants'])]
        sensitivity[mode]=dict(shared_uncapped_image_ids=[ids[i] for i in uncapped],
            scope='Output-defined descriptive sensitivity only; full500 remains the primary result.',
            methods={name:calculate([values[i] for i in uncapped]) for name,values in scores.items()} if uncapped else {})
        if mode!='sentence':continue
        by_id={r['image_id']:r['methods'] for r in rows}
        for name in plan['candidates']:
            changed=[image for image in ids if raw[name][image]['response_token_ids']!=raw[plan['base_by_candidate'][name]][image]['response_token_ids']]
            case_records[name]={}
            for image in changed:
                a,b=by_id[image][plan['base_by_candidate'][name]],by_id[image][name]
                case_records[name][str(image)]=dict(
                    false_mentions_changed=Counter(map(tuple,a['hallucinated_objects']))!=Counter(map(tuple,b['hallucinated_objects'])),
                    methods={m:dict(text=raw[m][image]['generated_text'],scores=by_id[image][m],
                        events=[brief_event(e,raw[m][image]['response_token_ids'],tokenizer)
                                for e in raw[m][image]['sar_ems_statistics'].get('full_joint_events',[])])
                        for m in plan['variants']},
                    causal_scope='A later mention can change indirectly after an earlier accepted proposal; token text alone does not establish image correctness.')
            baseline,candidate=scores[plan['base_by_candidate'][name]],scores[name]
            reductions=[a['hallucinated_mentions']-b['hallucinated_mentions'] for a,b in zip(baseline,candidate,strict=True)]
            recall_loss=[a['object_recall']-b['object_recall'] for a,b in zip(baseline,candidate,strict=True)]
            fixed=[ids[i] for i,(a,b) in enumerate(zip(baseline,candidate,strict=True)) if a['chair_s']>b['chair_s']]
            introduced=[ids[i] for i,(a,b) in enumerate(zip(baseline,candidate,strict=True)) if a['chair_s']<b['chair_s']]
            top=lambda values,reverse:[ids[i] for i in sorted(
                [i for i in range(len(ids)) if (values[i]>0 if reverse else values[i]<0)],
                key=lambda i:((-values[i] if reverse else values[i]),ids[i]))[:3]]
            queues=dict(all_fixed_hallucinated_images=fixed,all_introduced_hallucinated_images=introduced,
                largest_three_false_mention_reductions=top(reductions,True),largest_three_false_mention_increases=top(reductions,False),
                largest_three_recall_losses=top(recall_loss,True),largest_three_recall_gains=top(recall_loss,False))
            review[name]=dict(queues=queues,image_ids=sorted(set().union(*map(set,queues.values()))),
                confirmed_image_identity_corrections=[],image_review='pending',
                interpretation='These queues include automatic colour/sense/alias matches; inspect actual photos and earliest causal events before calling a physical-object correction.')
    activity={}
    for name in plan['candidates']:
        batches=[g[0]['sar_ems_statistics'] for g in groups[name]]
        events=[e for r in raw[name].values() for e in r['sar_ems_statistics']['full_joint_events']]
        kept=[e for e in events if e['accepted']]
        activity[name]=dict(proposals=len(events),kept=len(kept),terminal_eos_accepts=sum(e['terminal_eos'] for e in kept),
            active_images=sum(r['sar_ems_statistics']['total_accepted']>0 for r in raw[name].values()),
            changed_token_images=sum(raw[name][image]['response_token_ids']!=raw[plan['base_by_candidate'][name]][image]['response_token_ids'] for image in ids),
            kept_immediate_token_changes=sum(e['immediate_token_changed'] for e in kept),
            positive_margin_kept_without_token_change=sum(e['native_top2_gap']>0 and not e['immediate_token_changed'] for e in kept),
            gradient_calls=sum(s['gradient_probe_calls'] for s in batches),
            semantic_head_calls=sum(s['static_head_calls'] for s in batches),static_hits=sum(s['shared_static_hits'] for s in batches),
            native_feature_seconds=sum(s['native_feature_seconds'] for s in batches),
            semantic_setup_seconds=sum(s['static_seconds'] for s in batches),
            semantic_signature_seconds=sum(s['semantic_signature_seconds'] for s in batches),
            timing_scope='Feature time includes semantic setup/signature; do not add these nested values. Fixed runs first and pays shared setup, so this is not an intrinsic fixed-versus-guided speed comparison.',
            mean_native_top2_gap=sum(e['native_top2_gap'] for e in events)/max(1,len(events)),
            mean_accepted_log_probability_change=sum(e['actual_log_probability_change'] for e in kept)/max(1,len(kept)))
    write(root/'paired_mentions.json',summaries)
    write(root/'changed_text_cases.json',case_records)
    write(root/'image_review_queue.json',review)
    write(root/'uncapped_sensitivity.json',sensitivity)
    write(root/'activity.json',activity)
    write(root/'analysis_provenance.json',dict(passed=True,analysis_protocol_sha256=sha(root/'analysis_protocol.json'),
        analysis_source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(root/'protocol.json'),input_sha256=inputs,image_review='pending',
        screening_scope='One frozen prospective validation; no change to gate, scales or layers based on these outcomes.'))
    print(json.dumps(dict(activity=activity,review_queue_sizes={k:len(v['image_ids']) for k,v in review.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
