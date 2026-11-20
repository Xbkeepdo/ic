"""Validate both frozen signal tilts and matched controls on the reserved500."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.run_qwen3_block_acg_fresh500 import ids_sha

SOURCE=shared.OUT/'qwen3_bounded_visual_tilt_pilot_20261119'
RESERVED=shared.OUT/'qwen3_mitigation_reserved500_20261119'
OUT=shared.OUT/'qwen3_bounded_visual_tilt_validation500_20261119'
VARIANTS=['sadi','svar','tilt_fixed','tilt_drift','tilt_evidence']
MODES={'tilt_fixed':'fixed','tilt_drift':'drift','tilt_evidence':'evidence'}
SIGNALS=['tilt_drift','tilt_evidence']
FILES=['scripts/run_qwen3_bounded_visual_tilt_validation500.py',
    'scripts/audit_qwen3_bounded_visual_tilt_validation500.py','scripts/analyze_qwen3_bounded_visual_tilt_validation500.py']


def prepare():
    assert not OUT.exists(),'Do not overwrite an existing validation or redraw its cohort'
    assert shared.read(SOURCE/'post_audit_status.json')['status']=='passed'
    assert shared.read(SOURCE/'audit_recovery_status.json')['all_raw_captions_and_evaluations_unchanged']
    human=shared.read(SOURCE/'human_review.json')
    assert human['passed'] and human['advance_candidates_to_new_frozen_test']==SIGNALS
    assert all(n in shared.read(SOURCE/'screening.json')['advance_to_500'] for n in SIGNALS)
    reservation=shared.read(RESERVED/'reservation.json');selection=shared.read(RESERVED/'selection.json')
    ids=shared.read(RESERVED/'cohort.json')['coco500']
    assert reservation['before_pilot_metrics'] and reservation['reserve_only']
    assert len(ids)==len(set(ids))==500 and ids_sha(ids)==reservation['sample_sha256']==selection['sample_sha256']
    assert not set(ids)&set(selection['excluded_ids'])
    assert not set(ids)&set(shared.read(SOURCE/'cohort.json')['coco100'])
    assert not set(ids)&set(shared.read(shared.OUT/'qwen3_block_acg_fresh500_20261118/cohort.json')['coco500'])
    assert not (RESERVED/'consumed.json').exists(),'Reserved set already assigned'
    for info in selection['sources']:assert shared.sha(Path(info['path']))==info['file_sha256']
    for n,v in reservation['source_sha256'].items():assert shared.sha(Path(n))==v,n
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    hashes.update(shared.read(SOURCE/'analysis_protocol.json')['source_sha256'])
    paths=[SOURCE/n for n in ('protocol.json','analysis_protocol.json','human_review.json','screening.json',
        'post_audit_status.json','independent_summary.json','audit_recovery_protocol.json','audit_recovery_status.json')]
    paths += list((SOURCE/'coco100').glob('*.jsonl'))
    paths += [RESERVED/n for n in ('cohort.json','selection.json','reservation.json')]
    paths += [REPO/n for n in [*FILES,'scripts/audit_qwen3_bounded_visual_tilt_pilot_v2.py',
        'scripts/audit_qwen3_caption_denominator.py']]
    for p in paths:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco500').mkdir()
    for n in ('cohort.json','selection.json'):shutil.copyfile(RESERVED/n,OUT/n)
    shutil.copyfile(SOURCE/'object_lexicon.json',OUT/'object_lexicon.json')
    recipe=dict(old['recipe'])
    recipe['cache']='No generation reuse; original image semantics shared in RAM between tilt arms after exact prefix checks.'
    plan=dict(stage='independent-bounded500',model='Qwen3-VL-8B-Instruct',images=500,shards=2,
        cohort='coco500',max_batch=16,max_new_tokens=512,prompt='Describe this image.',
        variants=VARIANTS,candidates=list(MODES),signal_candidates=SIGNALS,
        copied_controls=[],cached_controls=['sadi'],base_by_candidate={n:'sadi' for n in MODES},
        mode_by_candidate=MODES,source=str(SOURCE),source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),
        source_sha256=hashes,fixed_at=time.time(),sample_sha256=ids_sha(ids),
        reservation=str(RESERVED),reservation_sha256=shared.sha(RESERVED/'reservation.json'),
        probe=old['probe'],real_threshold=old['real_threshold'],recipe=recipe,
        generation='All five methods fresh on identical ordered same-span batches; sorted IDs/stride2 shards; BF16/eager/greedy.',
        primary='Two frozen signal policies vs SADI and fixed tilt; SVAR reference; full500 sentence Cs/Ci, counts, recall/F1 and paired uncertainty.',
        interpretation='Cohort reserved before pilot scores; validation does not change methods, thresholds, layers, lengths or candidate choice.',
        cache='Only image semantics shared across tilt arms in one batch after exact native image-prefix checks. No generation skips.',
        failure_policy='No automatic OOM split or batch changes. Preserve completed outputs and inspect before any exact-group recovery.')
    shared.write(OUT/'protocol.json',plan)
    analysis_files=['scripts/analyze_qwen3_bounded_visual_tilt_validation500.py',
        'scripts/audit_qwen3_visual_feedback_mentions.py','scripts/audit_qwen3_fresh_caption500.py',
        'scripts/qwen3_caption_groups_with_cache.py']
    shared.write(OUT/'analysis_protocol.json',dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'),before_inspecting_validation_metrics=True,
        source_sha256={str(REPO/n):shared.sha(REPO/n) for n in analysis_files},
        primary='Full500 sentence results of both signal variants; all controls and legacy reported, not only the better variant.',
        cases='All fixes/introductions and top3 nonzero false/recall changes; ties by ID; inspect images.',
        sensitivity='Leave-largest/top3 and shared uncapped outputs are secondary. Report false repeats and true denominator changes.'))
    shared.write(RESERVED/'consumed.json',dict(assigned_to=str(OUT),assigned_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'),sample_sha256=ids_sha(ids)))
    print('RESERVED500_ASSIGNED',ids_sha(ids),flush=True);return plan


def verify_first_group(rows):
    native=rows['sadi'];cut=min((e['step'] for n in MODES for r in rows[n]
        for e in r['sar_ems_statistics']['full_joint_events'] if e['accepted']),default=512)
    signatures={}
    no_kept={}
    for name in MODES:
        signatures[name]=[];no_kept[name]=0
        for a,b in zip(native,rows[name],strict=True):
            assert a['image_id']==b['image_id'] and a['response_token_ids'][:cut]==b['response_token_ids'][:cut]
            info=b['sar_ems_statistics']
            signatures[name].append([g for g in info['full_gate_events'] if g['step']<=cut])
            if not info['total_accepted']:
                assert a['response_token_ids']==b['response_token_ids'] and a['generated_text']==b['generated_text']
                no_kept[name]+=1
    assert signatures['tilt_fixed']==signatures['tilt_drift']==signatures['tilt_evidence']
    return dict(passed=True,image_ids=[r['image_id'] for r in native],
        common_native_prefix_cut=cut,original_gate_decisions_exact_until_first_action=True,
        no_kept_rows_exact_to_sadi=no_kept,all_methods_same_ordered_batch=True)


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_bounded_visual_tilt import Qwen3BatchBoundedVisualTilt
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():assert shared.sha(Path(n))==v,n
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    nouns=lexical_nouns(wrapper.tokenizer,word_start='Ġ');assert nouns
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    ids=shared.read(OUT/'cohort.json')['coco500'][gpu::2]
    paths={n:OUT/f'coco500/{n}.shard{gpu}.jsonl' for n in VARIANTS};assert not any(p.exists() for p in paths.values())
    with torch.inference_mode():
        for index,(span,group) in enumerate(common.same_span_batches(wrapper,ids,COCO_ROOT,16)):
            images=[i for i,_ in group];static={};first={}
            for name in VARIANTS:
                if name in ('sadi','svar'):
                    tokens,stats,seconds=common.generate_group(wrapper,span,group,nouns,name,512);calls=None
                else:
                    holder={}
                    def factory(model,current_span,_nouns,mode,strength):
                        controller=Qwen3BatchBoundedVisualTilt(model,current_span,allowed,probe=probe,
                            semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                            shared_static=static,real_threshold=plan['real_threshold'],tilt_mode=MODES[name],gamma=1.)
                        holder['controller']=controller;return controller
                    with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                        tokens,stats,seconds=original.generate_group(wrapper,span,group,allowed,shared.POSITIVE,512)
                    calls=holder['controller'].steps;holder.clear()
                texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                rows=[dict(image_id=i,model=MODEL,variant=name,response_token_ids=t,generated_text=text.strip(),
                    seconds=seconds/len(images),batch_wall_seconds=seconds,batch_size=len(images),
                    visual_token_count=span[1]-span[0],sar_ems_statistics=stat,
                    decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                    actual_model_calls_in_batch=calls) for i,t,text,stat in zip(images,tokens,texts,stats,strict=True)]
                with paths[name].open('a',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                if index==0:first[name]=rows
                del stats;gc.collect();torch.cuda.empty_cache()
                print(f'[{gpu}] group={index} {name} B={len(images)} seconds={seconds:.3f}',flush=True)
            if index==0:shared.write(OUT/f'first_batch.gpu{gpu}.json',verify_first_group(first))
            del first;static.clear();gc.collect();torch.cuda.empty_cache()
    for p in paths.values():assert {r['image_id'] for g in shared.source_groups(p) for r in g}==set(ids)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
