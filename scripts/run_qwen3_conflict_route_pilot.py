"""Three fresh route-gating arms after complete SADI on the frozen development100."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE=shared.OUT/'qwen3_evidence_head_routing_pilot_20261119'
PREFLIGHT=shared.OUT/'qwen3_conflict_route_preflight_20261119'
RESERVATION=shared.OUT/'qwen3_mass_matched_visual_reserved500_20261119'
OUT=shared.OUT/'qwen3_conflict_route_pilot_20261119'
MODES={'route_vri':'vri','route_signal':'signal','route_roll':'roll'}
CONTROLS=['sadi','svar','causal_fixed']
FILES=['scripts/run_qwen3_conflict_route_pilot.py','scripts/audit_qwen3_conflict_route_pilot.py',
       'scripts/analyze_qwen3_conflict_route_pilot.py']
AUDITS=[FILES[1],'scripts/audit_qwen3_caption_with_native_cache_v2.py',FILES[2],
        'scripts/audit_qwen3_caption_denominator.py']


def prepare():
    assert not OUT.exists(), 'Inspect existing workers rather than restarting'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    proof=shared.read(PREFLIGHT/'verification.json');request=shared.read(PREFLIGHT/'request.json')
    assert proof['passed'] and proof['request_sha256']==shared.sha(PREFLIGHT/'request.json')
    assert set(proof['batches'])=={'B2','B16'}
    assert not (RESERVATION/'consumed.json').exists()
    hashes=dict(request['source_sha256'])
    files=[*FILES,*AUDITS,'scripts/analyze_qwen3_scalar_visual_pilot.py',
        'scripts/analyze_qwen3_causal_visual_tilt_pilot.py','scripts/audit_qwen3_visual_feedback_mentions.py',
        'scripts/audit_qwen3_fresh_caption500.py','scripts/qwen3_caption_groups_with_cache.py']
    for p in [PREFLIGHT/'verification.json',PREFLIGHT/'request.json',SOURCE/'protocol.json',
              SOURCE/'final_provenance.json',*(RESERVATION/n for n in ['cohort.json','selection.json','reservation.json']),
              *(REPO/n for n in files)]:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(REPO/n)==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for n in ['cohort.json','object_lexicon.json']:shutil.copyfile(SOURCE/n,OUT/n)
    assert not set(shared.read(OUT/'cohort.json')['coco100'])&set(shared.read(RESERVATION/'cohort.json')['coco500'])
    for shard in (0,1):
        for name in CONTROLS:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';hashes[str(p)]=shared.sha(p)
            shutil.copyfile(p,OUT/'coco100'/p.name)
    old=shared.read(SOURCE/'protocol.json')
    plan=dict(stage='conflict-route-development',images=100,cohort='coco100',shards=2,
        max_batch=16,max_new_tokens=512,variants=[*CONTROLS,*MODES],candidates=list(MODES),
        copied_controls=CONTROLS,cached_controls=['sadi'],source=str(SOURCE),
        source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),source_sha256=hashes,
        base_by_candidate={n:'sadi' for n in MODES},mode_by_candidate=MODES,
        no_learned_gate=True,head_count=32,route_k=8,eos_token_ids=old['eos_token_ids'],fixed_at=time.time(),
        recipe=dict(base_layers=[5,18],intervention_layers=list(range(19,36)),
            eligibility='Lexical309, unseen token and canonical class in generated prefix; no learned model, no annotation.',
            gradient='One native-point backward gives final-target sensitivities at all17 post-SADI layers.',
            conflict='visual_effect<0 and nonvisual_text_effect>0, leaving all other heads unchanged.',
            vri='Select up to8 conflict heads by low |visual|/(|visual|+|text|+1e-8).',
            signals='Q=Norm(write_norm*(1+log_s));P=softmax(cos(actual native FFN,hpre_visual)/.03).',
            risk='sum_j conditional_visual_attention_hj * P_j/(Q_j+P_j), descending; VRI tie-break.',
            roll='Move head risk by16/32 heads, then rank within the same local conflict set.',
            gates='Linear0--.5 for selected heads; identical conditional-on-conflict gate multiset across rules, not matched global trajectories.',
            action='Apply native frozen gates to actual current-query text-route context in each replayed layer; no attention renormalization.',
            acceptance='Actual original target log probability decreases; otherwise restore hidden/current20--35KV. HistoricalKV unchanged.',
            cost='No new training or extra encoder. Source Q/P semantic cache shared between signal/roll; no108features/OT calls.',
            cache='Every candidate caption generated fresh; old9 no-positive-gate certificates invalid here.',
            previous='Old31 history-only4head/repeated-token/.5+.5D trial failed; this changes route scope, layer range and tests ranking at fixed local schedules.',
            adaptation='CRG Conflict-B idea adapted after fullSADI, not a full-paper reproduction or refreshed-next-layer signal.'),
        development_screen='Signal must improve Cs/Ci/false counts beyond VRI and roll without repetition-denominator gaming; modest F1/recall loss only; inspect original images.',
        followup='Only useful signal increments may use unchanged reserved1903 independent500; no redraw or validation tuning.')
    shared.write(OUT/'protocol.json',plan)
    shared.write(OUT/'analysis_protocol.json',dict(fixed_at=time.time(),before_inspecting_pilot_metrics=True,
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'),source_sha256={str(REPO/n):shared.sha(REPO/n) for n in files},
        primary='Full100 sentence and legacy, all6 arms, paired intervals and counts/recall/F1.',
        cases='All fixes/introductions and largest3 nonzero false/recall changes; all false canonical-set changes against SADI/VRI/roll, original photos.',
        scope='Old development100, no independent validation claim. Output-defined uncapped subset secondary.'))
    return plan


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_conflict_route_feedback import Qwen3BatchConflictRoute
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    groups=source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    paths={n:OUT/f'coco100/{n}.shard{gpu}.jsonl' for n in MODES};assert not any(p.exists() for p in paths.values())
    wrapper=build_wrapper();allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py');memory=[]
    eos=plan['eos_token_ids'];eos=set(eos if isinstance(eos,list) else [eos])
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group];static={}
            (span,inputs),=common.same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
            assert [image for image,_ in inputs]==ids and span[1]-span[0]==group[0]['visual_token_count']
            if index==0:
                tokens,_,_=common.generate_group(wrapper,span,inputs,allowed,'sadi',32)
                assert tokens==[r['response_token_ids'][:32] for r in group]
                shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,ids=ids,native32_exact=True))
                gc.collect();torch.cuda.empty_cache()
            for name,route_mode in MODES.items():
                holder={}
                def factory(model,current_span,nouns,mode,strength):
                    controller=Qwen3BatchConflictRoute(model,current_span,allowed,tokenizer=wrapper.tokenizer,
                        history_parser=parser,shared_static=static,route_mode=route_mode,k=8,gamma=1.)
                    holder['controller']=controller;return controller
                torch.cuda.reset_peak_memory_stats(wrapper.device)
                with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                    tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
                calls=holder['controller'].steps;holder.clear()
                for stat in stats:
                    for event in stat['full_joint_events']:event['emitted_is_eos']=event['emitted_token'] in eos
                    stat['actual_eos_accepts']=sum(e['accepted'] and e['emitted_is_eos'] for e in stat['full_joint_events'])
                texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                rows=[dict(image_id=image,model=MODEL,variant=name,response_token_ids=t,generated_text=text.strip(),
                    seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),visual_token_count=span[1]-span[0],
                    sar_ems_statistics=stat,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                    actual_model_calls_in_batch=calls) for image,t,text,stat in zip(ids,tokens,texts,stats,strict=True)]
                with paths[name].open('a',encoding='utf-8') as h:
                    for row in rows:h.write(json.dumps(row,ensure_ascii=False)+'\n')
                memory.append(dict(group=index,variant=name,allocated=torch.cuda.memory_allocated(wrapper.device),
                    peak=torch.cuda.max_memory_allocated(wrapper.device),reused=False))
                shared.write(OUT/f'memory.gpu{gpu}.json',memory)
                print(f'[{gpu}] {name} group={index} B={len(ids)} seconds={seconds:.3f}',flush=True)
                gc.collect();torch.cuda.empty_cache()
            static.clear();gc.collect();torch.cuda.empty_cache()
    assert all(shared.group_keys(source_groups(p))==shared.group_keys(groups) for p in paths.values())


def run():
    with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
    commands=[[name,str(OUT)] for name in AUDITS]
    shared.write(OUT/'post_audit_status.json',dict(status='running',commands=commands,started=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
    try:
        for i,args in enumerate(commands):
            with (OUT/f'post_audit{i}.log').open('x') as h:
                subprocess.run([sys.executable,'-u',*args],cwd=REPO,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
        shared.write(OUT/'post_audit_status.json',dict(status='passed',commands=commands,finished=time.time()))
    except BaseException as exc:
        shared.write(OUT/'post_audit_status.json',dict(status='failed',commands=commands,error=repr(exc),failed=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1))
    args=p.parse_args();run() if args.worker is None else worker(args.worker)
