"""Compare fixed, drift-only and evidence-weighted bounded visual tilts."""

import argparse
import copy
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
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE=shared.OUT/'qwen3_collective_visual_pilot_20261119'
ORIGINAL=shared.OUT/'qwen3_block_acg_pilot_20261118'
OUT=shared.OUT/'qwen3_bounded_visual_tilt_pilot_20261119'
MODES={'tilt_fixed':'fixed','tilt_drift':'drift','tilt_evidence':'evidence'}
CONTROLS=['sadi','svar','sadi_sam_acg_guided_19_35','collective']
CERTIFICATE=ORIGINAL/'no_gate_cache_validation.json'
CPU=shared.OUT/'qwen3_bounded_visual_tilt_cpu/verification.json'
PREFLIGHT=shared.OUT/'qwen3_bounded_visual_tilt_preflight/verification.json'
FILES=['features/qwen3_bounded_visual_tilt.py','scripts/verify_qwen3_bounded_visual_tilt.py',
    'scripts/verify_qwen3_bounded_visual_tilt_gpu.py','scripts/run_qwen3_bounded_visual_tilt_pilot.py',
    'scripts/audit_qwen3_bounded_visual_tilt_pilot.py','scripts/analyze_qwen3_bounded_visual_tilt_pilot.py']


def prepare():
    assert not OUT.exists(),'Inspect an existing job before any resume'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert shared.read(CPU)['passed'] and shared.read(PREFLIGHT)['passed']
    for n,v in shared.read(PREFLIGHT)['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    hashes.update(shared.read(SOURCE/'analysis_protocol.json')['source_sha256'])
    certificate=shared.read(CERTIFICATE)
    assert certificate['passed'] and certificate['joint17_agreement_verified']
    for n,v in certificate['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    for p in [CPU,PREFLIGHT,CERTIFICATE,SOURCE/'protocol.json',SOURCE/'human_review.json',
            SOURCE/'final_provenance.json',*(REPO/n for n in FILES)]:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for n in ('cohort.json','object_lexicon.json'):shutil.copyfile(SOURCE/n,OUT/n)
    for shard in (0,1):
        for name in CONTROLS:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';hashes[str(p)]=shared.sha(p)
            shutil.copyfile(p,OUT/'coco100'/p.name)
    plan=dict(stage='bounded-visual-development',images=100,shards=2,cohort='coco100',
        max_batch=16,max_new_tokens=512,variants=[*CONTROLS,*MODES],candidates=list(MODES),
        copied_controls=CONTROLS,cached_controls=['sadi'],base_by_candidate={n:'sadi' for n in MODES},
        mode_by_candidate=MODES,source=str(SOURCE),source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),
        source_sha256=hashes,fixed_at=time.time(),probe=old['probe'],real_threshold=old['real_threshold'],
        cache_certificate=str(CERTIFICATE),cache_certificate_sha256=shared.sha(CERTIFICATE),
        recipe=dict(base_layers=[5,18],joint_layers=list(range(19,36)),gamma=1.,
            gate='Unchanged original108 SAM-V2, threshold, first token/class scope; exact original feature vectors checked in preflight.',
            shift='abs(native visual logit)*coefficient; fixed=.5; drift=.5D; evidence=D*Q/(Q+P).',
            signals='Q=Norm(write_norm*(1+log_s)); P=softmax(cos(uFFN,hpre_visual)/.03); D=HellingerSquared(Q,P).',
            replay='Native-point joint19--35 context differences; cached native context, one tilted softmax@V per layer, unchanged causal tail replay.',
            acceptance='Original target actual final logp decreases; otherwise exact hidden/current20--35 KV rollback.',
            costs='No backward, no second encoder; shared image semantics in RAM with exact prefix checks.',
            cache='Seven original complete zero-positive-gate batches, nine images per candidate; no positive-gate reuse or rebatching.',
            scope='CAI-style positive-logit-tilt adaptation after full SADI, not an original CAI reproduction.'),
        development_screen='Versus SADI both sentence Cs/Ci and false count decrease; F1 loss<=1.5pp and recall loss<=2pp; leave-largest Ci no worse. Inspect images and compare signal variants to fixed/drift controls before new500.',
        followup='Reused old100 is exploratory. Useful signal-guided candidates require a different frozen500, excluding all prior images.')
    assert len('chair_per_image.first100.'+'-'.join(sorted(plan['variants']))+'.jsonl')<=255
    shared.write(OUT/'protocol.json',plan)
    analysis_files=['scripts/analyze_qwen3_bounded_visual_tilt_pilot.py',
        'scripts/analyze_qwen3_scalar_visual_pilot.py','scripts/audit_qwen3_visual_feedback_mentions.py',
        'scripts/audit_qwen3_fresh_caption500.py','scripts/qwen3_caption_groups_with_cache.py']
    shared.write(OUT/'analysis_protocol.json',dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'),before_inspecting_pilot_metrics=True,
        source_sha256={str(REPO/n):shared.sha(REPO/n) for n in analysis_files},
        primary='Full100 sentence, counts/F1/recall and paired intervals; compare evidence to fixed and drift, and all candidates to SADI/SVAR.',
        cases='All hallucinated-image fixes/introductions plus top3 nonzero mention/recall changes; ties by ID; inspect photographs.',
        scope='Old development100; no independent validation claim; shared uncapped sensitivity cannot replace full results.'))
    return plan


def cache_rows(group,plan,shard,index,name,mode):
    source=ORIGINAL/f'coco100/sadi_sam_acg_guided_19_35.shard{shard}.jsonl'
    rows=[];started=time.perf_counter()
    for ref in group:
        stat=copy.deepcopy(ref['sar_ems_statistics'])
        assert not stat['recycle_events'] and not any(g['positive'] for g in stat['gate_events'])
        for k in ('native_risk_solver_calls','native_feature_seconds','static_seconds','static_head_calls',
                  'shared_static_hits','semantic_signature_seconds','gradient_probe_calls','probe_calls'):stat[k]=0
        stat.update(full_joint_events=[],full_gate_events=copy.deepcopy(stat['gate_events']),
            full_class_decisions=copy.deepcopy(stat['class_decisions']),total_proposals=0,total_accepted=0,
            terminal_eos_accepts=0,full_event_log_includes_terminal_eos=True,signal_guided=False,
            gradient_sources=[],one_backward_for_all_layers=False,bounded_visual_tilt=True,tilt_mode=mode,
            tilt_scale=1.,transport_mode='bounded_positive_visual_logit_tilt',native_context_captures=0,
            zero_tilt_exact_checks=0,diagnostic_zero_tilt=False,cached_gate_decisions=len(stat['gate_events']),
            event_log_from_certified_cache=True,acg_scope='No positive gate; bounded visual tilt generation reused unchanged.')
        rows.append(dict(ref,variant=name,seconds=0.,batch_wall_seconds=0.,sar_ems_statistics=stat,
            decoder='cached_native_no_positive_gate',generation_reused=True,actual_model_calls_in_batch=0,
            cache_origin=dict(source=str(source),source_sha256=shared.sha(source),certificate=str(CERTIFICATE),
                certificate_sha256=plan['cache_certificate_sha256'],shard=shard,group=index)))
    elapsed=time.perf_counter()-started
    for row in rows:row['cache_io_seconds']=elapsed/len(rows)
    return rows


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_bounded_visual_tilt import Qwen3BatchBoundedVisualTilt
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():assert shared.sha(Path(n))==v,n
    groups=source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    cached=source_groups(ORIGINAL/f'coco100/sadi_sam_acg_guided_19_35.shard{gpu}.jsonl')
    assert shared.group_keys(cached)==shared.group_keys(groups)
    eligible={e['group']:e for e in shared.read(CERTIFICATE)['eligible_groups'] if e['shard']==gpu}
    paths={n:OUT/f'coco100/{n}.shard{gpu}.jsonl' for n in MODES};assert not any(p.exists() for p in paths.values())
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py');verified=False;memory=[]
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group];static={}
            if index not in eligible:
                (span,inputs),=common.same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
                assert [i for i,_ in inputs]==ids and span[1]-span[0]==group[0]['visual_token_count']
                if not verified:
                    tokens,_,_=common.generate_group(wrapper,span,inputs,allowed,'sadi',32)
                    assert tokens==[r['response_token_ids'][:32] for r in group]
                    shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,ids=ids,native32_exact=True))
                    gc.collect();torch.cuda.empty_cache();verified=True
            for name,tilt_mode in MODES.items():
                if index in eligible:
                    assert eligible[index]['image_ids']==ids
                    rows=cache_rows(cached[index],plan,gpu,index,name,tilt_mode)
                else:
                    holder={}
                    def factory(model,current_span,nouns,mode,strength):
                        controller=Qwen3BatchBoundedVisualTilt(model,current_span,allowed,probe=probe,
                            semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                            shared_static=static,real_threshold=plan['real_threshold'],tilt_mode=tilt_mode,gamma=1.)
                        holder['controller']=controller;return controller
                    torch.cuda.reset_peak_memory_stats(wrapper.device)
                    with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                        tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
                    calls=holder['controller'].steps;holder.clear()
                    texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    rows=[dict(image_id=i,model=MODEL,variant=name,response_token_ids=t,generated_text=text.strip(),
                        seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),
                        visual_token_count=span[1]-span[0],sar_ems_statistics=stat,
                        decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                        actual_model_calls_in_batch=calls) for i,t,text,stat in zip(ids,tokens,texts,stats,strict=True)]
                with paths[name].open('a',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                gc.collect();torch.cuda.empty_cache()
                memory.append(dict(group=index,variant=name,allocated=torch.cuda.memory_allocated(wrapper.device),
                    peak=torch.cuda.max_memory_allocated(wrapper.device),reused=rows[0]['generation_reused']))
                shared.write(OUT/f'memory.gpu{gpu}.json',memory)
                print(f'[{gpu}] {name} group={index} B={len(ids)} seconds={rows[0]["batch_wall_seconds"]:.3f}',flush=True)
            static.clear();gc.collect();torch.cuda.empty_cache()
    assert all(shared.group_keys(source_groups(p))==shared.group_keys(groups) for p in paths.values())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
