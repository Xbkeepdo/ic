"""Test the complete-visual-support veto after unchanged full SADI on old100."""

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

SOURCE=shared.OUT/'qwen3_scalar_visual_pilot_20261119'
ORIGINAL=shared.OUT/'qwen3_block_acg_pilot_20261118'
OUT=shared.OUT/'qwen3_collective_visual_pilot_20261119'
NAME='collective'
CONTROLS=['sadi','svar','sadi_sam_acg_fixed_19_35','sadi_sam_acg_guided_19_35','sadi_scalar']
CERTIFICATE=ORIGINAL/'no_gate_cache_validation.json'
CPU=shared.OUT/'qwen3_collective_visual_cpu/verification.json'
PREFLIGHT=shared.OUT/'qwen3_collective_visual_preflight/verification.json'
FILES=['features/qwen3_collective_visual_gate.py','scripts/verify_qwen3_collective_visual_gate.py',
    'scripts/verify_qwen3_collective_visual_gpu.py','scripts/run_qwen3_collective_visual_pilot.py',
    'scripts/audit_qwen3_collective_visual_pilot.py','scripts/analyze_qwen3_collective_visual_pilot.py']


def prepare():
    assert not OUT.exists(),'Inspect existing job; never overwrite or resample'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert not shared.read(SOURCE/'human_review.json')['candidate_accepted']
    assert shared.read(CPU)['passed'] and shared.read(PREFLIGHT)['passed']
    for n,v in shared.read(PREFLIGHT)['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    hashes.update(shared.read(SOURCE/'analysis_protocol.json')['source_sha256'])
    coverage=shared.OUT/'qwen3_native_gate_coverage_20261118/coverage_provenance.json'
    hashes.update(shared.read(coverage)['supplemental_inputs'])
    certificate=shared.read(CERTIFICATE)
    assert certificate['passed'] and certificate['joint17_agreement_verified']
    for n,v in certificate['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    for p in [CPU,PREFLIGHT,CERTIFICATE,coverage,SOURCE/'protocol.json',SOURCE/'human_review.json',
            SOURCE/'final_provenance.json',*(REPO/n for n in FILES)]:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for n in ('cohort.json','object_lexicon.json'):shutil.copyfile(SOURCE/n,OUT/n)
    for shard in (0,1):
        for name in CONTROLS:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';hashes[str(p)]=shared.sha(p)
            shutil.copyfile(p,OUT/'coco100'/p.name)
    plan=dict(stage='collective-visual-development',images=100,shards=2,cohort='coco100',
        max_batch=16,max_new_tokens=512,variants=[*CONTROLS,NAME],candidates=[NAME],
        copied_controls=CONTROLS,cached_controls=['sadi'],base_by_candidate={NAME:'sadi'},
        source=str(SOURCE),source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),
        source_sha256=hashes,fixed_at=time.time(),probe=old['probe'],real_threshold=old['real_threshold'],
        cache_certificate=str(CERTIFICATE),cache_certificate_sha256=shared.sha(CERTIFICATE),
        recipe=dict(base_layers=[5,18],joint_layers=list(range(19,36)),gamma=1.4,
            unchanged='Original108 gate, first token/class, directions,1.4D negative-head scale, actual logp-decrease acceptance and rollback.',
            new_rule='E_vis=sum over17layers/32heads of native gradient dot complete orthogonal visual direction; E_vis>=0 vetoes the complete replay. No extra threshold or scale.',
            interpretation='First-order collective visual direction, not a hallucination label or exact nonlinear causal support.',
            costs='One joint backward per SAM-positive batch step, including steps later vetoed. No extra backward for the veto.',
            cache='Only the same seven certified complete zero-SAM-gate batches, nine images. No skipped row from a positive-gate group.',
            timing='Reused rows have zero model calls/seconds; cache I/O explicit. Native-prefix validation and loading included in driver wall time.'),
        development_screen=old['development_screen'],
        followup='Compare against SADI and original negative-head guided/fixed adaptations; require image evidence before a different independent500. This old100 is exploratory.')
    assert len('chair_per_image.first100.'+'-'.join(sorted(plan['variants']))+'.jsonl')<=255
    shared.write(OUT/'protocol.json',plan)
    analysis_files=['scripts/analyze_qwen3_collective_visual_pilot.py',
        'scripts/analyze_qwen3_scalar_visual_pilot.py','scripts/audit_qwen3_visual_feedback_mentions.py',
        'scripts/audit_qwen3_fresh_caption500.py','scripts/qwen3_caption_groups_with_cache.py']
    shared.write(OUT/'analysis_protocol.json',dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'),before_inspecting_pilot_metrics=True,
        source_sha256={str(REPO/n):shared.sha(REPO/n) for n in analysis_files},
        primary='Full100 sentence; SADI and original guided/fixed adaptations; false counts, F1/recall and paired intervals.',
        cases='All fixes/introductions and three largest nonzero false/recall changes, ties by image ID; inspect photographs.',
        interpretation='Old development100, not independent validation; first-order veto decisions are not object labels.',
        costs='Actual batch backward calls include candidates later vetoed; complete terminal-EOS and cache records retained.'))
    return plan


def cache_rows(group,plan,shard,index):
    source=ORIGINAL/f'coco100/sadi_sam_acg_guided_19_35.shard{shard}.jsonl'
    rows=[];started=time.perf_counter()
    for ref in group:
        stat=copy.deepcopy(ref['sar_ems_statistics'])
        assert not stat['recycle_events'] and not any(g['positive'] for g in stat['gate_events'])
        for k in ('native_risk_solver_calls','native_feature_seconds','static_seconds','static_head_calls',
                  'shared_static_hits','semantic_signature_seconds','gradient_probe_calls','probe_calls'):stat[k]=0
        stat.update(full_joint_events=[],full_gate_events=copy.deepcopy(stat['gate_events']),
            full_class_decisions=copy.deepcopy(stat['class_decisions']),total_proposals=0,total_accepted=0,
            terminal_eos_accepts=0,full_event_log_includes_terminal_eos=True,signal_guided=True,
            collective_visual_support_veto=True,full_collective_decisions=[],collective_vetoed_steps=0,
            cached_gate_decisions=len(stat['gate_events']),event_log_from_certified_cache=True)
        rows.append(dict(ref,variant=NAME,seconds=0.,batch_wall_seconds=0.,sar_ems_statistics=stat,
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
    from features.qwen3_collective_visual_gate import Qwen3BatchCollectiveVisualGate
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
    path=OUT/f'coco100/{NAME}.shard{gpu}.jsonl';assert not path.exists()
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py');verified=False;memory=[]
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group]
            if index in eligible:
                assert eligible[index]['image_ids']==ids
                rows=cache_rows(cached[index],plan,gpu,index)
            else:
                (span,inputs),=common.same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
                assert [i for i,_ in inputs]==ids and span[1]-span[0]==group[0]['visual_token_count']
                if not verified:
                    tokens,_,_=common.generate_group(wrapper,span,inputs,allowed,'sadi',32)
                    assert tokens==[r['response_token_ids'][:32] for r in group]
                    shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,ids=ids,native32_exact=True))
                    gc.collect();torch.cuda.empty_cache();verified=True
                holder={}
                def factory(model,current_span,nouns,mode,strength):
                    controller=Qwen3BatchCollectiveVisualGate(model,current_span,allowed,probe=probe,
                        semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                        real_threshold=plan['real_threshold'],signal_guided=True,gamma=1.4)
                    holder['controller']=controller;return controller
                torch.cuda.reset_peak_memory_stats(wrapper.device)
                with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                    tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
                calls=holder['controller'].steps;holder.clear()
                texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                rows=[dict(image_id=i,model=MODEL,variant=NAME,response_token_ids=t,generated_text=text.strip(),
                    seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),
                    visual_token_count=span[1]-span[0],sar_ems_statistics=stat,
                    decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                    actual_model_calls_in_batch=calls) for i,t,text,stat in zip(ids,tokens,texts,stats,strict=True)]
            with path.open('a',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            gc.collect();torch.cuda.empty_cache()
            memory.append(dict(group=index,allocated=torch.cuda.memory_allocated(wrapper.device),
                peak=torch.cuda.max_memory_allocated(wrapper.device),reused=rows[0]['generation_reused']))
            shared.write(OUT/f'memory.gpu{gpu}.json',memory)
            print(f'[{gpu}] group={index} B={len(ids)} seconds={rows[0]["batch_wall_seconds"]:.3f}',flush=True)
    assert shared.group_keys(source_groups(path))==shared.group_keys(groups)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
