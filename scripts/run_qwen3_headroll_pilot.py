"""One head-association control, reusing certified complete no-gate batches."""

import argparse
import copy
import gc
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.run_qwen3_semantic_final_pilot import build_wrapper

SOURCE=shared.OUT/'qwen3_block_acg_pilot_20261118'
OUT=shared.OUT/'qwen3_headroll_pilot_20261118'
NAME='headroll'
GUIDED='sadi_sam_acg_guided_19_35'
CONTROLS=['sadi','svar','sadi_sam_acg_fixed_19_35',GUIDED]
CPU=shared.OUT/'qwen3_block_headroll_cpu/verification.json'
PREFLIGHT=shared.OUT/'qwen3_headroll_preflight/verification.json'
CERTIFICATE=SOURCE/'no_gate_cache_validation.json'
FILES=['features/qwen3_block_acg_headroll.py','scripts/verify_qwen3_block_headroll.py',
 'scripts/verify_qwen3_headroll_gpu.py','scripts/run_qwen3_headroll_pilot.py',
 'scripts/audit_qwen3_headroll_pilot.py','scripts/audit_qwen3_caption_with_native_cache.py',
 'scripts/audit_qwen3_no_gate_cache.py']


def prepare():
    assert not OUT.exists()
    assert shared.read(SOURCE/'post_audit_status.json')['status']=='passed'
    assert shared.read(SOURCE/'human_review.json')['advance_to_head_control']
    assert shared.read(CPU)['passed'] and shared.read(PREFLIGHT)['passed']
    for n,v in shared.read(PREFLIGHT)['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    certificate=shared.read(CERTIFICATE);assert certificate['passed'] and certificate['joint17_agreement_verified']
    for n,v in certificate['source_sha256'].items():assert shared.sha(Path(n))==v,n
    for p in [CPU,PREFLIGHT,CERTIFICATE,SOURCE/'protocol.json',SOURCE/'human_review.json',*(REPO/n for n in FILES)]:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for n in ('cohort.json','object_lexicon.json'):shutil.copyfile(SOURCE/n,OUT/n)
    for shard in (0,1):
        for name in CONTROLS:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';hashes[str(p)]=shared.sha(p)
            shutil.copyfile(p,OUT/'coco100'/p.name)
    plan=dict(stage='head-association-control',images=100,shards=2,cohort='coco100',max_batch=16,max_new_tokens=512,
        variants=[*CONTROLS,NAME],candidates=[NAME],copied_controls=CONTROLS,cached_controls=['sadi'],source=str(SOURCE),
        source_sha256=hashes,fixed_at=time.time(),probe=old['probe'],real_threshold=old['real_threshold'],
        cache_certificate=str(CERTIFICATE),cache_certificate_sha256=shared.sha(CERTIFICATE),
        recipe=dict(base_layers=[5,18],joint_layers=list(range(19,36)),gamma=1.4,head_roll=10,
            change='Roll the same native per-layer scale vector by H//3; preserve its multiset and leave Q/D/gradients/contrast/gate/acceptance unchanged.',
            control_prediction='May be positive before actual acceptance; still retain only actual original-target probability decreases.',
            scope='Same-state head-scale multiset matched; context-vector dose and trajectories need not match.',
            complete_logs='Include terminal-EOS accepted events in separate full logs, so a truncated caption cannot falsely certify native reuse.',
            cache='Only the seven original complete ordered batches whose zero native gate and tokens were independently verified in both single/joint implementations. No rebatched rows.',
            cache_timing='Cached rows have0 model-stage seconds/actual model calls; cache I/O recorded separately. Raw decoder marks reuse and pins source+certificate SHA.'),
        development_screen='This is a head-association control. Screened names do not automatically replace or tune the frozen main candidate.',
        followup='Use the control to assess head alignment; keep main recipe fixed for an unseen500 comparison with SADI/SVAR/fixed adaptation.')
    shared.write(OUT/'protocol.json',plan);return plan


def worker(gpu):
    import json
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGHeadRoll
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():assert shared.sha(Path(n))==v,n
    cert=shared.read(CERTIFICATE);eligible={e['group']:e for e in cert['eligible_groups'] if e['shard']==gpu}
    groups=shared.source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    source_path=SOURCE/f'coco100/{GUIDED}.shard{gpu}.jsonl'
    source_groups=shared.source_groups(source_path)
    assert shared.group_keys(source_groups)==shared.group_keys(groups)
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    path=OUT/f'coco100/{NAME}.shard{gpu}.jsonl';assert not path.exists()
    verified=False
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group]
            if index in eligible:
                assert eligible[index]['image_ids']==ids
                start=time.perf_counter();rows=[]
                for ref in source_groups[index]:
                    stat=copy.deepcopy(ref['sar_ems_statistics'])
                    assert not stat['recycle_events'] and not any(g['positive'] for g in stat['gate_events'])
                    for key in ('native_risk_solver_calls','native_feature_seconds','static_seconds','static_head_calls','shared_static_hits','semantic_signature_seconds','gradient_probe_calls','probe_calls'):stat[key]=0
                    stat.update(full_joint_events=[],full_gate_events=copy.deepcopy(stat['gate_events']),
                        full_class_decisions=copy.deepcopy(stat['class_decisions']),total_proposals=0,total_accepted=0,terminal_eos_accepts=0,
                        full_event_log_includes_terminal_eos=True,permuted_head_control=True,head_scale_roll_by=10,
                        cached_gate_decisions=len(stat['gate_events']),event_log_from_certified_cache=True)
                    row=dict(ref,variant=NAME,seconds=0.,batch_wall_seconds=0.,sar_ems_statistics=stat,
                        decoder='cached_native_no_positive_gate',generation_reused=True,actual_model_calls_in_batch=0,
                        cache_origin=dict(source=str(source_path),source_sha256=shared.sha(source_path),
                            certificate=str(CERTIFICATE),certificate_sha256=plan['cache_certificate_sha256'],shard=gpu,group=index))
                    rows.append(row)
                cache_io=time.perf_counter()-start
                for row in rows:row['cache_io_seconds']=cache_io/len(rows)
                with path.open('a',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(f'[{gpu}] cached complete group={index} batch={len(ids)} model_calls=0',flush=True)
                continue
            (span,inputs),=common.same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
            assert [i for i,_ in inputs]==ids
            if not verified:
                tokens,_,_=original.generate_group(wrapper,span,inputs,allowed,'sadi',32)
                assert tokens==[r['response_token_ids'][:32] for r in group]
                shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,ids=ids,batch=len(ids),native32_exact=True))
                gc.collect();torch.cuda.empty_cache();verified=True
            holder={}
            def factory(model,current_span,nouns,mode,strength):
                c=Qwen3BatchBlockACGHeadRoll(model,current_span,allowed,probe=probe,semantic_tokens=sorted(allowed),
                    tokenizer=wrapper.tokenizer,history_parser=parser,real_threshold=plan['real_threshold'],signal_guided=True,gamma=1.4)
                holder['controller']=c;return c
            with patch.object(original,'Qwen3BatchLayerFeedback',factory):tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
            actual_calls=holder['controller'].steps;holder.clear()
            texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            with path.open('a',encoding='utf-8') as f:
                for image,t,text,stat in zip(ids,tokens,texts,stats,strict=True):
                    f.write(json.dumps(dict(image_id=image,model=MODEL,variant=NAME,response_token_ids=t,generated_text=text.strip(),
                        seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),visual_token_count=span[1]-span[0],
                        sar_ems_statistics=stat,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                        actual_model_calls_in_batch=actual_calls),ensure_ascii=False)+'\n')
            gc.collect();torch.cuda.empty_cache()
            print(f'[{gpu}] fresh group={index} batch={len(ids)} seconds={seconds:.3f}',flush=True)
    assert shared.group_keys(shared.source_groups(path))==shared.group_keys(groups)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
