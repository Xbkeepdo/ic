"""Preserve96 caption records; finish only uncompleted original ordered groups."""

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
from scripts import run_qwen3_headroll_pilot as original_run
from scripts.qwen3_caption_groups_with_cache import source_groups

FAILED=original_run.OUT
OUT=shared.OUT/'qwen3_headroll_pilot_20261118_cache_recovery_v2'
FILES=['scripts/qwen3_caption_groups_with_cache.py','scripts/resume_qwen3_headroll_pilot.py',
       'scripts/audit_qwen3_headroll_recovery.py','scripts/audit_qwen3_caption_with_native_cache_v2.py']


def prepare():
    assert not OUT.exists()
    assert shared.read(FAILED/'run_status.json')['status']=='failed'
    old=shared.read(FAILED/'protocol.json');hashes=dict(old['source_sha256'])
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    for p in [FAILED/'protocol.json',FAILED/'run_status.json',*(REPO/n for n in FILES)]:hashes[str(p)]=shared.sha(p)
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for name in ('cohort.json','object_lexicon.json'):shutil.copyfile(FAILED/name,OUT/name)
    prefixes={};counts={}
    for shard in (0,1):
        for name in old['variants']:
            p=FAILED/f'coco100/{name}.shard{shard}.jsonl'
            if not p.exists():continue
            groups=source_groups(p)
            if name==original_run.NAME:
                native=source_groups(FAILED/f'coco100/sadi.shard{shard}.jsonl')
                assert shared.group_keys(groups)==shared.group_keys(native[:len(groups)])
                prefixes[str(shard)]=dict(path=str(p),sha256=shared.sha(p),bytes=p.stat().st_size,
                    groups=len(groups),images=sum(map(len,groups)))
                counts[str(shard)]=sum(map(len,groups))
            hashes[str(p)]=shared.sha(p);shutil.copyfile(p,OUT/'coco100'/p.name)
    assert sum(counts.values())==96
    plan=dict(old,source_sha256=hashes,fixed_at=time.time(),recovery_from=str(FAILED),
        recovery_prefixes=prefixes,recovery_reason='Frozen native-only source_groups rejected explicit certified cache labels after96 records. Method, cache rules and batches unchanged.',
        recovery_policy='Preserve every existing byte; skip complete original groups, append only missing groups. Never split or reorder a group.')
    shared.write(OUT/'protocol.json',plan);return plan


def worker(gpu):
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():assert shared.sha(Path(n))==v,n
    groups=source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    path=OUT/f'coco100/{original_run.NAME}.shard{gpu}.jsonl'
    existing=source_groups(path);done=len(existing)
    assert shared.group_keys(existing)==shared.group_keys(groups[:done])
    ref=plan['recovery_prefixes'][str(gpu)];before=Path(ref['path']).read_bytes()
    assert path.read_bytes()==before
    if done==len(groups):
        shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,all_original_groups_already_complete=True,new_model_calls=0))
        return
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGHeadRoll
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    certificate=shared.read(original_run.CERTIFICATE)
    eligible={e['group'] for e in certificate['eligible_groups'] if e['shard']==gpu}
    # Every certified reuse group has already been written; no new cache path.
    assert not set(range(done,len(groups)))&eligible
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py');verified=False
    with torch.inference_mode():
        for index,group in enumerate(groups[done:],start=done):
            ids=[r['image_id'] for r in group]
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
            with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
            calls=holder['controller'].steps;holder.clear()
            texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            with path.open('a',encoding='utf-8') as f:
                for image,t,text,stat in zip(ids,tokens,texts,stats,strict=True):
                    f.write(json.dumps(dict(image_id=image,model=MODEL,variant=original_run.NAME,response_token_ids=t,generated_text=text.strip(),
                        seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),visual_token_count=span[1]-span[0],
                        sar_ems_statistics=stat,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                        actual_model_calls_in_batch=calls),ensure_ascii=False)+'\n')
            gc.collect();torch.cuda.empty_cache()
            print(f'[{gpu}] appended original group={index} B={len(ids)} seconds={seconds:.3f}',flush=True)
    assert path.read_bytes().startswith(before)
    assert shared.group_keys(source_groups(path))==shared.group_keys(groups)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
