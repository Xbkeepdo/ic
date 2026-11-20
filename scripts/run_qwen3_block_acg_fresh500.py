"""Locked joint ACG adaptation and signal guidance on500 unused COCO images."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.run_qwen3_semantic_final_pilot import build_wrapper

SOURCE=shared.OUT/'qwen3_block_acg_pilot_20261118'
CONTROL=shared.OUT/'qwen3_headroll_pilot_20261118_cache_recovery_v2'
OUT=shared.OUT/'qwen3_block_acg_fresh500_20261118'
VARIANTS=['sadi','svar','acg_fixed','acg_guided']
SEED=2026111801
FILES=['scripts/run_qwen3_block_acg_fresh500.py','scripts/audit_qwen3_block_acg_fresh.py',
       'features/qwen3_block_acg_headroll.py','scripts/audit_qwen3_caption_with_native_cache_v2.py',
       'scripts/audit_qwen3_caption_denominator.py',
       'scripts/experiment_snapshots/qwen3_new200_b16/prepare.py']


def select_images():
    from scripts.experiment_snapshots.qwen3_new200_b16.prepare import cohort_ids,record_ids
    coco=REPO.parent/'dataset/coco'
    all_ids={int(p.stem.rsplit('_',1)[-1]) for p in (coco/'val2014').glob('COCO_val2014_*.jpg')}
    assert len(all_ids)==40504
    excluded={485532,558955}&all_ids;sources=[];seen=set()
    def add(path,kind,json_lines=False):
        resolved=path.resolve()
        if resolved in seen:return
        seen.add(resolved);body=path.read_bytes()
        if not body.strip():return
        if kind=='captions':
            value=json.loads(body);assert isinstance(value,dict) and all(str(k).isdigit() for k in value)
            ids=set(map(int,value))
        elif kind=='cohort':ids=cohort_ids(json.loads(body))
        elif json_lines:
            ids=set()
            for line in body.splitlines():
                if line.strip():ids.update(record_ids(json.loads(line)))
        else:ids=record_ids(json.loads(body))
        matched=ids&all_ids;excluded.update(matched)
        digest=shared.sha(path)
        sources.append(dict(path=str(path),file_sha256=digest,kind=kind,in_val2014=len(matched),
            matched_ids_sha256=ids_sha(matched)))
    for base in [REPO.parent/'ic/outputs',shared.OUT]:
        for p in sorted(base.glob('*/COCO*/generations.json')):add(p,'captions')
        for p in sorted(base.glob('*/cohort.json')):add(p,'cohort')
        # Manifests cover running cohorts. Read only orphan raw caption files;
        # do not pin the SHA of an actively appended file covered by a manifest.
        for p in sorted(base.glob('*/*/*.shard*.jsonl')):
            if not (p.parent.parent/'cohort.json').exists():add(p,'orphan_records',True)
    for p in sorted((REPO.parent/'dataset/pope').glob('coco_pope_*.json')):add(p,'pope',True)
    add(REPO.parent/'dataset/qa_sam_svar_base/pope/questions.jsonl','pope',True)
    pool=sorted(all_ids-excluded)
    ids=sorted(np.random.default_rng(SEED).choice(pool,500,replace=False).tolist())
    assert len(set(ids))==500 and not set(ids)&excluded
    return ids,dict(seed=SEED,rng='NumPy PCG64',sources=sources,available_images=len(all_ids),
        pool_size=len(pool),excluded_ids=sorted(excluded),sample_sha256=ids_sha(ids),
        scope='COCO4000, all prior ic/ic-ac cohort manifests, orphan caption records, POPE and diagnostic image IDs.',
        inspected_new_outputs_before_selection=False)


def ids_sha(ids):
    import hashlib
    return hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()


def prepare():
    assert not OUT.exists(),'Inspect existing output; never resample an existing cohort.'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert shared.read(SOURCE/'screening.json')['advance_to_500']==['sadi_sam_acg_guided_19_35']
    assert shared.read(CONTROL/'post_audit_status.json')['status']=='passed'
    assert shared.read(CONTROL/'human_review.json')['advance_frozen_main_to_500']
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    coverage=shared.OUT/'qwen3_native_gate_coverage_20261118/coverage_provenance.json'
    hashes.update(shared.read(coverage)['supplemental_inputs'])
    for p in [SOURCE/'protocol.json',SOURCE/'human_review.json',SOURCE/'final_provenance.json',
              CONTROL/'protocol.json',CONTROL/'post_audit_status.json',CONTROL/'human_review.json',
              coverage,*(REPO/n for n in FILES)]:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    ids,selection=select_images()
    for row in selection['sources']:assert shared.sha(Path(row['path']))==row['file_sha256']
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco500').mkdir()
    shared.write(OUT/'selection.json',selection);shared.write(OUT/'cohort.json',dict(coco500=ids))
    shutil.copyfile(SOURCE/'object_lexicon.json',OUT/'object_lexicon.json')
    plan=dict(stage='independent-block500',model='Qwen3-VL-8B-Instruct',images=500,shards=2,cohort='coco500',
        max_batch=16,max_new_tokens=512,prompt='Describe this image.',variants=VARIANTS,
        candidates=['acg_fixed','acg_guided'],cached_controls=['sadi'],copied_controls=[],
        source=str(SOURCE),source_sha256=hashes,fixed_at=time.time(),sample_sha256=ids_sha(ids),
        probe=old['probe'],real_threshold=old['real_threshold'],seed=SEED,
        recipe=old['recipe'],candidate_mapping=dict(acg_fixed='sadi_sam_acg_fixed_19_35',acg_guided='sadi_sam_acg_guided_19_35'),
        generation='BF16/eager/greedy; sorted IDs, stride2 shards; all four methods use the same realized ordered batches.',
        parameters='Complete SADI5--18 and joint19--35, gamma1.4, original108 SAM-V2 gate. No refit or test-set tuning.',
        shared_cache='Original image-only semantics, with full native image-prefix verification, shared across fixed/guided within one batch; RAM only.',
        numerical_logging='Full proposal/gate/history logs include terminal EOS. Cache does not skip any image or method.',
        failure_policy='Preserve failed outputs and original ordered groups; no automatic OOM split or rebatched retry.',
        primary='Sentence CHAIRs/CHAIRi of guided versus SADI; also SVAR, fixed, F1, recall, counts, all paired intervals and legacy parser.',
        inference='One frozen prospective validation; inherited screening.json is a numeric summary, not a test-time parameter selection rule.')
    shared.write(OUT/'protocol.json',plan)
    print('LOCKED_INDEPENDENT500',selection['sample_sha256'],'pool',selection['pool_size'],flush=True)
    return plan


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
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
    files={n:OUT/f'coco500/{n}.shard{gpu}.jsonl' for n in VARIANTS}
    assert not any(p.exists() for p in files.values())
    with torch.inference_mode():
        for index,(span,group) in enumerate(common.same_span_batches(wrapper,ids,COCO_ROOT,16)):
            image_ids=[i for i,_ in group];static={}
            for name in VARIANTS:
                holder={}
                if name in ('sadi','svar'):
                    tokens,stats,seconds=common.generate_group(wrapper,span,group,nouns,name,512)
                    calls=None
                else:
                    def factory(model,current_span,_nouns,mode,strength):
                        c=Qwen3BatchBlockACGLogged(model,current_span,allowed,probe=probe,semantic_tokens=sorted(allowed),
                            tokenizer=wrapper.tokenizer,history_parser=parser,real_threshold=plan['real_threshold'],
                            signal_guided=name=='acg_guided',gamma=1.4,shared_static=static)
                        holder['controller']=c;return c
                    with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                        tokens,stats,seconds=original.generate_group(wrapper,span,group,allowed,shared.POSITIVE,512)
                    calls=holder['controller'].steps;holder.clear()
                texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                with files[name].open('a',encoding='utf-8') as f:
                    for image,t,text,stat in zip(image_ids,tokens,texts,stats,strict=True):
                        f.write(json.dumps(dict(image_id=image,model=MODEL,variant=name,response_token_ids=t,generated_text=text.strip(),
                            seconds=seconds/len(group),batch_wall_seconds=seconds,batch_size=len(group),visual_token_count=span[1]-span[0],
                            sar_ems_statistics=stat,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                            actual_model_calls_in_batch=calls),ensure_ascii=False)+'\n')
                del stats;gc.collect();torch.cuda.empty_cache()
                print(f'[{gpu}] group={index} {name} B={len(group)} seconds={seconds:.3f}',flush=True)
            static.clear();gc.collect();torch.cuda.empty_cache()
    for p in files.values():assert {r['image_id'] for g in shared.source_groups(p) for r in g}==set(ids)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
