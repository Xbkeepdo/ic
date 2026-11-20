"""Exact rollback tests, then complete captions on the two original groups."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.qwen3_caption_groups_with_cache import source_groups
from scripts.run_qwen3_region_guidance import setup,batch,generate

SOURCE=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
PREVIOUS=REPO/'outputs/qwen3_supported_rivals_probe_20261120'
MODES=['calibrated','calibrated_shifted']


def prepare(root):
    assert not root.exists() and read(PREVIOUS/'pipeline_complete.json')['passed']
    old=read(PREVIOUS/'protocol.json');base=read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    files=[Path(__file__),REPO/'features/qwen3_supported_rival_guidance.py',
        REPO/'scripts/evaluate_qwen3_region_caption_chair.py',
        REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',
        PREVIOUS/'protocol.json',PREVIOUS/'results0.json',PREVIOUS/'results1.json']
    for name in ['sadi','svar']:
        for shard in [0,1]:files.append(SOURCE/f'coco100/{name}.shard{shard}.jsonl')
    hashes.update({str(p):sha(p) for p in files})
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    root.mkdir();(root/'coco10').mkdir();(root/'source').mkdir()
    for p in files:
        if p.suffix=='.py':(root/'source'/p.name).write_bytes(p.read_bytes())
    groups=old['groups'];ids=sorted(i for g in groups for i in g['ids']);assert len(set(ids))==len(ids)==10
    write(root/'cohort.json',dict(coco10=ids));(root/'object_lexicon.json').write_bytes((SOURCE/'object_lexicon.json').read_bytes())
    for name in ['sadi','svar']:
        source=[g for s in [0,1] for g in source_groups(SOURCE/f'coco100/{name}.shard{s}.jsonl')]
        for shard,g in enumerate(groups):
            rows,=[r for r in source if [item['image_id'] for item in r]==g['ids']]
            with (root/f'coco10/{name}.shard{shard}.jsonl').open('x',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for path in [root/'cohort.json',root/'object_lexicon.json',*sorted((root/'coco10').glob('*.jsonl'))]:hashes[str(path)]=sha(path)
    for image in ids:
        path=REPO.parent/'dataset/coco/val2014'/f'COCO_val2014_{image:012d}.jpg';hashes[str(path)]=sha(path)
    write(root/'protocol.json',dict(fixed_at=time.time(),source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),
        source_sha256=hashes,groups=groups,cohort='coco10',images=10,shards=2,max_batch=16,max_new_tokens=512,
        variants=['sadi','svar',*MODES],candidates=MODES,copied_controls=[],probe=base['probe'],real_threshold=base['real_threshold'],
        controls='Original complete ordered6/4 groups filtered from frozen100, tokens and metadata retained; source hashes fixed. No new baseline generation except128-token equivalence checks.',
        generation='Complete captions for calibrated and spatially shifted control, cap512; unchanged full SADI5--18.',
        construction='Frozen supported-rivals calibrated formula: native class mass * write/whole-image support lift; positive normalized class contrast times writes gives ROI; strength5*D*relative lift gap.',
        preflight='Native128, disabled128, zero128, reject128 exact tokens/logits/all36 current and final fullKV; then two live512 runs.',
        purpose='Implementation and complete-caption screening on old10 development images. If no useful physical correction, do not spend another full100. No500 decision from10 alone.'))


def worker(root,shard):
    import torch
    from features.qwen3_supported_rival_guidance import Qwen3SupportedRivalGuidance
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    plan,wrapper,probe,allowed,parser=setup(root);ids=plan['groups'][shard]['ids']
    span,inputs,photos=batch(wrapper,ids);shared={};reports=[];started=time.time()
    expected=[r['response_token_ids'] for g in source_groups(root/f'coco10/sadi.shard{shard}.jsonl') for r in g]
    with torch.inference_mode():
        native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'),128,True)
        assert native[0]==[t[:128] for t in expected]
        write(root/f'native128_proof{shard}.json',dict(tokens=native[0],per_step=native[1],full_cache_sha256=native[2]))
        for mode in ['disabled','zero','reject',*MODES]:
            diagnostic=mode in ['disabled','zero','reject'];cap=128 if diagnostic else 512
            controller=Qwen3SupportedRivalGuidance(wrapper,span,allowed,probe=probe,history_parser=parser,image_ids=ids,images=photos,
                output_dir=root/'unused_crops',shared_cache=shared,policy='calibrated' if diagnostic else mode,
                disabled=mode=='disabled',zero_strength=mode=='zero',reject_all=mode=='reject')
            result=generate(wrapper,inputs,controller,cap,diagnostic);stats=controller.statistics(result[0])
            events=[e for s in stats for e in s['full_events']]
            if diagnostic:
                assert result[:3]==native[:3],mode
                assert not any(s['accepted'] for s in stats)
                if shard==1 and mode in ['zero','reject']:assert events
                if mode=='zero':assert all(e['proposed_current_kv_max_difference']==0 for e in events)
            else:
                texts=wrapper.processor.batch_decode(result[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                for s in stats:
                    s['guided_attention_calls']=len(s['full_events']);s['accepted_guidance_calls']=s['accepted']
                    assert s['crop_encoder_calls_in_batch']==s['fine_prefill_calls_in_batch']==0
                rows=[dict(image_id=i,model='qwen3_vl_8b',variant=mode,response_token_ids=t,generated_text=text.strip(),
                    seconds=result[3]/len(ids),batch_wall_seconds=result[3],batch_size=len(ids),visual_token_count=span[1]-span[0],
                    sar_ems_statistics=s,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                    actual_model_calls_in_batch=controller.step+1) for i,t,text,s in zip(ids,result[0],texts,stats,strict=True)]
                with (root/f'coco10/{mode}.shard{shard}.jsonl').open('x',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            entry=dict(mode=mode,cap=cap,diagnostic_exact=diagnostic,proposals=len(events),accepted=sum(e['accepted'] for e in events),
                changed_token_images=sum(a!=b[:cap] for a,b in zip(result[0],expected,strict=True)),seconds=result[3],stats=stats)
            reports.append(entry);write(root/f'progress{shard}.json',reports)
            print({k:entry[k] for k in ['mode','cap','diagnostic_exact','proposals','accepted','changed_token_images','seconds']},flush=True)
            controller.close();del controller,result,stats;gc.collect();torch.cuda.empty_cache()
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,cases=reports,source_sha256=plan['source_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:prepare(a.root) if a.prepare else worker(a.root,a.worker)
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
