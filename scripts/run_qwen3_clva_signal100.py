"""Matched original100 captions: full SADI followed by visual-anchor feedback."""

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
from scripts.run_qwen3_region_guidance import batch,generate

SOURCE=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
PREFLIGHT=REPO/'outputs/qwen3_clva_signal_preflight10_v2_20261120'
MODES=['clva','clva_fixed','clva_drift','clva_sar']


def check(plan):
    for name,expected in plan['source_sha256'].items():assert sha(REPO/name)==expected,name


def key(group):
    return tuple(row['image_id'] for row in group),group[0]['visual_token_count']


def prepare(root):
    assert not root.exists(),'Preserve existing runs; inspect instead of overwriting.'
    pre=read(PREFLIGHT/'protocol.json')
    assert read(PREFLIGHT/'pipeline_complete.json')['passed']
    assert read(PREFLIGHT/'analysis_provenance.json')['passed']
    for shard in [0,1]:assert read(PREFLIGHT/f'worker{shard}.json')['passed']
    check(pre)
    paths=[Path(__file__),REPO/'scripts/analyze_qwen3_clva_signal100.py',
        REPO/'scripts/analyze_qwen3_clva_signal.py',SOURCE/'protocol.json',SOURCE/'cohort.json',SOURCE/'object_lexicon.json',
        PREFLIGHT/'protocol.json',PREFLIGHT/'pipeline_complete.json',PREFLIGHT/'analysis_provenance.json']
    paths += [SOURCE/f'coco100/{mode}.shard{s}.jsonl' for mode in ['sadi','svar'] for s in [0,1]]
    paths += [PREFLIGHT/f'coco10/{mode}.shard{s}.jsonl' for mode in MODES for s in [0,1]]
    paths += [PREFLIGHT/f'worker{s}.json' for s in [0,1]]
    hashes=dict(pre['source_sha256']);hashes.update({str(p):sha(p) for p in paths})
    groups={str(s):[dict(ids=list(key(g)[0]),visual_tokens=key(g)[1])
        for g in source_groups(SOURCE/f'coco100/sadi.shard{s}.jsonl')] for s in [0,1]}
    available={mode:{key(g):(s,g) for s in [0,1] for g in source_groups(PREFLIGHT/f'coco10/{mode}.shard{s}.jsonl')} for mode in MODES}
    reused=[]
    for shard,shard_groups in groups.items():
        for index,g in enumerate(shard_groups):
            k=tuple(g['ids']),g['visual_tokens']
            for mode in MODES:
                if k in available[mode]:
                    original_shard,_=available[mode][k]
                    path=PREFLIGHT/f'coco10/{mode}.shard{original_shard}.jsonl'
                    reused.append(dict(shard=int(shard),group=index,mode=mode,ids=g['ids'],visual_tokens=g['visual_tokens'],
                        source=str(path),source_sha256=sha(path)))
    assert len(reused)==8 and sum(len(g['ids']) for g in reused)==40
    plan=dict(source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),source_sha256=hashes,
        fixed_at=time.time(),cohort='coco100',images=100,shards=2,max_batch=16,max_new_tokens=512,
        groups=groups,variants=['sadi','svar',*MODES],candidates=MODES[1:],copied_controls=['sadi','svar'],
        preflight=str(PREFLIGHT),preflight_protocol_sha256=sha(PREFLIGHT/'protocol.json'),
        base_layers=list(range(5,19)),raw_recipe=pre['raw_recipe'],modes=pre['modes'],
        feedback=pre['feedback'],acceptance=pre['acceptance'],differences=pre['differences'],
        generation_reuse_groups=reused,
        reuse_semantics='Exact whole original ordered B/P groups, same frozen module and cap512. Preserve original rows and times byte-for-byte as JSON objects, with explicit group receipts. No model calls for these40 precomputed image-arm pairs; row generation flags describe their original calls.',
        grouping='Original22 ordered groups; prompt/processor/weights/base/method settings unchanged.',
        comparison='Retain complete SADI5--18 in all new arms. Raw anchors vs SADI, fixed amplification vs raw, adaptive drift vs fixed, and SAR anchor vs adaptive drift.',
        selection='Reused exploratory100, not independent validation; all arms retained, no new parameters chosen using this cohort.',
        followup='Review both CHAIR metrics, false counts, recall/F1, repetitions and physical object corrections. Useful full100 candidate only then independent reserved500, no validation tuning.')
    check(plan)
    root.mkdir();(root/'coco100').mkdir();(root/'source').mkdir()
    for name in ['cohort.json','object_lexicon.json']:(root/name).write_bytes((SOURCE/name).read_bytes())
    for name in ['sadi','svar']:
        for s in [0,1]:
            path=SOURCE/f'coco100/{name}.shard{s}.jsonl';(root/'coco100'/path.name).write_bytes(path.read_bytes())
    for path in paths:
        if path.suffix=='.py':(root/'source'/path.name).write_bytes(path.read_bytes())
    write(root/'protocol.json',plan)
    print(dict(prepared=True,groups={s:len(v) for s,v in groups.items()},reused_image_arm_pairs=40),flush=True)


def worker(root,shard):
    import torch
    from features.qwen3_clva_signal import Qwen3CLVASignal
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    plan=read(root/'protocol.json');check(plan)
    wrapper=build_wrapper();allowed=frozenset(read(root/'object_lexicon.json')['token_ids'])
    groups=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
    started=time.time();reports=[];receipts=[];checked_large=False
    streams={m:(root/f'coco100/{m}.shard{shard}.jsonl').open('x',encoding='utf-8') for m in MODES}
    try:
        with torch.inference_mode():
            for index,group in enumerate(groups):
                ids=list(key(group)[0]);reuse=[r for r in plan['generation_reuse_groups'] if r['shard']==shard and r['group']==index]
                if reuse:
                    assert len(reuse)==len(MODES)
                    for entry in reuse:
                        source=Path(entry['source']);assert sha(source)==entry['source_sha256']
                        matches=[g for g in source_groups(source) if key(g)==key(group)];original,=matches
                        for row in original:streams[entry['mode']].write(json.dumps(row,ensure_ascii=False)+'\n')
                        streams[entry['mode']].flush();receipts.append(entry)
                    reports.append(dict(group=index,ids=ids,reused=True,image_arm_pairs=len(ids)*len(MODES)))
                    write(root/f'progress{shard}.json',reports);continue
                span,inputs,photos=batch(wrapper,ids);del photos
                assert span[1]-span[0]==key(group)[1]
                shared={};checks=[];large=not checked_large
                if large:
                    def harvest(_m,args,out):
                        assert '_vision' not in shared
                        shared['_vision']=dict(pixels=args[0].detach().clone(),grid=inputs['image_grid_thw'].detach().clone(),
                            output=(out[0].detach().clone(),[x.detach().clone() for x in out[1]]))
                    handle=wrapper.model.model.visual.register_forward_hook(harvest)
                    try:native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'),128,True)
                    finally:handle.remove()
                    assert native[0]==[r['response_token_ids'][:128] for r in group]
                    disabled=Qwen3CLVASignal(wrapper,span,allowed,shared_static=shared,disabled=True)
                    current=generate(wrapper,inputs,disabled,128,True);assert current[:3]==native[:3]
                    checks.append(dict(mode='disabled',passed=True,exact_sadi=True))
                    disabled.close();del disabled,current,native;gc.collect();torch.cuda.empty_cache()
                    raw=Qwen3CLVASignal(wrapper,span,allowed,mode='clva',shared_static=shared)
                    reference=generate(wrapper,inputs,raw,128,True);raw.close();del raw
                    for mode in ['zero','reject']:
                        controller=Qwen3CLVASignal(wrapper,span,allowed,mode='clva_sar',shared_static=shared,
                            zero_guidance=mode=='zero',reject_all=mode=='reject')
                        current=generate(wrapper,inputs,controller,128,True)
                        assert current[:3]==reference[:3],mode
                        statistics=controller.statistics(current[0]);events=[e for s in statistics for e in s['full_events']]
                        assert events,mode
                        checks.append(dict(mode=mode,passed=True,exact_raw_clva=True,events=len(events)))
                        controller.close();del controller,current,statistics;gc.collect();torch.cuda.empty_cache()
                    write(root/f'large_batch_preflight{shard}.json',dict(passed=True,ids=ids,checks=checks,
                        tokens=reference[0],per_step=reference[1],full_cache_sha256=reference[2]))
                    del reference;checked_large=True
                mode_reports=[]
                for mode in MODES:
                    controller=Qwen3CLVASignal(wrapper,span,allowed,mode=mode,shared_static=shared)
                    result=generate(wrapper,inputs,controller,512,False);statistics=controller.statistics(result[0])
                    texts=wrapper.processor.batch_decode(result[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    for image,tokens,text,stats in zip(ids,result[0],texts,statistics,strict=True):
                        row=dict(image_id=image,model='qwen3_vl_8b',variant=mode,response_token_ids=tokens,generated_text=text.strip(),
                            seconds=result[3]/len(ids),batch_wall_seconds=result[3],batch_size=len(ids),visual_token_count=span[1]-span[0],
                            sar_ems_statistics=stats,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                            actual_model_calls_in_batch=controller.step+1)
                        streams[mode].write(json.dumps(row,ensure_ascii=False)+'\n')
                    streams[mode].flush()
                    mode_reports.append(dict(mode=mode,seconds=result[3],feedback_rows=sum(len(s['full_events']) for s in statistics),
                        token_changes=sum(s['token_changes'] for s in statistics),caps=sum(len(t)==512 for t in result[0]),
                        original_encoder_calls=controller.encoder_calls,vision_cache_hits=controller.vision_cache_hits))
                    controller.close();del controller,result,statistics;gc.collect();torch.cuda.empty_cache()
                reports.append(dict(group=index,ids=ids,reused=False,checks=checks,modes=mode_reports))
                write(root/f'progress{shard}.json',reports);print(dict(shard=shard,group=index,images=len(ids),modes=mode_reports),flush=True)
                shared.clear();del shared,inputs;gc.collect();torch.cuda.empty_cache()
    finally:
        for stream in streams.values():stream.close()
    assert checked_large;check(plan)
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,groups=len(groups),
        new_images=sum(len(r['ids']) for r in reports if not r['reused']),reuse_receipts=receipts,cases=reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:prepare(args.root) if args.prepare else worker(args.root,args.worker)
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
