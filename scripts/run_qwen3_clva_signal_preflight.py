"""Full-SADI plus CLVA adaptation: baseline, constant and signal feedback."""

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
from scripts.run_qwen3_region_guidance import batch,generate,digest
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE=REPO/'outputs/qwen3_supported_rival_preflight10_20261120'
MATH=REPO/'outputs/qwen3_clva_signal_math_20261120.json'
MODES=['clva','clva_fixed','clva_drift','clva_sar']


def prepare(root):
    assert not root.exists() and read(SOURCE/'pipeline_complete.json')['passed']
    proof=read(MATH);assert proof['passed'] and not proof['cuda_initialized']
    old=read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    files=[Path(__file__),REPO/'features/qwen3_clva_signal.py',REPO/'scripts/verify_qwen3_clva_signal.py',
        REPO/'scripts/evaluate_qwen3_region_caption_chair.py',MATH,SOURCE/'protocol.json']
    for n,v in proof['source_sha256'].items():assert sha(Path(n))==v,n
    root.mkdir();(root/'coco10').mkdir();(root/'source').mkdir()
    for name in ['cohort.json','object_lexicon.json']:
        (root/name).write_bytes((SOURCE/name).read_bytes());files.extend([SOURCE/name,root/name])
    for name in ['sadi','svar']:
        for shard in [0,1]:
            p=SOURCE/f'coco10/{name}.shard{shard}.jsonl';target=root/'coco10'/p.name
            target.write_bytes(p.read_bytes());files.extend([p,target])
    hashes.update({str(p):sha(p) for p in files})
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    for p in files:
        if p.suffix=='.py':(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source=str(SOURCE),source_sha256=hashes,groups=old['groups'],
        cohort='coco10',images=10,shards=2,max_batch=16,max_new_tokens=512,variants=['sadi','svar',*MODES],
        candidates=MODES[1:],copied_controls=['sadi','svar'],base_layers=list(range(5,19)),
        paper='https://arxiv.org/html/2603.25088v1',adaptation=True,
        raw_recipe=dict(positive_layer=16,negative_layer=0,action_layers=list(range(19,36)),alpha=14.,beta=.9,z=2.,head_sigma=1.,population_std=True),
        differences='Project Qwen3/full-SADI adaptation, not author-code reproduction. Normalize anchor maps before z scores; empty head sets give empty masks. Actions start19 to retain all SADI5--18; only last query is edited.',
        feedback='Actual rawCLVA final greedy token in fixed309 lexical vocabulary, all occurrences; no trained detector or class-history filter. Source16 original SAR/FFN HellingerD, tau_ffn=.03.',
        modes=dict(clva='Raw two-anchor method after fullSADI; one pass.',clva_fixed='Extra feedback with D=.5, raw positive mask.',
            clva_drift='Extra feedback alpha=14*(1+D), raw positive mask.',
            clva_sar='Same alpha; positive anchor=(1-D)*raw_anchor+D*Q_SAR, then same z mask.'),
        acceptance='Every eligible feedback proposal is committed, including same-token updates; no posterior target-logp decrease gate. Diagnostic reject-all restores current19--35 KV and logits.',
        cost='Only noun-eligible single-query tail19--35 feedback; no extra image encoding, model-wide forward, FFN edit or backward. Share vision/DeepStack and source16 semantics between modes after exact input/hpre/value/head checks.',
        checks='Disabled exactSADI128; rawCLVA512 simultaneously records128-prefix reference; zero-guidance and reject-all exactrawCLVA128, logits/all36 current/fullKV.',
        decision='Old10 screening only. If complete-caption physical improvement, original100 next, independent500 afterward. Do not interpret CLVA paper metrics as these experiments.'))


def worker(root,shard):
    import torch
    from features.qwen3_clva_signal import Qwen3CLVASignal
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    plan=read(root/'protocol.json')
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    wrapper=build_wrapper();ids=plan['groups'][shard]['ids'];span,inputs,_=batch(wrapper,ids)
    allowed=frozenset(read(root/'object_lexicon.json')['token_ids']);shared={};reports=[];started=time.time()
    baseline=[r['response_token_ids'] for g in source_groups(root/f'coco10/sadi.shard{shard}.jsonl') for r in g]
    def save(mode,tokens,stats,seconds,steps):
        texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        rows=[dict(image_id=i,model='qwen3_vl_8b',variant=mode,response_token_ids=t,generated_text=text.strip(),
            seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),visual_token_count=span[1]-span[0],
            sar_ems_statistics=s,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
            actual_model_calls_in_batch=steps) for i,t,text,s in zip(ids,tokens,texts,stats,strict=True)]
        with (root/f'coco10/{mode}.shard{shard}.jsonl').open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    def capture_vision(_m,_args,out):
        assert '_vision' not in shared
        shared['_vision']=dict(pixels=inputs['pixel_values'].detach().clone(),grid=inputs['image_grid_thw'].detach().clone(),
            output=(out[0].detach().clone(),[x.detach().clone() for x in out[1]]))
    with torch.inference_mode():
        handle=wrapper.model.model.visual.register_forward_hook(capture_vision)
        try:native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'),128,True)
        finally:handle.remove()
        assert native[0]==[t[:128] for t in baseline]
        disabled=Qwen3CLVASignal(wrapper,span,allowed,shared_static=shared,disabled=True)
        check=generate(wrapper,inputs,disabled,128,True);assert check[:3]==native[:3]
        reports.append(dict(mode='disabled',passed=True,exact_sadi=True,seconds=check[3]));disabled.close();del disabled,check,native
        gc.collect();torch.cuda.empty_cache()
        controller=Qwen3CLVASignal(wrapper,span,allowed,mode='clva',shared_static=shared)
        snapshot={};seen=[]
        def at128(_m,_args,out):
            seen.append(1)
            if len(seen)==128:snapshot['full']=digest([t for layer in out.past_key_values.layers for t in [layer.keys,layer.values]])
        handle=wrapper.model.register_forward_hook(at128)
        try:raw=generate(wrapper,inputs,controller,512,True)
        finally:handle.remove()
        raw_tokens=[t[:128] for t in raw[0]];raw_trace=raw[1][:128];raw_full=snapshot.get('full',raw[2])
        stats=controller.statistics(raw[0]);save('clva',raw[0],stats,raw[3],controller.step+1)
        write(root/f'raw128_proof{shard}.json',dict(tokens=raw_tokens,per_step=raw_trace,full_cache_sha256=raw_full))
        reports.append(dict(mode='clva',passed=True,seconds=raw[3],tokens_per_image=[len(t) for t in raw[0]]))
        controller.close();del controller,stats,raw;gc.collect();torch.cuda.empty_cache()
        for mode in ['zero','reject',*MODES[1:]]:
            diagnostic=mode in ['zero','reject'];cap=128 if diagnostic else 512
            controller=Qwen3CLVASignal(wrapper,span,allowed,mode='clva_sar' if diagnostic else mode,shared_static=shared,
                zero_guidance=mode=='zero',reject_all=mode=='reject')
            current=generate(wrapper,inputs,controller,cap,diagnostic);stats=controller.statistics(current[0])
            events=[e for s in stats for e in s['full_events']]
            if diagnostic:
                assert current[0]==raw_tokens and current[1]==raw_trace and current[2]==raw_full,mode
                assert events
                if mode=='reject':assert not any(e['committed'] for e in events)
            else:save(mode,current[0],stats,current[3],controller.step+1)
            reports.append(dict(mode=mode,passed=True,exact_raw_clva=diagnostic,seconds=current[3],feedback_rows=len(events),
                token_changes=sum(e['token_changed'] for e in events),model_steps=controller.step+1))
            write(root/f'progress{shard}.json',reports);print(reports[-1],flush=True)
            controller.close();del controller,current,stats;gc.collect();torch.cuda.empty_cache()
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,cases=reports,original_encoder_calls=1,
        source_sha256=plan['source_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:prepare(a.root) if a.prepare else worker(a.root,a.worker)
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
