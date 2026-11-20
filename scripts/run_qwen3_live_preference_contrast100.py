"""Full100 FFN preference versus equal-strength spatial controls."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE = REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
PREVIOUS = REPO/'outputs/qwen3_live_preference_contrast10_20261120'
MODES = ['p_fixed', 'uniform_fixed', 'rolled_fixed']


def check(plan):
    for name, wanted in plan['source_sha256'].items(): assert sha(REPO/name) == wanted, name


def verify(destination):
    import torch
    from features.qwen3_live_preference_fixed_controls import control_prior
    from features.qwen3_signal_preference_contrast import mix_visual
    generator=torch.Generator().manual_seed(43);checks=0
    for length in [11,32,192,300]:
        p=torch.rand(length,generator=generator).softmax(0)
        uniform=control_prior(p,'uniform_fixed');rolled=control_prior(p,'rolled_fixed')
        assert torch.equal(uniform,torch.full_like(p,1/length))
        assert torch.equal(rolled.roll(-(length//2)),p)
        assert torch.equal(uniform,control_prior(p.flip(0),'uniform_fixed'))
        a=torch.randn((3,4,1,length+8),generator=generator).softmax(-1)
        expected=a.clone()
        for row,prior in [(0,uniform),(2,rolled)]:
            actual=mix_visual(expected,(3,length+3),row,prior,.5)
            assert torch.equal(actual[1],a[1])
            assert torch.equal(actual[...,:3],a[...,:3]) and torch.equal(actual[...,length+3:],a[...,length+3:])
            assert torch.allclose(actual.sum(-1),a.sum(-1),rtol=1e-6,atol=1e-7)
            assert torch.equal(mix_visual(expected,(3,length+3),row,prior,0),expected)
            expected=actual;checks+=4
        checks+=3
    write(destination,dict(passed=True,checks=checks,cuda_initialized=torch.cuda.is_initialized(),
        source_sha256={str(REPO/n):sha(REPO/n) for n in ['features/qwen3_live_preference_fixed_controls.py',
            'features/qwen3_signal_preference_contrast.py',str(Path(__file__))]}))


def prepare(root, proof):
    assert not root.exists(),'Preserve existing runs; inspect rather than relaunch.'
    old=read(PREVIOUS/'protocol.json');source=read(SOURCE/'protocol.json');check(old);check(source)
    for name in ['pipeline_complete.json','analysis_recovery_complete.json','independent_summary.json','quality_complete.json']:
        assert read(PREVIOUS/name)['passed'],name
    assert read(PREVIOUS/'analysis_provenance.json')['recovery_request_sha256']==sha(PREVIOUS/'analysis_recovery_request.json')
    assert read(proof)['passed'] and not read(proof)['cuda_initialized']
    for name,wanted in read(proof)['source_sha256'].items():assert sha(Path(name))==wanted,name
    paths=[Path(__file__),REPO/'features/qwen3_live_preference_fixed_controls.py',
        REPO/'features/qwen3_live_preference_contrast.py',REPO/'scripts/run_qwen3_full_sadi_next6.py',
        REPO/'scripts/run_qwen3_region_guidance.py',REPO/'scripts/analyze_qwen3_clva_quality.py',
        REPO/'scripts/evaluate_caption_chair.py',REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',
        SOURCE/'protocol.json',SOURCE/'cohort.json',PREVIOUS/'protocol.json',PREVIOUS/'object_lexicon.json',
        PREVIOUS/'pipeline_complete.json',PREVIOUS/'analysis_recovery_request.json',PREVIOUS/'analysis_recovery_complete.json',
        PREVIOUS/'analysis_provenance.json',PREVIOUS/'independent_summary.json',PREVIOUS/'quality_complete.json',proof]
    paths += [SOURCE/f'coco100/{n}.shard{s}.jsonl' for n in ['sadi','svar'] for s in [0,1]]
    paths += [PREVIOUS/f'coco10/p_fixed.shard{s}.jsonl' for s in [0,1]]
    hashes={**source['source_sha256'],**old['source_sha256'],**{str(p):sha(p) for p in paths}}
    groups={};cached=[]
    previous={s:source_groups(PREVIOUS/f'coco10/p_fixed.shard{s}.jsonl')[0] for s in [0,1]}
    for shard in [0,1]:
        groups[str(shard)]=[]
        native=source_groups(SOURCE/f'coco100/sadi.shard{shard}.jsonl')
        svar=source_groups(SOURCE/f'coco100/svar.shard{shard}.jsonl')
        assert [[r['image_id'] for r in g] for g in native]==[[r['image_id'] for r in g] for g in svar]
        for index,rows in enumerate(native):
            ids=[r['image_id'] for r in rows]
            groups[str(shard)].append(dict(ids=ids,visual_tokens=rows[0]['visual_token_count']))
            for previous_shard,records in previous.items():
                if ids==[r['image_id'] for r in records]:
                    assert all(r['visual_token_count']==rows[0]['visual_token_count'] and r['variant']=='p_fixed' for r in records)
                    origin=PREVIOUS/f'coco10/p_fixed.shard{previous_shard}.jsonl'
                    cached.append(dict(shard=shard,group=index,ids=ids,source=str(origin),source_sha256=sha(origin),
                        previous_shard=previous_shard,records_preserved_exactly=True,previous_algorithm='Frozen Qwen3LivePreferenceContrast p_fixed, eta.5/alpha1/beta.1'))
    assert sum(len(e['ids']) for e in cached)==10 and len(cached)==2
    plan=dict(fixed_at=time.time(),source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),
        source_sha256=hashes,previous=str(PREVIOUS),cohort='coco100',images=100,shards=2,max_batch=16,max_new_tokens=512,
        variants=['sadi','svar',*MODES],candidates=MODES,copied_controls=['sadi','svar'],groups=groups,cached_candidate_groups=cached,
        base_layers=list(range(5,19)),source_layer=19,action_layers=list(range(20,36)),eta=.5,alpha=1.,beta=.1,
        grouping='Original ordered22 groups, same B/P/images/prompt. Two old p_fixed groups retain identical original records, after actual128-token compatibility checks.',
        action='All three use native SADI positive, one read-only counterfactual tail20--35 and identical eta.5; negative priors are actual source19 FFN preference, uniform, or half-span rolled preference.',
        eligibility='Same frozen309 COCO object-start tokens, repeats allowed, no MLP/history filter; skip prefill.',
        decoding='2native-negative; candidate native>=max(native)+log(.1). Counterfactual KV always restored, native history owns decoding.',
        signal_scope='Partial EMS spatial preference only. Source Q/P drift is diagnostic; no SAR/risk-adaptive strength in these constant arms.',
        preflight='Both original B16 first groups: native128 matches historical SADI; disabled128, each actual nonzero branch with alpha0, and p_fixed eta0 all exact tokens/logits/all36current/fullKV. Cached p_fixed groups separately regenerate128 to verify tokens and original early change events.',
        runtime='290 new image-arm descriptions,10 old p_fixed,200 old baseline; raw generation_reused flags describe original records. Runtime receipt separately counts actual reuse.',
        scope='Reused development100, not independent validation. Full cohort sentence primary, legacy secondary, paired5000 seed43; image/repetition review before any500.',
        advance='Compare p_fixed with both matched-strength controls and originalSADI/SVAR. Only useful physical gains without single-loop/alias/denominator explanation may enter reserved independent500. Freeze choices before500; no validation tuning.')
    # Required by the frozen independent auditor. Check schema before any GPU.
    assert set(plan['copied_controls'])=={'sadi','svar'} and all(Path(n).is_absolute() or (REPO/n).exists() for n in hashes)
    check(plan);root.mkdir();(root/'coco100').mkdir();(root/'source').mkdir()
    for p in paths[:3]:(root/'source'/p.name).write_bytes(p.read_bytes())
    (root/'cohort.json').write_bytes((SOURCE/'cohort.json').read_bytes())
    (root/'object_lexicon.json').write_bytes((PREVIOUS/'object_lexicon.json').read_bytes())
    assert len(read(root/'object_lexicon.json')['token_ids'])==309
    for n in ['sadi','svar']:
        for s in [0,1]:(root/f'coco100/{n}.shard{s}.jsonl').write_bytes((SOURCE/f'coco100/{n}.shard{s}.jsonl').read_bytes())
    (root/'math_proof.json').write_bytes(proof.read_bytes());write(root/'protocol.json',plan)
    write(root/'analysis_request.json',dict(fixed_at=time.time(),protocol_sha256=sha(root/'protocol.json'),
        primary='Complete100 sentence Cs/Ci/false counts/image recall and F1. Legacy secondary, all-pair5000 image bootstrap seed43.',
        before_new_metrics=True,review='All changed false classes, true-class losses, fixed/introduced images, repeated mention concentrations and original photographs.',
        comparison='p_fixed versus same eta.5 uniform_fixed and rolled_fixed; no adaptive coefficient attribution.'))
    from scripts.analyze_qwen3_clva_quality import prepare as quality_prepare
    quality_prepare(root)
    print(dict(prepared=True,groups={k:len(v) for k,v in groups.items()},cached_groups=cached),flush=True)


def worker(root,shard):
    import torch
    from features.qwen3_live_preference_contrast import Qwen3LivePreferenceContrast
    from features.qwen3_live_preference_fixed_controls import Qwen3FixedPreferenceControl
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,generate
    from scripts.run_qwen3_full_sadi_next6 import VisionCache
    plan=read(root/'protocol.json');check(plan);wrapper=build_wrapper()
    nouns=frozenset(read(root/'object_lexicon.json')['token_ids']);progress=[];started=time.time();proof_done=False
    native_groups=source_groups(root/f'coco100/sadi.shard{shard}.jsonl');actual_new=actual_cached=0;encoders=vision_hits=0
    def make(mode,span,shared,**kwargs):
        if mode=='p_fixed':return Qwen3LivePreferenceContrast(wrapper,span,nouns,mode=mode,shared=shared,**kwargs)
        return Qwen3FixedPreferenceControl(wrapper,span,nouns,policy=mode,shared=shared,**kwargs)
    streams={n:(root/f'coco100/{n}.shard{shard}.jsonl').open('x',encoding='utf-8') for n in MODES}
    try:
        for index,group in enumerate(plan['groups'][str(shard)]):
            ids=group['ids'];span,inputs,_=batch(wrapper,ids);shared={};vision=VisionCache(wrapper,inputs)
            assert span[1]-span[0]==group['visual_tokens'] and [r['image_id'] for r in native_groups[index]]==ids
            with torch.inference_mode(),vision.capture():
                if not proof_done:
                    assert len(ids)==16
                    native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,nouns,'sadi'),128,True)
                    assert native[0]==[r['response_token_ids'][:128] for r in native_groups[index]]
                    reports=[]
                    diagnostics=[('disabled','p_fixed',dict(disabled=True))]
                    diagnostics += [('alpha0_'+n,n,dict(zero_alpha=True)) for n in MODES]
                    diagnostics += [('eta0','p_fixed',dict(zero_amount=True))]
                    for label,mode,kwargs in diagnostics:
                        controller=make(mode,span,shared,**kwargs);result=generate(wrapper,inputs,controller,128,True)
                        assert result[:3]==native[:3],label
                        events=[e for s in controller.statistics(result[0]) for e in s['full_events']]
                        assert not any(e['applied'] for e in events)
                        if label!='disabled':assert events,label
                        reports.append(dict(mode=label,passed=True,events=len(events),exact_sadi=True,seconds=result[3]))
                        write(root/f'preflight_progress{shard}.json',reports);controller.close();del controller,result
                    write(root/f'preflight{shard}.json',dict(passed=True,checks=reports,native_tokens=native[0],
                        native_per_step=native[1],full_cache_sha256=native[2],all36_current_and_full_kv_exact=True))
                    while not all((root/f'preflight{s}.json').exists() for s in [0,1]):
                        assert not any((root/f'failure{s}.json').exists() for s in [0,1]);time.sleep(1)
                    assert all(read(root/f'preflight{s}.json')['passed'] for s in [0,1]);proof_done=True;del native
                reuse=[e for e in plan['cached_candidate_groups'] if e['shard']==shard and e['group']==index]
                for mode in MODES:
                    if mode=='p_fixed' and reuse:
                        entry,=reuse;assert sha(Path(entry['source']))==entry['source_sha256']
                        records=source_groups(Path(entry['source']))[0]
                        controller=make(mode,span,shared);result=generate(wrapper,inputs,controller,128)
                        assert result[0]==[r['response_token_ids'][:128] for r in records]
                        fresh=controller.statistics(result[0])
                        for r,s in zip(records,fresh,strict=True):
                            old_events=[e for e in r['sar_ems_statistics']['full_events'] if e['step']<len(r['response_token_ids'][:128])]
                            assert s['full_events']==old_events
                        progress.append(dict(group=index,mode=mode,images=len(ids),reused=True,compatibility128=True,
                            compatibility_seconds=result[3],source=entry['source'],source_sha256=entry['source_sha256']))
                        actual_cached+=len(ids);controller.close();del controller,result,fresh
                    else:
                        controller=make(mode,span,shared);trace=[0];prompt=inputs['input_ids'].shape[1]
                        def lengths(_m,_a,out):
                            assert all(out.past_key_values.get_seq_length(i)==prompt+trace[0] for i in range(36));trace[0]+=1
                        with ExitStack() as stack:
                            stack.callback(wrapper.model.register_forward_hook(lengths).remove)
                            result=generate(wrapper,inputs,controller,512)
                        stats=controller.statistics(result[0]);assert trace[0]==controller.step+1
                        texts=wrapper.processor.batch_decode(result[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                        records=[dict(image_id=image,model='qwen3_vl_8b',variant=mode,response_token_ids=tokens,generated_text=text.strip(),
                            seconds=result[3]/len(ids),batch_wall_seconds=result[3],batch_size=len(ids),visual_token_count=span[1]-span[0],
                            sar_ems_statistics=s,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                            actual_model_calls_in_batch=controller.step+1) for image,tokens,text,s in zip(ids,result[0],texts,stats,strict=True)]
                        events=[e for s in stats for e in s['full_events']]
                        progress.append(dict(group=index,mode=mode,images=len(ids),reused=False,seconds=result[3],
                            row_events=len(events),immediate_changes=sum(e['immediate_change'] for e in events),
                            replay_model_steps=controller.replay_steps,semantic_head_calls=controller.reader.head_calls))
                        actual_new+=len(ids);controller.close();del controller,result,stats
                    for row in records:streams[mode].write(json.dumps(row,ensure_ascii=False)+'\n')
                    streams[mode].flush();write(root/f'progress{shard}.json',progress);print(progress[-1],flush=True)
                    del records;gc.collect();torch.cuda.empty_cache()
            assert vision.calls==1;encoders+=vision.calls;vision_hits+=vision.hits;del shared,vision,inputs
        check(plan)
        write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,groups=len(native_groups),
            actual_new_image_arms=actual_new,actual_cached_image_arms=actual_cached,encoder_calls=encoders,vision_hits=vision_hits))
    finally:
        for stream in streams.values():stream.close()


def analyze(root):
    plan=read(root/'protocol.json');check(plan);request=read(root/'analysis_request.json')
    assert request['protocol_sha256']==sha(root/'protocol.json') and read(root/'pipeline_complete.json')['passed']
    workers=[read(root/f'worker{s}.json') for s in [0,1]]
    assert all(w['passed'] and read(root/f'preflight{s}.json')['passed'] for s,w in enumerate(workers))
    assert sum(w['actual_new_image_arms'] for w in workers)==290 and sum(w['actual_cached_image_arms'] for w in workers)==10
    ids=read(root/'cohort.json')['coco100'];counts={};cached_verified=0
    for mode in MODES:
        raw=[];counts[mode]=dict(events=0,immediate_changes=0)
        for shard in [0,1]:
            groups=source_groups(root/f'coco100/{mode}.shard{shard}.jsonl')
            assert [[r['image_id'] for r in g] for g in groups]==[g['ids'] for g in plan['groups'][str(shard)]]
            for index,group in enumerate(groups):
                reuse=[e for e in plan['cached_candidate_groups'] if e['shard']==shard and e['group']==index]
                if mode=='p_fixed' and reuse:
                    entry,=reuse;assert group==source_groups(Path(entry['source']))[0];cached_verified+=len(group)
                for row in group:
                    raw.append(row['image_id']);s=row['sar_ems_statistics'];assert s['mode']==mode
                    assert s['eligibility_tokens']==309 and s['read_only_counterfactual']
                    assert not any(s[k] for k in ['disabled','zero_alpha','zero_amount'])
                    for e in s['full_events']:
                        assert e['amount']==.5 and not e['counterfactual_kv_committed']
                        assert row['response_token_ids'][e['step']]==e['new_token']
                    counts[mode]['events']+=len(s['full_events']);counts[mode]['immediate_changes']+=sum(e['immediate_change'] for e in s['full_events'])
        assert len(raw)==len(set(raw))==100 and set(raw)==set(ids)
    assert cached_verified==10
    write(root/'runtime_receipt.json',dict(passed=True,new_image_arms=290,cached_candidate_image_arms=10,cached_baseline_image_arms=200,
        native_encoder_calls=sum(w['encoder_calls'] for w in workers),activity=counts,
        raw_reuse_flags_belong_to_original_records=True,cached_records_verified_exactly=True))
    for version in ['sentence','legacy']:
        with (root/f'evaluate_{version}.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_caption_chair.py','--generation-root',str(root),
                '--cohort','coco100','--limit','100','--variants',*plan['variants'],'--evaluator',version],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'run_status.json',dict(status='complete',finished=time.time()))
    with (root/'independent_audit.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/audit_qwen3_caption_with_native_cache_v2.py',str(root)],
            cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(root/'independent_summary.json')['passed'];check(plan)
    write(root/'analysis_provenance.json',dict(passed=True,request_sha256=sha(root/'analysis_request.json'),
        protocol_sha256=sha(root/'protocol.json'),finished=time.time(),physical_review='pending',direct_advance_to500=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true')
    p.add_argument('--verify',action='store_true');p.add_argument('--proof',type=Path);p.add_argument('--analyze',action='store_true')
    p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:
        if a.verify:verify(a.root)
        elif a.prepare:assert a.proof is not None;prepare(a.root,a.proof)
        elif a.analyze:analyze(a.root)
        elif a.worker is not None:worker(a.root,a.worker)
        else:p.error('Select a stage.')
    except BaseException:
        if a.root.is_dir():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
