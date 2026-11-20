"""Preflight and caption pilot for full SADI followed by region guidance."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE = REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
MECHANISM = REPO/'outputs/qwen3_region_counterevidence_probe_20261120'
MODES = ['verified', 'unverified', 'shifted']


def check(plan):
    for name, expected in plan['source_sha256'].items():
        assert sha(REPO/name) == expected, name


def prepare(root):
    assert not root.exists(), 'Inspect existing work; never overwrite or relaunch it.'
    old = read(SOURCE/'protocol.json')
    recipe = REPO/'outputs/qwen3_region_probe_analysis_20261120/caption_pilot_recipe.json'
    assert sha(recipe) == 'e842e8f4dbe5a509370fa72aed41a89fee86f2d21f100c5cfdc696ee451bf4b2'
    hashes = dict(read(MECHANISM/'protocol.json')['source_sha256'])
    paths = [Path(__file__), REPO/'features/qwen3_region_guidance.py',
             REPO/'scripts/qwen3_caption_groups_with_cache.py',
             REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',
             SOURCE/'protocol.json', SOURCE/'cohort.json', SOURCE/'object_lexicon.json',
             recipe, MECHANISM/'protocol.json', MECHANISM/'pipeline_complete.json']
    paths += [SOURCE/f'coco100/{name}.shard{s}.jsonl' for name in ['sadi', 'svar'] for s in [0, 1]]
    paths += [Path(old['probe'])/n for n in ['model.pt', 'input_scaler.npz']]
    hashes.update({str(p): sha(p) for p in paths})
    plan = dict(source=str(SOURCE), source_protocol_sha256=sha(SOURCE/'protocol.json'),
        source_sha256=hashes, fixed_at=time.time(), cohort='coco100', images=100, shards=2,
        max_new_tokens=512, max_batch=16, variants=['sadi', 'svar', *MODES],
        candidates=MODES, copied_controls=['sadi', 'svar'], probe=old['probe'],
        real_threshold=old['real_threshold'], eos_token_ids=old['eos_token_ids'],
        recipe_sha256=sha(recipe), base_layers=list(range(5, 19)), source_layer=19,
        intervention_layers=list(range(20, 36)),
        grouping='Exact original ordered22 groups; no change to batch, image order or prompt.',
        action='Risk-gated semantic3x3 region boost with visual mass preserved; verified5Dr, unverified5D, shifted same5Dr at half-grid offset.',
        acceptance='Changed actual final argmax and lower original-target logp; defer all commits until every row has been evaluated against pristine native batch state.',
        sharing='Original vision, native image108 static semantics, crop encoding, fine whole-M semantics shared only within identical original B/P group.',
        scope='Reused development100, exploratory. Full captions required; token-piece changes alone are not mitigation evidence.',
        followup='Independent reserved500 only after useful full-caption effects and original-image review; no validation tuning.')
    check(plan)
    root.mkdir(); (root/'coco100').mkdir(); (root/'crops').mkdir(); (root/'source').mkdir()
    for path in paths:
        if path.suffix == '.py' and path.is_relative_to(REPO):
            dest=root/'source'/path.relative_to(REPO); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path,dest)
    for name in ['cohort.json', 'object_lexicon.json']: shutil.copyfile(SOURCE/name, root/name)
    for name in ['sadi','svar']:
        for s in [0,1]: shutil.copyfile(SOURCE/f'coco100/{name}.shard{s}.jsonl', root/f'coco100/{name}.shard{s}.jsonl')
    write(root/'protocol.json', plan)
    print(dict(prepared=True, variants=plan['variants']), flush=True)


def setup(root):
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    plan=read(root/'protocol.json'); check(plan)
    wrapper=build_wrapper()
    probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(read(root/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    return plan,wrapper,probe,allowed,parser


def batch(wrapper, ids):
    import torch
    from PIL import Image
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    (span, group),=same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
    assert [i for i,_ in group] == ids
    inputs={key:torch.cat([v[key] for _,v in group]).to(wrapper.device) for key in group[0][1]}
    photos={}
    for image in ids:
        with Image.open(COCO_ROOT/'val2014'/f'COCO_val2014_{image:012d}.jpg') as f: photos[image]=f.convert('RGB')
    return span,inputs,photos


def digest(tensors):
    import torch
    h=hashlib.sha256()
    for tensor in tensors:
        value=tensor.detach().contiguous()
        h.update(str((tuple(value.shape),str(value.dtype))).encode())
        h.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def generate(wrapper, inputs, controller, cap, observe=False):
    import torch
    from scripts.generation_time_evidence_drift_qwen3 import MODEL
    from scripts.generation_time_evidence_drift_qwen3_batched import trim_eos
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    prompt=inputs['input_ids'].shape[1]; trace=[]; final={}
    def hook(_m,_a,out):
        cache=out.past_key_values
        assert all(cache.get_seq_length(i)==prompt+len(trace) for i in range(36))
        trace.append(dict(logits=digest([out.logits[:,-1]]),
                          current_kv=digest([t[:,:,-1:] for layer in cache.layers for t in [layer.keys,layer.values]])))
        final['cache']=cache
    torch.cuda.synchronize(); started=time.time()
    with ExitStack() as stack:
        stack.enter_context(controller.capture())
        if observe: stack.callback(wrapper.model.register_forward_hook(hook).remove)
        output=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=cap))
    torch.cuda.synchronize(); seconds=time.time()-started
    eos=wrapper.model.generation_config.eos_token_id; eos=set(eos if isinstance(eos,list) else [eos])
    tokens=[trim_eos(t,eos) for t in output[:,prompt:].tolist()]
    full=digest([t for layer in final['cache'].layers for t in [layer.keys,layer.values]]) if observe else None
    return tokens,trace,full,seconds


def preflight(root, shard):
    import torch
    from features.qwen3_region_guidance import Qwen3RegionGuidance
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    plan,wrapper,probe,allowed,parser=setup(root)
    group=read(MECHANISM/'protocol.json')['groups'][shard]; ids=group['ids']
    span,inputs,photos=batch(wrapper,ids); shared={}; report=[]
    cap=128
    references={r['image_id']:r for s in [0,1] for g in source_groups(SOURCE/f'coco100/sadi.shard{s}.jsonl') for r in g}
    with torch.inference_mode():
        native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'),cap,True)
        assert native[0]==[references[i]['response_token_ids'][:cap] for i in ids]
        for mode in ['disabled','zero','reject','live']:
            controller=Qwen3RegionGuidance(wrapper,span,allowed,probe=probe,history_parser=parser,
                image_ids=ids,images=photos,output_dir=root/'crops',shared_cache=shared,
                disabled=mode=='disabled',zero_strength=mode=='zero',reject_all=mode=='reject')
            current=generate(wrapper,inputs,controller,cap,True)
            stats=controller.statistics(current[0]); exact=mode!='live'
            if exact:
                assert current[:3]==native[:3], mode
                assert not any(s['accepted'] for s in stats)
            events=[e for s in stats for e in s['full_events']]
            if mode in ['zero','reject']: assert events,mode
            if mode=='zero': assert all(e['proposed_current_kv_max_difference']==0 for e in events)
            entry=dict(mode=mode,all36_current_and_full_KV_logits_tokens_native_exact=exact,
                proposals=len(events),accepted=sum(e['accepted'] for e in events),seconds=current[3],
                max_current_kv_difference=max([e['proposed_current_kv_max_difference'] for e in events],default=0),
                stats=stats,tokens=current[0])
            # Cached gate values are valid only on the unchanged original prefix.
            if exact:
                for case in group['cases']:
                    matches=[g for g in stats[case['row']]['gates'] if g['step']==case['step'] and g['token']==case['token_id']]
                    if mode!='disabled':
                        assert len(matches)==1,case
                        assert abs(matches[0]['p_real']-case['cached_p_real'])<1e-5
            report.append(entry);write(root/f'preflight_progress{shard}.json',report)
            print(dict(shard=shard,mode=mode,proposals=len(events),accepted=entry['accepted'],exact=exact),flush=True)
            controller.close();del controller,current,stats;gc.collect();torch.cuda.empty_cache()
    if shard==1:
        live=report[-1]; event,=[e for e in live['stats'][0]['full_events'] if e['step']==97]
        assert event['token']==22435 and event['accepted'] and event['proposed_token']!=22435
        assert event['proposed_current_kv_max_difference']>0
    check(plan)
    write(root/f'preflight{shard}.json',dict(passed=True,ids=ids,cases=report,cap=cap,
        source_sha256=plan['source_sha256'],frozen_protocol_sha256=sha(root/'protocol.json')))


def worker(root,shard):
    import torch
    from features.qwen3_region_guidance import Qwen3RegionGuidance
    plan,wrapper,probe,allowed,parser=setup(root)
    assert all(read(root/f'preflight{s}.json')['passed'] for s in [0,1])
    groups=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
    paths={mode:root/f'coco100/{mode}.shard{shard}.jsonl' for mode in MODES}
    assert not any(p.exists() for p in paths.values())
    progress=[];started=time.time()
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group]; span,inputs,photos=batch(wrapper,ids)
            assert span[1]-span[0]==group[0]['visual_token_count'];shared={}
            for mode in MODES:
                torch.cuda.reset_peak_memory_stats()
                controller=Qwen3RegionGuidance(wrapper,span,allowed,probe=probe,history_parser=parser,
                    image_ids=ids,images=photos,output_dir=root/'crops',shared_cache=shared,policy=mode)
                tokens,_,_,seconds=generate(wrapper,inputs,controller,512)
                stats=controller.statistics(tokens)
                texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                rows=[dict(image_id=i,model='qwen3_vl_8b',variant=mode,response_token_ids=t,
                    generated_text=text.strip(),seconds=seconds/len(ids),batch_wall_seconds=seconds,
                    batch_size=len(ids),visual_token_count=span[1]-span[0],sar_ems_statistics=s,
                    decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                    actual_model_calls_in_batch=controller.step+1)
                    for i,t,text,s in zip(ids,tokens,texts,stats,strict=True)]
                with paths[mode].open('a',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                entry=dict(group=index,mode=mode,images=len(ids),seconds=seconds,
                    accepted=sum(s['accepted'] for s in stats),
                    proposals=sum(len(s['full_events']) for s in stats),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
                progress.append(entry);write(root/f'progress{shard}.json',progress);print(entry,flush=True)
                controller.close();del controller,stats,rows;gc.collect();torch.cuda.empty_cache()
            shared.clear();del inputs,photos,shared;gc.collect();torch.cuda.empty_cache()
    check(plan)
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,groups=len(groups),progress=progress))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',type=int,choices=[0,1]);p.add_argument('--worker',type=int,choices=[0,1])
    a=p.parse_args()
    try:
        if a.prepare:prepare(a.root)
        elif a.preflight is not None:preflight(a.root,a.preflight)
        elif a.worker is not None:worker(a.root,a.worker)
        else:p.error('Choose prepare, preflight or worker.')
    except BaseException:
        if a.root.exists():
            path=a.root/f'failure.{"preflight" if a.preflight is not None else "worker"}{a.preflight if a.preflight is not None else a.worker}.json'
            if not path.exists():write(path,dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
