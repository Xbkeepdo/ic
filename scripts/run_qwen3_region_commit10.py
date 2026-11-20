"""Full-caption acceptance ablation on the two existing complete groups."""

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
from scripts.run_qwen3_region_guidance import setup,batch,generate
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE=REPO/'outputs/qwen3_clva_signal_preflight10_v2_20261120'
OLD=REPO/'outputs/qwen3_region_guidance100_20261120'
PROOF=REPO/'outputs/qwen3_region_commit_predicates_20261120.json'
MODES=['probability','direct','shifted']


def prepare(root):
    assert not root.exists(),'Preserve existing runs; inspect instead of relaunching.'
    base=read(SOURCE/'protocol.json');previous=read(OLD/'protocol.json');proof=read(PROOF)
    assert read(SOURCE/'pipeline_complete.json')['passed'] and proof['passed'] and not proof['cuda_initialized']
    hashes=dict(previous['source_sha256'])
    paths=[Path(__file__),REPO/'features/qwen3_region_commit.py',REPO/'scripts/verify_qwen3_region_commit.py',
        REPO/'scripts/analyze_qwen3_region_commit10.py',REPO/'scripts/analyze_qwen3_clva_quality.py',
        REPO/'scripts/evaluate_qwen3_region_caption_chair.py',REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',
        SOURCE/'protocol.json',SOURCE/'pipeline_complete.json',SOURCE/'cohort.json',SOURCE/'object_lexicon.json',
        OLD/'protocol.json',OLD/'analysis_complete.json',PROOF]
    # The old root uses a recovery analysis stage, whose provenance is frozen
    # explicitly below. Never infer completion from directory names.
    if not paths[-2].exists():
        paths.remove(OLD/'analysis_complete.json')
        paths.extend([OLD/'independent_summary.json',OLD/'run_status.json'])
    for name,value in proof['source_sha256'].items():assert sha(Path(name))==value,name
    paths += [SOURCE/f'coco10/{name}.shard{s}.jsonl' for name in ['sadi','svar'] for s in [0,1]]
    paths += [OLD/f'coco100/unverified.shard{s}.jsonl' for s in [0,1]]
    hashes.update({str(p):sha(p) for p in paths})
    for name,wanted in hashes.items():assert sha(REPO/name)==wanted,name
    root.mkdir();(root/'coco10').mkdir();(root/'source').mkdir()
    for name in ['cohort.json','object_lexicon.json']:(root/name).write_bytes((SOURCE/name).read_bytes())
    for name in ['sadi','svar']:
        for shard in [0,1]:
            p=SOURCE/f'coco10/{name}.shard{shard}.jsonl';(root/'coco10'/p.name).write_bytes(p.read_bytes())
    oldgroups=[(s,g) for s in [0,1] for g in source_groups(OLD/f'coco100/unverified.shard{s}.jsonl')]
    controls=[]
    for shard,group in enumerate(base['groups']):
        origin,rows=next((s,g) for s,g in oldgroups if [r['image_id'] for r in g]==group['ids'])
        assert all(r['variant']=='unverified' for r in rows)
        path=root/f'coco10/unverified.shard{shard}.jsonl'
        with path.open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        source=OLD/f'coco100/unverified.shard{origin}.jsonl'
        controls.append(dict(shard=shard,ids=group['ids'],source=str(source),source_sha256=sha(source),
            materialized_sha256=sha(path),records_preserved_exactly=True))
    hashes.update({str(p):sha(p) for p in [root/'cohort.json',root/'object_lexicon.json',*sorted((root/'coco10').glob('*.jsonl'))]})
    for p in paths:
        if p.suffix=='.py':(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),
        source_sha256=hashes,groups=base['groups'],cohort='coco10',images=10,shards=2,max_batch=16,max_new_tokens=512,
        variants=['sadi','svar','unverified',*MODES],candidates=MODES,copied_controls=['sadi','svar'],
        historical_controls=controls,probe=previous['probe'],real_threshold=.418672114610672,
        base_layers=list(range(5,19)),source_layer=19,action_layers=list(range(20,36)),
        budget='5*HellingerSquared(Q_SAR19,P_FFN19), tau_ffn=.03; same semantic-class ROI3x3 and visual-mass-preserving enhancement. No crop encoding or fine verification.',
        eligibility='Frozen108 risk MLP threshold, fixed309 tokens, first token/class as in the parent; no threshold or classifier change.',
        modes=dict(unverified='Historical immediate-token-change plus probability-decrease acceptance.',
            probability='Commit any lower-original-target-logp proposal, including unchanged argmax.',
            direct='Commit every nonzero proposal, including unchanged argmax and increased original-target logp.',
            shifted='Same direct acceptance and real5D budget, ROI shifted half of original image grid.'),
        preflight='NativeSADI128; frozen parent unverified128; own wordchange128 exact parent; disabled/zero/reject128 exact SADI, logits/all36 current/fullKV; then three full512 arms.',
        reasoning='Old ROI operator rejected earlier same-token changes that could affect later generation. Isolate this acceptance assumption without strengthening the operator, replacing the backbone or requiring immediate word correction.',
        scope='Old10 development implementation and complete-caption screen; no500 decision on10. Full captions, physical review, repetitions, recall and false counts required.'))
    print(dict(prepared=True,variants=['sadi','svar','unverified',*MODES]),flush=True)


def worker(root,shard):
    import torch
    from features.qwen3_region_commit import Qwen3RegionCommit
    from features.qwen3_region_guidance import Qwen3RegionGuidance
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    plan,wrapper,probe,allowed,parser=setup(root);ids=plan['groups'][shard]['ids']
    span,inputs,photos=batch(wrapper,ids);shared={};reports=[];started=time.time()
    baseline=[r['response_token_ids'] for g in source_groups(root/f'coco10/sadi.shard{shard}.jsonl') for r in g]
    historical=[r['response_token_ids'] for g in source_groups(root/f'coco10/unverified.shard{shard}.jsonl') for r in g]
    def make(kind,**kwargs):
        common=dict(probe=probe,history_parser=parser,image_ids=ids,images=photos,
            output_dir=root/'unused_crops',shared_cache=shared)
        if kind=='parent':return Qwen3RegionGuidance(wrapper,span,allowed,policy='unverified',**common)
        return Qwen3RegionCommit(wrapper,span,allowed,commit_mode=kind,**common,**kwargs)
    def harvest(_m,args,out):
        assert 'vision' not in shared
        shared['vision']=dict(pixels=args[0].detach().clone(),grid=inputs['image_grid_thw'].detach().clone(),
            output=(out[0].detach().clone(),[x.detach().clone() for x in out[1]]))
    with torch.inference_mode():
        handle=wrapper.model.model.visual.register_forward_hook(harvest)
        try:native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi'),128,True)
        finally:handle.remove()
        assert native[0]==[t[:128] for t in baseline]
        write(root/f'native128_proof{shard}.json',dict(tokens=native[0],per_step=native[1],full_cache_sha256=native[2]))
        controller=make('parent');parent=generate(wrapper,inputs,controller,128,True)
        assert parent[0]==[t[:128] for t in historical]
        parent_stats=controller.statistics(parent[0]);controller.close();del controller
        controller=make('wordchange');current=generate(wrapper,inputs,controller,128,True)
        assert current[:3]==parent[:3]
        events=[e for s in controller.statistics(current[0]) for e in s['full_events']]
        reports.append(dict(mode='wordchange_compatibility',passed=True,exact_parent=True,proposals=len(events),seconds=current[3]))
        write(root/f'wordchange128_proof{shard}.json',dict(passed=True,tokens=current[0],per_step=current[1],full_cache_sha256=current[2]))
        controller.close();del controller,current,parent,parent_stats;gc.collect();torch.cuda.empty_cache()
        for mode in ['disabled','zero','reject',*MODES]:
            diagnostic=mode in ['disabled','zero','reject'];cap=128 if diagnostic else 512
            controller=make('direct' if diagnostic else mode,disabled=mode=='disabled',zero_strength=mode=='zero',reject_all=mode=='reject')
            result=generate(wrapper,inputs,controller,cap,diagnostic);stats=controller.statistics(result[0])
            events=[e for s in stats for e in s['full_events']]
            if diagnostic:
                assert result[:3]==native[:3],mode
                assert not any(e['accepted'] for e in events)
                if shard==1 and mode in ['zero','reject']:assert events,mode
                if mode=='zero':assert all(e['proposed_current_kv_max_difference']==0 for e in events)
            else:
                texts=wrapper.processor.batch_decode(result[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                rows=[dict(image_id=image,model='qwen3_vl_8b',variant=mode,response_token_ids=tokens,generated_text=text.strip(),
                    seconds=result[3]/len(ids),batch_wall_seconds=result[3],batch_size=len(ids),visual_token_count=span[1]-span[0],
                    sar_ems_statistics=s,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                    actual_model_calls_in_batch=controller.step+1) for image,tokens,text,s in zip(ids,result[0],texts,stats,strict=True)]
                with (root/f'coco10/{mode}.shard{shard}.jsonl').open('x',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            report=dict(mode=mode,passed=True,exact_sadi=diagnostic,seconds=result[3],proposals=len(events),
                accepted=sum(e['accepted'] for e in events),same_token_commits=sum(e['same_token_commit'] for e in events),
                token_changes=sum(e['accepted'] and e['proposed_token']!=e['token'] for e in events))
            reports.append(report);write(root/f'progress{shard}.json',reports);print(report,flush=True)
            controller.close();del controller,result,stats;gc.collect();torch.cuda.empty_cache()
    for name,wanted in plan['source_sha256'].items():assert sha(REPO/name)==wanted,name
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,cases=reports,original_encoder_calls=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:prepare(args.root) if args.prepare else worker(args.root,args.worker)
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
