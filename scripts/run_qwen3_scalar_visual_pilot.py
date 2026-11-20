"""Compare all-head visual contrasts with scalar SAR/FFN drift on old100."""

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE=shared.OUT/'qwen3_block_acg_pilot_20261118'
FRESH=shared.OUT/'qwen3_block_acg_fresh500_20261118'
OUT=shared.OUT/'qwen3_scalar_visual_pilot_20261119'
CONTROLS=['sadi','svar','sadi_sam_acg_fixed_19_35','sadi_sam_acg_guided_19_35']
MODES={'sadi_scalar':('sadi',True),'svar_fixed':('svar',False),'svar_scalar':('svar',True)}
CPU=shared.OUT/'qwen3_scalar_visual_cpu/verification.json'
PREFLIGHT=shared.OUT/'qwen3_scalar_visual_preflight/verification.json'
CERTIFICATE=SOURCE/'no_gate_cache_validation.json'
FILES=['features/qwen3_scalar_visual_feedback.py','scripts/verify_qwen3_scalar_visual.py',
       'scripts/verify_qwen3_scalar_visual_gpu.py','scripts/run_qwen3_scalar_visual_pilot.py',
       'scripts/audit_qwen3_scalar_visual_pilot.py','scripts/qwen3_caption_groups_with_cache.py',
       'scripts/audit_qwen3_caption_with_native_cache_v2.py']


def prepare():
    assert not OUT.exists(),'Inspect an existing run before any restart'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert shared.read(FRESH/'post_audit_status.json')['status']=='passed'
    assert shared.read(FRESH/'screening.json')['advance_to_500']==[]
    assert shared.read(CPU)['passed'] and shared.read(PREFLIGHT)['passed']
    for n,v in shared.read(PREFLIGHT)['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    cert=shared.read(CERTIFICATE);assert cert['passed'] and cert['joint17_agreement_verified']
    for n,v in cert['source_sha256'].items():assert shared.sha(REPO/n)==v,n
    inputs=[CPU,PREFLIGHT,CERTIFICATE,SOURCE/'protocol.json',SOURCE/'final_provenance.json',
        FRESH/'protocol.json',FRESH/'independent_summary.json',FRESH/'screening.json',FRESH/'direction_signs.json',
        *(REPO/n for n in FILES)]
    for p in inputs:hashes[str(p)]=shared.sha(p)
    for n,v in hashes.items():assert shared.sha(Path(n))==v,n
    OUT.mkdir();(OUT/'logs').mkdir();(OUT/'coco100').mkdir()
    for n in ('cohort.json','object_lexicon.json'):shutil.copyfile(SOURCE/n,OUT/n)
    for shard in (0,1):
        for name in CONTROLS:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';hashes[str(p)]=shared.sha(p)
            shutil.copyfile(p,OUT/'coco100'/p.name)
    plan=dict(stage='scalar-visual-development',images=100,shards=2,cohort='coco100',max_batch=16,
        max_new_tokens=512,variants=[*CONTROLS,*MODES],candidates=list(MODES),copied_controls=CONTROLS,
        cached_controls=['sadi','svar'],base_by_candidate={n:b for n,(b,_) in MODES.items()},source=str(SOURCE),
        source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),source_sha256=hashes,fixed_at=time.time(),
        probe=old['probe'],real_threshold=old['real_threshold'],cache_certificate=str(CERTIFICATE),
        cache_certificate_sha256=shared.sha(CERTIFICATE),
        recipe=dict(base_layers=[5,18],joint_layers=list(range(19,36)),gamma=1.4,
            contrast='Same complete all-head orthogonal visual contrast as the earlier fixed adaptation.',
            scalar='Uniform across heads within a layer: gamma_lh=1.4*D_l; fixed SVAR control uses1.4.',
            drift='Q=Norm(write_norm*(1+log_s)); P=softmax(cos(u_FFN,hpre_visual)/.03); D=HellingerSquared(Q,P).',
            unchanged='Original108 gate/threshold, first token/class scope, native directions, joint19--35 replay, actual target-logp decrease acceptance and exact rollback.',
            no_backward=True,static_cache='Separate RAM caches by base; SVAR fixed/scalar share after exact prefix checks.',
            cache='Only sadi_scalar reuses the seven certified complete zero-gate SADI batches (nine images). SVAR generations are all fresh.',
            scope='Development after negative independent500; this old100 does not independently validate either variant.'),
        development_screen='Versus each own base: both Cs/Ci and false count decrease; F1 loss<=1.5pp; recall loss<=2pp; leave-largest-reduction Ci no worse. Scalar additionally assessed against same-base fixed; require physical-image confirmation before a new independent500.')
    assert len('chair_per_image.first100.'+'-'.join(sorted(plan['variants']))+'.jsonl')<=255
    shared.write(OUT/'protocol.json',plan);return plan


def cached_sadi_rows(group,plan,shard,index):
    path=SOURCE/f'coco100/sadi_sam_acg_guided_19_35.shard{shard}.jsonl'
    rows=[];started=time.perf_counter()
    for ref in group:
        stat=copy.deepcopy(ref['sar_ems_statistics'])
        assert not stat['recycle_events'] and not any(g['positive'] for g in stat['gate_events'])
        for k in ('native_risk_solver_calls','native_feature_seconds','static_seconds','static_head_calls',
                  'shared_static_hits','semantic_signature_seconds','gradient_probe_calls','probe_calls'):stat[k]=0
        stat.update(full_joint_events=[],full_gate_events=copy.deepcopy(stat['gate_events']),
            full_class_decisions=copy.deepcopy(stat['class_decisions']),total_proposals=0,total_accepted=0,
            terminal_eos_accepts=0,full_event_log_includes_terminal_eos=True,base_variant='sadi',
            signal_guided=False,uses_scalar_sar_ffn_drift=True,head_gradient_guidance=False,
            gradient_sources=[],one_backward_for_all_layers=False,cached_gate_decisions=len(stat['gate_events']),
            event_log_from_certified_cache=True)
        rows.append(dict(ref,variant='sadi_scalar',seconds=0.,batch_wall_seconds=0.,sar_ems_statistics=stat,
            decoder='cached_native_no_positive_gate',generation_reused=True,actual_model_calls_in_batch=0,
            cache_origin=dict(source=str(path),source_sha256=shared.sha(path),certificate=str(CERTIFICATE),
                certificate_sha256=plan['cache_certificate_sha256'],shard=shard,group=index)))
    elapsed=time.perf_counter()-started
    for row in rows:row['cache_io_seconds']=elapsed/len(rows)
    return rows


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_scalar_visual_feedback import Qwen3BatchScalarVisualFeedback
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():assert shared.sha(Path(n))==v,n
    groups=source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    base_groups={b:source_groups(OUT/f'coco100/{b}.shard{gpu}.jsonl') for b in ('sadi','svar')}
    cached=source_groups(SOURCE/f'coco100/sadi_sam_acg_guided_19_35.shard{gpu}.jsonl')
    eligible={e['group']:e for e in shared.read(CERTIFICATE)['eligible_groups'] if e['shard']==gpu}
    assert all(shared.group_keys(g)==shared.group_keys(groups) for g in [cached,*base_groups.values()])
    paths={n:OUT/f'coco100/{n}.shard{gpu}.jsonl' for n in MODES};assert not any(p.exists() for p in paths.values())
    wrapper=build_wrapper();probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py');memory=[]
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group]
            (span,inputs),=common.same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
            assert [i for i,_ in inputs]==ids and span[1]-span[0]==group[0]['visual_token_count']
            if index==0:
                checks={}
                for base in ('sadi','svar'):
                    tokens,_,_=common.generate_group(wrapper,span,inputs,allowed,base,32)
                    assert tokens==[r['response_token_ids'][:32] for r in base_groups[base][index]]
                    checks[base]=True;gc.collect();torch.cuda.empty_cache()
                shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,image_ids=ids,native32_exact=checks))
            caches={'sadi':{},'svar':{}}
            for name,(base,use_drift) in MODES.items():
                if name=='sadi_scalar' and index in eligible:
                    assert eligible[index]['image_ids']==ids
                    rows=cached_sadi_rows(cached[index],plan,gpu,index)
                else:
                    holder={}
                    def factory(model,current_span,nouns,mode,strength):
                        c=Qwen3BatchScalarVisualFeedback(model,current_span,allowed,probe=probe,
                            semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                            shared_static=caches[base],real_threshold=plan['real_threshold'],
                            base_variant=base,use_drift=use_drift,gamma=1.4)
                        holder['controller']=c;return c
                    torch.cuda.reset_peak_memory_stats(wrapper.device)
                    with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                        tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
                    calls=holder['controller'].steps;holder.clear()
                    texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    rows=[dict(image_id=i,model=MODEL,variant=name,response_token_ids=t,generated_text=text.strip(),
                        seconds=seconds/len(ids),batch_wall_seconds=seconds,batch_size=len(ids),
                        visual_token_count=span[1]-span[0],sar_ems_statistics=stat,
                        decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                        actual_model_calls_in_batch=calls) for i,t,text,stat in zip(ids,tokens,texts,stats,strict=True)]
                with paths[name].open('a',encoding='utf-8') as f:
                    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                gc.collect();torch.cuda.empty_cache()
                memory.append(dict(group=index,variant=name,allocated=torch.cuda.memory_allocated(wrapper.device),
                    peak=torch.cuda.max_memory_allocated(wrapper.device),reused=rows[0]['generation_reused']))
                shared.write(OUT/f'memory.gpu{gpu}.json',memory)
                print(f'[{gpu}] {name} group={index} batch={len(ids)} seconds={rows[0]["batch_wall_seconds"]:.3f}',flush=True)
            del caches;gc.collect();torch.cuda.empty_cache()
    assert all(shared.group_keys(source_groups(p))==shared.group_keys(groups) for p in paths.values())


def evaluate(output,plan):
    for n,v in plan['source_sha256'].items():assert shared.sha(Path(n))==v,n
    for mode in ('sentence','legacy'):
        subprocess.run([sys.executable,'scripts/evaluate_caption_chair.py','--generation-root',str(output),
            '--cohort','coco100','--limit','100','--variants',*plan['variants'],'--evaluator',mode],cwd=REPO,check=True)
    folder=output/'evaluations/sentence_v1/coco100';tag='-'.join(sorted(plan['variants']))
    metrics=shared.read(folder/f'metrics.first100.{tag}.json')['methods']
    rows=[json.loads(line)['methods'] for line in (folder/f'chair_per_image.first100.{tag}.jsonl').read_bytes().splitlines()]
    counts={n:sum(r[n]['hallucinated_mentions'] for r in rows) for n in plan['variants']}
    results={};advance=[]
    for name,base in plan['base_by_candidate'].items():
        a,b=metrics[base],metrics[name]
        drop=max(range(100),key=lambda i:rows[i][base]['hallucinated_mentions']-rows[i][name]['hallucinated_mentions'])
        trimmed={n:sum(r[n]['hallucinated_mentions'] for i,r in enumerate(rows) if i!=drop)/
            max(1,sum(r[n]['object_mentions'] for i,r in enumerate(rows) if i!=drop)) for n in (base,name)}
        checks=dict(CHAIRs_decreases=b['chair_s']<a['chair_s'],CHAIRi_decreases=b['chair_i']<a['chair_i'],
            false_mentions_decrease=counts[name]<counts[base],F1_loss_at_most_1_5pp=b['object_f1']>=a['object_f1']-.015,
            recall_loss_at_most_2pp=b['object_recall']>=a['object_recall']-.02,
            CHAIRi_no_worse_without_largest_reduction=trimmed[name]<=trimmed[base])
        results[name]=dict(base=base,checks=checks,leave_one_out_chair_i=trimmed,dropped_image_index=drop)
        if all(checks.values()):advance.append(name)
    shared.write(output/'screening.json',dict(advance_to_500=advance,candidates=results,
        hallucinated_mentions=counts,requires_image_review=True,not_automatic_confirmation=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    if args.worker is not None:worker(args.worker)
    else:
        with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),\
                patch.object(driver,'prepare',prepare),patch.object(shared,'evaluate',evaluate):driver.run()
