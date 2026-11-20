"""Test whether omitting I from the existing detector preserves real objects."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_causal_visual_tilt_pilot as causal
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE = shared.OUT/'qwen3_mass_matched_visual_pilot_20261119'
DIAGNOSTIC = shared.OUT/'qwen3_risk_semantic_gate_transfer_20261119'
RESERVED = shared.OUT/'qwen3_mass_matched_visual_reserved500_20261119'
OUT = shared.OUT/'qwen3_risk_semantic_gate_pilot_20261119'
NAME = 'risk_el_fixed'
CONTROLS = ['sadi', 'svar', 'causal_fixed']
PROBE = REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b/risk+E_l/seed43'
FILES = ['features/qwen3_risk_semantic_probe.py', 'scripts/run_qwen3_risk_semantic_gate_pilot.py',
    'scripts/audit_qwen3_risk_semantic_gate_pilot.py', 'scripts/analyze_qwen3_risk_semantic_gate_pilot.py']
AUDITS = [FILES[2], 'scripts/audit_qwen3_caption_with_native_cache_v2.py', FILES[3],
    'scripts/audit_qwen3_caption_denominator.py']


def prepare():
    assert not OUT.exists(), 'Inspect existing outputs and actual processes before any resume'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert shared.read(DIAGNOSTIC/'analysis.json')['passed']
    assert not (RESERVED/'consumed.json').exists()
    old = shared.read(SOURCE/'protocol.json'); diagnostic = shared.read(DIAGNOSTIC/'protocol.json')
    hashes = dict(old['source_sha256']); hashes.update(diagnostic['source_sha256'])
    paths = [SOURCE/n for n in ('protocol.json', 'analysis_protocol.json', 'final_provenance.json')]
    paths += [DIAGNOSTIC/n for n in ('protocol.json', 'analysis.json', 'human_review.json', 'generation_followup.json')]
    paths += [RESERVED/n for n in ('cohort.json', 'reservation.json', 'batch_plan.json')]
    paths += [REPO/n for n in [*FILES, *AUDITS, 'scripts/run_qwen3_causal_visual_tilt_pilot.py',
        'scripts/run_qwen3_history_route_pilot.py', 'scripts/analyze_qwen3_causal_visual_tilt_pilot.py',
        'scripts/analyze_qwen3_scalar_visual_pilot.py', 'scripts/qwen3_caption_groups_with_cache.py']]
    for p in paths:
        hashes[str(p)] = shared.sha(p)
    for n,v in hashes.items():
        assert shared.sha(REPO/n) == v,n
    OUT.mkdir(); (OUT/'logs').mkdir(); (OUT/'coco100').mkdir()
    for n in ('cohort.json', 'object_lexicon.json'):
        shutil.copyfile(SOURCE/n, OUT/n)
    for shard in (0,1):
        for name in CONTROLS:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl'
            hashes[str(p)]=shared.sha(p); shutil.copyfile(p,OUT/'coco100'/p.name)
    recipe=dict(base_layers=[5,18],unchanged_anchor=19,budget_source_layers=list(range(19,35)),
        intervention_layers=list(range(20,36)),gamma=1.,
        gate='Existing risk+E_l/seed43 checkpoint, original training threshold; I omitted.',
        shift='abs(actual current-query visual logit)*.5; gamma1; native source, no refresh.',
        coefficient='Uniform .5, exactly the existing causal_fixed operator.',
        acceptance=old['recipe']['acceptance'],
        cache='All100 new candidate captions. Three controls copied byte-for-byte; original batch boundaries retained.',
        component_scope='Native108 reader retained; frozen72-input probe sees risk[0:36] and E_l[72:108] only.')
    plan=dict(stage='risk-semantic-preservation-development',images=100,shards=2,cohort='coco100',
        max_batch=16,max_new_tokens=512,variants=[*CONTROLS,NAME],candidates=[NAME],
        copied_controls=CONTROLS,cached_controls=['sadi'],base_by_candidate={NAME:'sadi'},
        mode_by_candidate={NAME:'fixed'},refresh_by_candidate={NAME:False},
        source=str(SOURCE),source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),
        source_sha256=hashes,fixed_at=time.time(),probe=str(PROBE),
        real_threshold=diagnostic['threshold_p_real'],eos_token_ids=old['eos_token_ids'],recipe=recipe,
        hypothesis='Fewer actions on supported objects may preserve recall and improve the complete generation trajectory. This does not require the newly triggered token itself to be an unsupported object.',
        relationship_to_offline_screen='No new confirmed unsupported trigger was found, so the prior added-error screen did not pass. This separately frozen pilot tests preservation rather than added-error detection; no checkpoint or threshold is refit.',
        development_screen='Compare full sentence Cs/Ci and recall/F1 against SADI, original108 fixed and SVAR. Require genuine useful incremental behavior, not only denominator growth. Old100 is exploratory; all outcomes retained.',
        followup='Only a useful new candidate advances to the already reserved, unchanged500. No automatic validation or redraw.')
    shared.write(OUT/'protocol.json',plan)
    names=[*AUDITS,'scripts/analyze_qwen3_scalar_visual_pilot.py','scripts/analyze_qwen3_causal_visual_tilt_pilot.py',
        'scripts/audit_qwen3_visual_feedback_mentions.py','scripts/audit_qwen3_fresh_caption500.py']
    shared.write(OUT/'analysis_protocol.json',dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'),before_inspecting_pilot_metrics=True,
        source_sha256={str(REPO/n):shared.sha(REPO/n) for n in names},
        primary='Full100 sentence/legacy, counts/recall/F1 and paired5000 seed43; all4 arms.',
        cases='All fixes/introductions and top3 nonzero false/recall changes; original photos; distinguish aliases/parts/pictures from physical errors.',
        scope='Development only. Do not claim the initial added-error diagnostic screen passed.'))
    return plan


def decorate(stats,eos):
    causal.decorate_statistics(stats,False,eos)
    for s in stats:
        s.update(gate_feature_components=['risk','E_l'],gate_feature_indices=[*range(36),*range(72,108)],
            feature_reader_output_width=108,probe_input_width=72,
            acg_scope='Frozen risk+E_l gate, I omitted, followed by unchanged uniform adjacent attention enhancement.')


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_semantic_probe import FullRiskSemanticProbe
    from features.qwen3_causal_visual_tilt import Qwen3BatchCausalVisualTilt
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT,MODEL
    plan=shared.read(OUT/'protocol.json')
    for n,v in plan['source_sha256'].items():
        assert shared.sha(REPO/n)==v,n
    groups=source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl');path=OUT/f'coco100/{NAME}.shard{gpu}.jsonl'
    assert not path.exists()
    wrapper=build_wrapper();probe=FullRiskSemanticProbe(plan['probe']).to(wrapper.device).eval()
    allowed=frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    eos=plan['eos_token_ids'];eos=set(eos if isinstance(eos,list) else [eos]);memory=[]
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group];static={}
            (span,inputs),=common.same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
            assert [i for i,_ in inputs]==ids and span[1]-span[0]==group[0]['visual_token_count']
            if index==0:
                tokens,_,_=common.generate_group(wrapper,span,inputs,allowed,'sadi',32)
                assert tokens==[r['response_token_ids'][:32] for r in group]
                shared.write(OUT/f'verification.gpu{gpu}.json',dict(passed=True,native32_exact=True,ids=ids))
            holder={}
            def factory(model,current_span,nouns,mode,strength):
                controller=Qwen3BatchCausalVisualTilt(model,current_span,allowed,probe=probe,
                    semantic_tokens=sorted(allowed),tokenizer=wrapper.tokenizer,history_parser=parser,
                    shared_static=static,real_threshold=plan['real_threshold'],tilt_mode='fixed',refresh=False,gamma=1.)
                holder['controller']=controller;return controller
            torch.cuda.reset_peak_memory_stats(wrapper.device)
            with patch.object(original,'Qwen3BatchLayerFeedback',factory):
                tokens,stats,seconds=original.generate_group(wrapper,span,inputs,allowed,shared.POSITIVE,512)
            calls=holder['controller'].steps;holder.clear();decorate(stats,eos)
            texts=wrapper.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            with path.open('a',encoding='utf-8') as h:
                for image,t,text,s in zip(ids,tokens,texts,stats,strict=True):
                    row=dict(image_id=image,model=MODEL,variant=NAME,response_token_ids=t,
                        generated_text=text.strip(),seconds=seconds/len(ids),batch_wall_seconds=seconds,
                        batch_size=len(ids),visual_token_count=span[1]-span[0],sar_ems_statistics=s,
                        decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                        actual_model_calls_in_batch=calls)
                    h.write(json.dumps(row,ensure_ascii=False)+'\n')
            memory.append(dict(group=index,peak=torch.cuda.max_memory_allocated(wrapper.device)))
            shared.write(OUT/f'memory.gpu{gpu}.json',memory)
            print(f'[{gpu}] {NAME} group={index} B={len(ids)} seconds={seconds:.3f}',flush=True)
            static.clear();gc.collect();torch.cuda.empty_cache()
    assert shared.group_keys(source_groups(path))==shared.group_keys(groups)


def run():
    with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):
        driver.run()
    import os
    commands=[[n,str(OUT)] for n in AUDITS]
    shared.write(OUT/'post_audit_status.json',dict(status='running',commands=commands,started=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
    try:
        for i,args in enumerate(commands):
            with (OUT/f'post_audit{i}.log').open('x') as h:
                subprocess.run([sys.executable,*args],cwd=REPO,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
        assert all(shared.read(OUT/n)['passed'] for n in ('mechanism_audit.json','independent_summary.json','analysis_provenance.json'))
        shared.write(OUT/'post_audit_status.json',dict(status='passed',commands=commands,finished=time.time()))
    except BaseException as exc:
        shared.write(OUT/'post_audit_status.json',dict(status='failed',error=repr(exc),failed=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));args=p.parse_args()
    run() if args.worker is None else worker(args.worker)
