"""Test positive Q/P-imbalance doses after complete SADI; compare nine arms."""

import argparse
from contextlib import ExitStack,contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import run_qwen3_conflict_route_pilot as base
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver

OUT=shared.OUT/'qwen3_soft_conflict_route_pilot_20261119'
SOURCE=shared.OUT/'qwen3_mild_conflict_route_pilot_20261119'
PREFLIGHT=shared.OUT/'qwen3_soft_conflict_route_preflight_20261119'
MODES={'soft_signal':'signal','soft_flat':'flat','soft_roll':'roll'}
CONTROLS=['sadi','svar','causal_fixed','mild_vri','mild_signal','mild_roll']
FILES=['scripts/run_qwen3_soft_conflict_route_pilot.py','scripts/audit_qwen3_soft_conflict_route_pilot.py',
       'scripts/analyze_qwen3_soft_conflict_route_pilot.py']
AUDITS=[FILES[1],'scripts/audit_qwen3_caption_with_native_cache_v2.py',FILES[2],
        'scripts/audit_qwen3_caption_denominator.py']


@contextmanager
def configured_base():
    with ExitStack() as stack:
        for name,value in dict(OUT=OUT,SOURCE=SOURCE,PREFLIGHT=PREFLIGHT,MODES=MODES,
                               CONTROLS=CONTROLS,FILES=FILES,AUDITS=AUDITS).items():
            stack.enter_context(patch.object(base,name,value))
        yield


def prepare():
    with configured_base():plan=base.prepare()
    assert len(plan['variants'])==9 and plan['candidates']==list(MODES)
    plan.update(stage='soft-conflict-route-development',soft_dose_rule='d_h=relu(R_h-.5); g_h=1-d_h',
        strength_scope='Neutral-zero positive imbalance, fixed before cached dose inspection; no amplitude or threshold search.',
        development_screen='SoftSignal must improve bothCHAIR/falsecounts over softFlat and softRoll; F1 loss<=1.5pp and recall loss<=2pp vsSADI; actual-photo and repetition review.',
        followup='Only useful physical/semantic signal increments can consume fixed reserved1903 independent500; no redraw or validation tuning.')
    plan['recipe'].update(
        risk='R_h=sum_j conditional_visual_attention_hj*P_j/(Q_j+P_j); used as severity, not ranking.',
        vri='All3 select the same at most8 native Conflict-B heads, smallest double-precision VRI, head-index ties.',
        roll='Cyclically move positive doses by half of the selected-head count, within the same VRI-selected set.',
        gates='Signal d=relu(R-.5), g=1-d; flat uses mean selected dose; roll preserves exact local selected-dose multiset; unselected1.',
        previous='Rank-only strong and half suppression lost real content. Test neutral-zero signal-sized doses without changing SADI.',
        cost='All3 compute Q/P; first signal builds image semantic cache, flat/roll share it. One native backward; no108features/MLP/OT/extraencoder.',
        cache='All3 soft captions fresh;6 earlier controls copied byte-identically.',
        saved='Raw generation/evaluators unchanged; case summaries omit duplicated statistical arrays.',
        integration='Pilot adapter only maps route_mode to dose_mode and corrects inherited statistics labels; no numeric method override.')
    shared.write(OUT/'protocol.json',plan)
    analysis=shared.read(OUT/'analysis_protocol.json')
    analysis.update(generation_protocol_sha256=shared.sha(OUT/'protocol.json'),
        primary='Full100 sentence+legacy, all9 arms; primary softSignal vs softFlat/roll, secondary SADI and prior half-strength.',
        cases='All fixes/new and top3 nonzero false/recall changes vsSADI plus every changed false canonical class against softFlat/roll; original photos.',
        storage='Same scalar metrics, counts, text and brief causal events; exclude repeated sar_ems_statistics from changed-text score summaries only.')
    shared.write(OUT/'analysis_protocol.json',analysis)
    return plan


def worker(gpu):
    from features import qwen3_conflict_route_feedback as original
    from features.qwen3_soft_conflict_route_feedback import Qwen3BatchSoftConflictRoute

    class PilotSoftRoute(Qwen3BatchSoftConflictRoute):
        def __init__(self,*args,route_mode,**kwargs):
            super().__init__(*args,dose_mode=route_mode,**kwargs)

        def statistics(self,responses):
            result=super().statistics(responses)
            for stat in result:
                stat.update(parent_legacy_rank_flag=stat['head_ranking_feature_guided'],
                    head_ranking_feature_guided=False,head_dose_feature_guided=True,
                    route_mode=stat['dose_mode'],transport_mode='native_conflict_actual_text_route_soft_dose',
                    borrowed_operator='CRG Conflict-B/VRI selection; Q/P positive-imbalance dose is a project adaptation.')
            return result

    with configured_base(),patch.object(original,'Qwen3BatchConflictRoute',PilotSoftRoute):
        base.worker(gpu)


def run():
    with patch.object(driver,'OUT',OUT),patch.object(driver,'__file__',__file__),patch.object(driver,'prepare',prepare):driver.run()
    commands=[[name,str(OUT)] for name in AUDITS]
    shared.write(OUT/'post_audit_status.json',dict(status='running',commands=commands,started=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
    try:
        for i,args in enumerate(commands):
            with (OUT/f'post_audit{i}.log').open('x') as h:
                subprocess.run([sys.executable,'-u',*args],cwd=REPO,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
        shared.write(OUT/'post_audit_status.json',dict(status='passed',commands=commands,finished=time.time()))
    except BaseException as exc:
        shared.write(OUT/'post_audit_status.json',dict(status='failed',commands=commands,error=repr(exc),failed=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1))
    args=p.parse_args();run() if args.worker is None else worker(args.worker)
