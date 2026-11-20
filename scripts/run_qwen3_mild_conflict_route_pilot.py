"""Change only route attenuation to one half; compare nine frozen/fresh arms."""

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

OUT=shared.OUT/'qwen3_mild_conflict_route_pilot_20261119'
SOURCE=shared.OUT/'qwen3_conflict_route_pilot_20261119'
PREFLIGHT=shared.OUT/'qwen3_mild_conflict_route_preflight_20261119'
MODES={'mild_vri':'vri','mild_signal':'signal','mild_roll':'roll'}
CONTROLS=['sadi','svar','causal_fixed','route_vri','route_signal','route_roll']
FILES=['scripts/run_qwen3_mild_conflict_route_pilot.py','scripts/audit_qwen3_mild_conflict_route_pilot.py',
       'scripts/analyze_qwen3_mild_conflict_route_pilot.py']
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
    plan.update(stage='mild-conflict-route-development',attenuation_scale=.5,
        strength_scope='One prospective amplitude ablation on old development100; no other parameter search.',
        development_screen='Signal must improve bothCHAIR/falsecounts over mildVRI and roll; F1 loss<=1.5pp and recall loss<=2pp vsSADI, no loop/denominator-only gains; original-image review.',
        followup='Only useful physical/semantic signal increments can consume fixed reserved1903 independent500. No redraw or validation tuning.')
    plan['recipe'].update(attenuation_scale=.5,
        gates='g_mild=1-.5*(1-g_strong), selectedgates .5--.75, others1; same ranking rule and conflict definition, not same eventual heads/trajectories.',
        previous='Full-strength19--35 route pilot lost F1/recall and introduced loops. Only text-removal amplitude is halved here.',
        cache='All3 mild candidate captions generated fresh;6 prior controls copied byte-identically.',
        saved='Raw generation and evaluators unchanged; case summaries omit duplicated per-image statistical arrays.')
    shared.write(OUT/'protocol.json',plan)
    analysis=shared.read(OUT/'analysis_protocol.json')
    analysis.update(generation_protocol_sha256=shared.sha(OUT/'protocol.json'),
        primary='Full100 sentence+legacy, all9 arms; primary mildSignal vs mildVRI/roll, secondary full-strength amplitude comparisons.',
        cases='All fixes/new and top3 nonzero false/recall changes vsSADI plus every changed false canonical class against mildVRI/roll; original photos.',
        storage='Same scalar metrics, counts, text and brief causal events; exclude repeated sar_ems_statistics from changed-text score summaries only.')
    shared.write(OUT/'analysis_protocol.json',analysis)
    return plan


def worker(gpu):
    from features import qwen3_conflict_route_feedback as original
    from features.qwen3_mild_conflict_route_feedback import Qwen3BatchMildConflictRoute
    with configured_base(),patch.object(original,'Qwen3BatchConflictRoute',Qwen3BatchMildConflictRoute):
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
