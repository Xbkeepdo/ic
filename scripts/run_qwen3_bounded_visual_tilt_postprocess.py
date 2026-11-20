"""Run three frozen CPU audits after the actual bounded visual-tilt old100 driver terminates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/qwen3_bounded_visual_tilt_pilot_20261119'
DRIVER_NAME=b'run_qwen3_bounded_visual_tilt_pilot.py'
AUDITS=['scripts/audit_qwen3_bounded_visual_tilt_pilot.py',
        'scripts/audit_qwen3_caption_with_native_cache_v2.py',
        'scripts/analyze_qwen3_bounded_visual_tilt_pilot.py']


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value,exclusive=False):
    if exclusive:
        with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2);handle.write('\n')
    else:
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def process_identity(pid):
    stat=Path(f'/proc/{pid}/stat')
    try:
        fields=stat.read_text().rsplit(') ',1)[1].split()
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return None
    return dict(state=fields[0],start_ticks=int(fields[19]),
        command_sha256=hashlib.sha256(command).hexdigest(),expected_driver=DRIVER_NAME in command)


def same_live_process(pid,identity):
    current=process_identity(pid)
    return bool(current and current['state']!='Z' and current['start_ticks']==identity['start_ticks']
        and current['command_sha256']==identity['command_sha256']),current


def check_inputs(request):
    assert sha(ROOT/'protocol.json')==request['generation_protocol_sha256']
    assert sha(ROOT/'analysis_protocol.json')==request['analysis_protocol_sha256']
    hashes=dict(read(ROOT/'protocol.json')['source_sha256'])
    hashes.update(read(ROOT/'analysis_protocol.json')['source_sha256'])
    hashes.update(request['extra_source_sha256'])
    for name,expected in hashes.items():assert sha(Path(name))==expected,name
    return len(hashes)


def prepare(pid):
    plan=read(ROOT/'protocol.json');status=read(ROOT/'run_status.json')
    assert plan['stage']=='bounded-visual-development' and plan['images']==100
    assert plan['variants']==['sadi','svar','sadi_sam_acg_guided_19_35','collective','tilt_fixed','tilt_drift','tilt_evidence']
    assert plan['max_batch']==16 and plan['max_new_tokens']==512
    assert status['status'] in ('generating','evaluating','complete')
    assert not (ROOT/'postprocess_request.json').exists(),'Inspect the existing postprocessor; never launch another.'
    assert not (ROOT/'post_audit_status.json').exists(),'Audits already claimed by another process.'
    identity=process_identity(pid)
    assert identity and identity['state']!='Z' and identity['expected_driver'],(pid,identity)
    if 'pid' in status:assert status['pid']==pid
    request=dict(queued=time.time(),driver_pid=pid,driver_identity=identity,
        generation_protocol_sha256=sha(ROOT/'protocol.json'),analysis_protocol_sha256=sha(ROOT/'analysis_protocol.json'),
        extra_source_sha256={str(Path(__file__).resolve()):sha(Path(__file__))},commands=[[n,str(ROOT)] for n in AUDITS],
        scope='One-time CPU postprocessing of this live driver. No generation, batching, method, parameter or cohort changes.',
        terminal_rule='Wait for the original PID/start-time/command identity to terminate; then require completed upstream status.',
        conflict_rule='Claim post_audit_status exclusively; preserve failure records and never retry or overwrite existing audit logs.')
    count=check_inputs(request)
    return request,count


def run(request):
    state_path=ROOT/'postprocess_state.json';post_path=ROOT/'post_audit_status.json'
    write(ROOT/'postprocess_request.json',request,exclusive=True)
    write(ROOT/'postprocess_pid.json',dict(pid=os.getpid()),exclusive=True)
    request_sha=sha(ROOT/'postprocess_request.json');phase='waiting_driver';claimed=False
    try:
        while True:
            alive,identity=same_live_process(request['driver_pid'],request['driver_identity'])
            upstream=read(ROOT/'run_status.json')
            write(state_path,dict(status='waiting',phase=phase,pid=os.getpid(),checked=time.time(),
                request_sha256=request_sha,driver_live_identity_verified=alive,driver_observed=identity,
                upstream_status=upstream['status']))
            if upstream['status']=='failed':raise RuntimeError('Upstream driver failed; inspect original logs. No retry.')
            if not alive:break
            time.sleep(10)
        assert upstream['status']=='complete',upstream
        phase='checking_frozen_inputs';check_inputs(request)
        assert sha(ROOT/'postprocess_request.json')==request_sha
        assert not any(ROOT.glob('post_audit[0-9].log'))
        write(post_path,dict(status='running',pid=os.getpid(),started=time.time(),
            commands=request['commands'],postprocess_request_sha256=request_sha),exclusive=True)
        claimed=True;phase='auditing'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
            NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
        for index,args in enumerate(request['commands']):
            write(state_path,dict(status='running',phase=phase,pid=os.getpid(),command=index,args=args,
                checked=time.time(),request_sha256=request_sha,upstream_actual_process_terminal=True))
            print('START_AUDIT',index,args[0],flush=True)
            with (ROOT/f'post_audit{index}.log').open('x') as log:
                subprocess.run([sys.executable,*args],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('PASS_AUDIT',index,args[0],flush=True)
        check_inputs(request)
        for name in ('mechanism_audit.json','independent_summary.json','analysis_provenance.json'):
            assert read(ROOT/name)['passed'],name
        finished=time.time()
        write(post_path,dict(status='passed',commands=request['commands'],finished=finished,
            postprocess_request_sha256=request_sha,analysis_protocol_sha256=request['analysis_protocol_sha256']))
        write(state_path,dict(status='complete',phase='complete',pid=os.getpid(),finished=finished,
            request_sha256=request_sha,upstream_actual_process_terminal=True,all_three_audits_passed=True))
        print('ALL_BOUNDED_TILT100_POST_AUDITS_PASSED',flush=True)
    except BaseException as exc:
        if claimed:write(post_path,dict(status='failed',commands=request['commands'],error=repr(exc),
            failed=time.time(),postprocess_request_sha256=request_sha))
        write(state_path,dict(status='failed',phase=phase,pid=os.getpid(),error=repr(exc),
            failed=time.time(),request_sha256=request_sha))
        traceback.print_exc();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver',type=int,required=True);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();request,count=prepare(args.driver)
    if args.check:print(json.dumps(dict(ready=True,driver=args.driver,identity=request['driver_identity'],verified_source_files=count)))
    else:run(request)
