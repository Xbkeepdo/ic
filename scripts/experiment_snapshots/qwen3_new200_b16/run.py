"""Run both fixed interventions on two GPUs, then evaluate the new 200 images."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path('/root/rivermind-data/ic-ac')
ROOT=REPO/'outputs/generation_time_evidence_drift_qwen3_new200_b16'
SCRIPT=REPO/'scripts/generation_time_evidence_drift_qwen3_batched.py'
PYTHON='/root/rivermind-data/envs/ic-feature-extract/bin/python'
SNAPSHOT=REPO/'scripts/experiment_snapshots/qwen3_new200_b16'

def stamp(): return datetime.now(timezone.utc).isoformat()
def state(payload):
    path=ROOT/'run_status.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n')
    temporary.replace(path)

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    ids=json.loads((ROOT/'cohort.json').read_text())['coco200']
    if len(ids)!=200 or len(set(ids))!=200 or ids!=sorted(ids):
        raise ValueError('Expected 200 unique sorted image IDs')
    if hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=protocol['selected_sha256']:
        raise ValueError('Cohort differs from the frozen protocol')
    for rel,expected in protocol['source_sha256'].items():
        if hashlib.sha256((REPO/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source changed after protocol freeze: {rel}')
    if any((ROOT/'coco200').glob('*.jsonl')):
        raise FileExistsError('Existing captions require an explicit batch audit before resuming')
    (ROOT/'logs').mkdir(exist_ok=True)
    started=stamp()
    processes=[]
    handles=[]
    for shard in range(2):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
        log=(ROOT/f'logs/gpu{shard}.log').open('w')
        handles.append(log)
        cmd=[PYTHON,str(SCRIPT),'--output-root',str(ROOT),'--cohort','coco200','--shards','2','--shard',str(shard),'--batch-size','16','--variants','svar','svar_sar_ems_attention']
        process=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append(process)
    state({'status':'generating','started_utc':started,'driver_pid':os.getpid(),'worker_pids':[p.pid for p in processes]})
    print('STARTED',started,'workers',[p.pid for p in processes],flush=True)
    while any(p.poll() is None for p in processes):
        failed=[p.returncode for p in processes if p.poll() not in (None,0)]
        if failed:
            for p in processes:
                if p.poll() is None: p.terminate()
            for p in processes: p.wait()
            state({'status':'generation_failed','started_utc':started,'finished_utc':stamp(),'exit_codes':[p.returncode for p in processes]})
            raise RuntimeError('Worker failed; consult GPU logs before any resume')
        time.sleep(10)
    for log in handles: log.close()
    if any(p.returncode for p in processes):
        state({'status':'generation_failed','started_utc':started,'finished_utc':stamp(),'exit_codes':[p.returncode for p in processes]})
        raise RuntimeError('Generation failed')
    state({'status':'evaluating','started_utc':started,'generation_finished_utc':stamp(),'driver_pid':os.getpid()})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
    result=subprocess.run([PYTHON,str(SNAPSHOT/'evaluate.py')],cwd=REPO,env=env)
    state({'status':'complete' if result.returncode==0 else 'evaluation_failed','started_utc':started,'finished_utc':stamp(),'evaluation_exit_code':result.returncode})
    if result.returncode: raise RuntimeError('Evaluation failed')
    print('NEW200_B16_COMPLETE',flush=True)

if __name__=='__main__': main()
