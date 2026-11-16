"""Run the two matched shards concurrently, reusing verified unchanged methods."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path('/root/rivermind-data/ic-ac')
ROOT=REPO/'outputs/generation_time_evidence_drift_qwen3_next200_t256_b16'
SNAPSHOT=REPO/'scripts/experiment_snapshots/qwen3_next200_t256_b16'
PYTHON='/root/rivermind-data/envs/ic-feature-extract/bin/python'
def stamp(): return datetime.now(timezone.utc).isoformat()
def state(payload):
    path=ROOT/'run_status.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n')
    temporary.replace(path)

def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    ids=json.loads((ROOT/'cohort.json').read_text())['coco200']
    assert len(ids)==len(set(ids))==200 and ids==sorted(ids)
    assert protocol['max_new_tokens']==256
    assert protocol['selected_sha256']==hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    for name,expected in protocol['source_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source changed after freeze: {name}')
    reused=protocol['verification_reuse']
    for path_key,hash_key in (('path','sha256'),('protocol_path','protocol_sha256')):
        if hashlib.sha256(Path(reused[path_key]).read_bytes()).hexdigest()!=reused[hash_key]:
            raise ValueError('Previous verification provenance changed after freeze')
    if any((ROOT/'coco200').glob('*.jsonl')): raise FileExistsError('Existing captions need review before resume')
    (ROOT/'logs').mkdir(exist_ok=True)
    started=stamp()
    processes,logs=[],[]
    try:
        for shard in range(2):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                     NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
            log=(ROOT/f'logs/gpu{shard}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen([PYTHON,str(SNAPSHOT/'worker.py'),'--output-root',str(ROOT),'--shard',str(shard)],
                              cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT))
        state({'status':'generating','started_utc':started,'driver_pid':os.getpid(),'worker_pids':[p.pid for p in processes]})
        print('BOTH_WORKERS_STARTED',[p.pid for p in processes],flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes): raise RuntimeError('Generation worker failed')
            time.sleep(10)
        if any(p.returncode for p in processes): raise RuntimeError('Generation failed')
        for log in logs: log.close()
        generated=stamp()
        state({'status':'evaluating','started_utc':started,'generation_finished_utc':generated})
        result=subprocess.run([PYTHON,str(SNAPSHOT/'evaluate.py')],cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                              NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data')))
        if result.returncode: raise RuntimeError('Evaluation failed')
        state({'status':'complete','started_utc':started,'generation_finished_utc':generated,'finished_utc':stamp(),
               'evaluation_exit_code':result.returncode})
        print('NEXT200_T256_COMPLETE',flush=True)
    except Exception as exc:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes: p.wait()
        state({'status':'failed','started_utc':started,'finished_utc':stamp(),'error':str(exc),'exit_codes':[p.returncode for p in processes]})
        raise
    finally:
        for log in logs: log.close()

if __name__=='__main__': main()
