"""Run GPU validation once, then both paired shards and the frozen evaluation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path('/root/rivermind-data/ic-ac')
ROOT = REPO / 'outputs/generation_time_evidence_drift_qwen3_layer_feedback100_b16'
SNAPSHOT = REPO / 'scripts/experiment_snapshots/qwen3_layer_feedback100_b16'
PYTHON = '/root/rivermind-data/envs/ic-feature-extract/bin/python'

def stamp(): return datetime.now(timezone.utc).isoformat()

def state(payload):
    path = ROOT / 'run_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)

def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    ids = json.loads((ROOT / 'cohort.json').read_text())['coco100']
    assert len(ids) == len(set(ids)) == 100 and ids == sorted(ids)
    assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == protocol['selected_sha256']
    for rel, expected in protocol['source_sha256'].items():
        if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed after freeze: {rel}')
    if any((ROOT / 'coco100').glob('*.jsonl')) or (ROOT / 'verification.json').exists():
        raise FileExistsError('Existing output requires explicit audit before resume')
    (ROOT / 'logs').mkdir(exist_ok=True)
    started = stamp()
    processes, logs = [], []
    def launch(shard, verify=False):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', NLTK_DATA=str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
        log = (ROOT / f'logs/gpu{shard}.log').open('w')
        logs.append(log)
        command = [PYTHON, str(REPO / 'scripts/generation_time_qwen3_layer_feedback.py'),
                   '--output-root', str(ROOT), '--shard', str(shard)]
        if verify: command.append('--verify')
        p = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(p)
    try:
        launch(0, verify=True)
        state({'status': 'validating', 'started_utc': started, 'driver_pid': os.getpid(), 'worker_pids': [p.pid for p in processes]})
        while not (ROOT / 'verification.json').exists():
            if processes[0].poll() is not None:
                raise RuntimeError('GPU preflight failed before generation; inspect gpu0.log')
            time.sleep(3)
        launch(1)
        state({'status': 'generating', 'started_utc': started, 'driver_pid': os.getpid(), 'worker_pids': [p.pid for p in processes]})
        print('BOTH_WORKERS_STARTED', [p.pid for p in processes], flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Generation worker failed')
            time.sleep(10)
        if any(p.returncode for p in processes): raise RuntimeError('Generation failed')
        for log in logs: log.close()
        generation_finished = stamp()
        state({'status': 'evaluating', 'started_utc': started, 'generation_finished_utc': generation_finished})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NLTK_DATA=str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
        result = subprocess.run([PYTHON, str(SNAPSHOT / 'evaluate.py')], cwd=REPO, env=env)
        if result.returncode: raise RuntimeError('Evaluation failed')
        state({'status': 'complete', 'started_utc': started, 'generation_finished_utc': generation_finished,
               'finished_utc': stamp(), 'evaluation_exit_code': result.returncode})
        print('LAYER_FEEDBACK100_COMPLETE', flush=True)
    except Exception as exc:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes: p.wait()
        state({'status': 'failed', 'started_utc': started, 'finished_utc': stamp(),
               'error': str(exc), 'exit_codes': [p.returncode for p in processes]})
        raise
    finally:
        for log in logs: log.close()

if __name__ == '__main__': main()
