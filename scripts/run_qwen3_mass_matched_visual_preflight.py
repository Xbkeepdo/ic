"""Wait for the frozen500 GPU workers, then run bounded B2/B16 preflight once."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO/'outputs/qwen3_mass_matched_visual_preflight_20261119'
UPSTREAM = REPO/'outputs/qwen3_causal_visual_tilt_validation500_20261119'
CPU = REPO/'outputs/qwen3_mass_matched_visual_cpu_20261119'
OLD_WORKERS = {709809: 89767940, 709810: 89767940}
FILES = ['features/qwen3_mass_matched_visual.py', 'features/qwen3_mass_matched_feedback.py',
         'scripts/verify_qwen3_mass_matched_visual.py', 'scripts/verify_qwen3_mass_matched_visual_gpu.py',
         'scripts/run_qwen3_mass_matched_visual_preflight.py']


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    if exclusive:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2); handle.write('\n')
    else:
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8'); temp.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        return dict(state=fields[0], start_ticks=int(fields[19]), command_sha256=hashlib.sha256(command).hexdigest())
    except FileNotFoundError:
        return dict(state='missing')


def check_inputs(request):
    for name, value in request['source_sha256'].items():
        assert sha(REPO/name) == value, name


def run():
    assert not ROOT.exists(), 'Inspect the queued/running/finished preflight; do not launch twice'
    assert read(CPU/'verification.json')['passed']
    upstream_plan = read(UPSTREAM/'protocol.json')
    assert upstream_plan['stage'] == 'independent-causal500'
    assert read(UPSTREAM/'run_status.json')['status'] in ('generating', 'evaluating', 'complete')
    hashes = dict(upstream_plan['source_sha256'])
    for path in [UPSTREAM/'protocol.json', CPU/'request.json', CPU/'verification.json', *(REPO/name for name in FILES)]:
        hashes[str(path)] = sha(path)
    for name, value in read(CPU/'verification.json')['source_sha256'].items():
        assert sha(Path(name)) == value, name
    request = dict(fixed_at=time.time(), source_sha256=hashes, new_files=FILES,
        upstream=str(UPSTREAM), upstream_protocol_sha256=sha(UPSTREAM/'protocol.json'),
        workers_to_wait=OLD_WORKERS, new_preflight_only=True, pilot_generation_authorized_here=False,
        max_new_tokens=32, batch_labels=['B2', 'B16'], full_sadi_layers=[5, 18],
        target_layers=list(range(20, 36)), reference_uniform_coefficient=.5, solver_iterations=32,
        gamma=1., raw_contrast='D*(2*Q/(Q+P)-1)',
        controls='SADI disabled/gamma0/actual zero replay/forced rejection; flat contrast exactly equals frozen gated uniform reference.',
        checks='Original108 inputs, full historical/current KV, actual preceding attention/FFN hooks, no backward, one encoder, matching errors, peak<29GiB.',
        fp32_mass_tolerance=32*2**-23, bf16_rounding_bound=2**-7,
        scope='A new prototype check; current500 and its frozen358 sources remain unchanged.')
    check_inputs(request)
    ROOT.mkdir()
    write(ROOT/'request.json', request, exclusive=True)
    write(ROOT/'queue_identity.json', dict(pid=os.getpid(), identity=identity(os.getpid())), exclusive=True)
    children = []; phase = 'waiting_upstream_gpus'
    try:
        while True:
            upstream_status = read(UPSTREAM/'run_status.json')['status']
            if upstream_status == 'failed':
                raise RuntimeError('Current500 failed; preserve it and inspect before any new GPU work')
            observed = {str(pid): identity(pid) for pid in OLD_WORKERS}
            live = any(info['state'] not in ('missing', 'Z') and info.get('start_ticks') == OLD_WORKERS[int(pid)]
                       for pid, info in observed.items())
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            free = len(memory.splitlines()) == 2 and max(map(int, memory.splitlines())) < 100
            write(ROOT/'run_status.json', dict(status='waiting', phase=phase, pid=os.getpid(), checked=time.time(),
                upstream_status=upstream_status, original_workers=observed, memory=memory))
            if not live and free and upstream_status in ('evaluating', 'complete'):
                break
            time.sleep(10)
        check_inputs(request); phase = 'gpu_preflight'
        workers = []
        for gpu, label in enumerate(('B2', 'B16')):
            command = [sys.executable, '-u', 'scripts/verify_qwen3_mass_matched_visual_gpu.py', '--batch-label', label]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
            with (ROOT/f'{label}.log').open('x') as handle:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT)
            children.append(process)
            workers.append(dict(pid=process.pid, gpu=gpu, label=label, identity=identity(process.pid), command=command))
        write(ROOT/'launch.json', dict(started=time.time(), workers=workers, request_sha256=sha(ROOT/'request.json')), exclusive=True)
        write(ROOT/'run_status.json', dict(status='running', phase=phase, pid=os.getpid(), workers=workers, checked=time.time()))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Mass-matched GPU preflight failed; inspect preserved case log')
            time.sleep(3)
        assert all(p.returncode == 0 for p in children)
        check_inputs(request); phase = 'merging'
        parts = {}
        for label in ('B2', 'B16'):
            part = read(ROOT/f'verification.{label}.json')
            assert part['passed'] and part['request_sha256'] == sha(ROOT/'request.json')
            assert set(part['batches']) == {label}
            parts.update(part['batches'])
        result = dict(passed=True, batches=parts, request_sha256=sha(ROOT/'request.json'),
            source_sha256={name: sha(REPO/name) for name in FILES},
            all_frozen_inputs_verified=len(hashes), terminal_workers={str(p.pid):identity(p.pid) for p in children},
            finished=time.time(), scope='Implementation preflight only, not hallucination-mitigation effectiveness.')
        write(ROOT/'verification.json', result, exclusive=True)
        write(ROOT/'run_status.json', dict(status='complete', pid=os.getpid(), finished=time.time()))
        print('MASS_MATCHED_B2_B16_PREFLIGHT_PASSED', flush=True)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        write(ROOT/'run_status.json', dict(status='failed', phase=phase, pid=os.getpid(), error=repr(exc), failed=time.time()))
        raise


if __name__ == '__main__':
    run()
