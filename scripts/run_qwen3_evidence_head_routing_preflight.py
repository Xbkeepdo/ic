"""Freeze the head-allocation prototype and check B2/B16 on separate GPUs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/qwen3_evidence_head_routing_preflight_20261119'
CPU=REPO/'outputs/qwen3_evidence_head_routing_cpu_20261119'
UPSTREAM=REPO/'outputs/qwen3_risk_semantic_gate_pilot_20261119'
GATE=REPO/'outputs/qwen3_block_acg_pilot_20261118'
FILES=['features/qwen3_evidence_head_routing.py','features/qwen3_evidence_head_feedback.py',
    'scripts/verify_qwen3_evidence_head_routing.py','scripts/verify_qwen3_evidence_head_routing_gpu.py',
    'scripts/run_qwen3_evidence_head_routing_preflight.py']


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,value,exclusive=False):
    if exclusive:
        with p.open('x',encoding='utf-8') as h:json.dump(value,h,indent=2);h.write('\n')
    else:
        temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(p)


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(state=fields[0],start_ticks=int(fields[19]),
            command_sha256=hashlib.sha256(Path(f'/proc/{pid}/cmdline').read_bytes()).hexdigest())
    except FileNotFoundError:
        return dict(state='missing')


def check(request):
    for n,v in request['source_sha256'].items():
        assert sha(REPO/n)==v,n


def run():
    assert not ROOT.exists(), 'Inspect the existing job and preserve its frozen sources'
    assert read(CPU/'verification.json')['passed']
    assert read(UPSTREAM/'final_provenance.json')['passed']
    assert read(UPSTREAM/'run_status.json')['status']=='complete'
    hashes=dict(read(UPSTREAM/'protocol.json')['source_sha256'])
    paths=[CPU/'request.json',CPU/'verification.json',UPSTREAM/'final_provenance.json',
        GATE/'protocol.json',GATE/'object_lexicon.json',*(REPO/n for n in FILES)]
    for p in paths:hashes[str(p)]=sha(p)
    for n,v in read(CPU/'verification.json')['source_sha256'].items():
        assert sha(Path(n))==v,n
    request=dict(fixed_at=time.time(),source_sha256=hashes,new_files=FILES,
        preflight_only=True,pilot_generation_authorized_here=False,
        batch_labels=['B2','B16'],max_new_tokens=32,full_sadi_layers=[5,18],
        sources=list(range(19,35)),targets=list(range(20,36)),
        gate='Original108 unchanged; diagnostic force flag is confined to implementation checks.',
        score='D times centered/max-absolute-scaled BC(head,Q)-BC(head,P); max scale clamped only at FP32 epsilon.',
        reference='.5*mean_abs_actual_visual_logit per head, constant across visual positions.',
        matching='32 image-wise bisections to match mean visual attention mass across heads; no cross-image scalar.',
        controls='Native disabled/gamma0/actual zero replay/forced rejection; flat and uniform exact independent constant-bias reference.',
        required='Original108 inputs, actual preceding attention/FFN hooks, historical/current KV, one encoder, no backward, peak<29GiB.',
        fp32_mean_mass_tolerance=32*2**-23,bf16_mean_mass_tolerance=2**-7,
        conditional_visual_tolerance=1e-5,
        inspiration=['https://arxiv.org/html/2604.00983','https://aclanthology.org/2026.findings-acl.425/',
            'https://github.com/jijy20/CausalLens/tree/7d93bf291f8d3052926c4d058a7b4f8435ded407'],
        adaptation_scope='New matched head allocation using this project Q/P; no full ACT/AHEAD/CausalLens reproduction.')
    check(request);ROOT.mkdir();write(ROOT/'request.json',request,True)
    for n in FILES:
        target=ROOT/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,target)
    write(ROOT/'parent_identity.json',dict(pid=os.getpid(),identity=identity(os.getpid())),True)
    children=[];phase='waiting_for_idle_gpus'
    try:
        while True:
            used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if len(used.splitlines())==2 and max(map(int,used.splitlines()))<100:break
            write(ROOT/'run_status.json',dict(status='waiting',phase=phase,checked=time.time(),memory=used))
            time.sleep(10)
        check(request);phase='gpu_preflight';workers=[]
        for gpu,label in enumerate(('B2','B16')):
            command=[sys.executable,'-u','scripts/verify_qwen3_evidence_head_routing_gpu.py','--batch-label',label]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
            with (ROOT/f'{label}.log').open('x') as h:
                process=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
            children.append(process);workers.append(dict(pid=process.pid,gpu=gpu,label=label,
                identity=identity(process.pid),command=command))
        write(ROOT/'launch.json',dict(started=time.time(),workers=workers,request_sha256=sha(ROOT/'request.json')),True)
        write(ROOT/'run_status.json',dict(status='running',phase=phase,pid=os.getpid(),workers=workers))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Head-routing preflight failed; preserve logs and sources')
            time.sleep(3)
        assert all(p.returncode==0 for p in children);check(request);parts={}
        for label in ('B2','B16'):
            part=read(ROOT/f'verification.{label}.json')
            assert part['passed'] and part['request_sha256']==sha(ROOT/'request.json')
            assert set(part['batches'])=={label};parts.update(part['batches'])
        write(ROOT/'verification.json',dict(passed=True,batches=parts,request_sha256=sha(ROOT/'request.json'),
            source_sha256={n:sha(REPO/n) for n in FILES},all_frozen_inputs_verified=len(hashes),
            terminal_workers={str(p.pid):identity(p.pid) for p in children},finished=time.time(),
            scope='Implementation verification, not mitigation effectiveness.'),True)
        write(ROOT/'run_status.json',dict(status='complete',pid=os.getpid(),finished=time.time()))
        print('EVIDENCE_HEAD_ROUTING_B2_B16_PASSED',flush=True)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait()
        write(ROOT/'run_status.json',dict(status='failed',phase=phase,pid=os.getpid(),error=repr(exc),failed=time.time()))
        raise


if __name__=='__main__':
    run()
