"""Freeze a single half-attenuation policy; run implementation checks on both GPUs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_qwen3_evidence_head_routing_preflight import read,write,sha,identity

ROOT=REPO/'outputs/qwen3_mild_conflict_route_preflight_20261119'
CPU=REPO/'outputs/qwen3_mild_conflict_route_cpu_20261119'
SOURCE=REPO/'outputs/qwen3_conflict_route_pilot_20261119'
FILES=['features/qwen3_mild_conflict_route_feedback.py','scripts/verify_qwen3_mild_conflict_route.py',
       'scripts/verify_qwen3_mild_conflict_route_gpu.py','scripts/run_qwen3_mild_conflict_route_preflight.py']


def run():
    assert not ROOT.exists() and read(CPU/'verification.json')['passed']
    assert read(SOURCE/'final_provenance.json')['passed']
    hashes=dict(read(SOURCE/'protocol.json')['source_sha256'])
    for n,v in read(CPU/'verification.json')['source_sha256'].items():assert sha(REPO/n)==v,n
    for path in [CPU/'request.json',CPU/'verification.json',SOURCE/'protocol.json',SOURCE/'final_provenance.json',
                 *(REPO/n for n in FILES),REPO/'scripts/run_qwen3_evidence_head_routing_preflight.py']:
        hashes[str(path)]=sha(path)
    def check():
        for n,v in hashes.items():assert sha(REPO/n)==v,n
    request=dict(fixed_at=time.time(),source_sha256=hashes,new_files=FILES,preflight_only=True,
        formal_attenuation_scale=.5,formal_selected_gate_range=[.5,.75],endpoint_scales_diagnostic_only=[0.,1.],
        base_layers=[5,18],route_layers=list(range(19,36)),batch_labels=['B2','B16'],max_new_tokens=32,
        source='Frozen original rank/eligibility/gradients/routes/acceptance, change actual text attenuation only.',
        required='Scale0/disable/gamma0/zero/reject exactSADI tokens/logits/current36KV;scale1 exactstrong;flat exactmildVRI;historyKV;singleencoder;noMLP/OT.',
        maximum_peak_bytes=30*1024**3)
    check();ROOT.mkdir();write(ROOT/'request.json',request,True)
    for n in FILES:
        target=ROOT/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,target)
    write(ROOT/'parent_identity.json',dict(pid=os.getpid(),identity=identity(os.getpid())),True)
    children=[]
    try:
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert len(used.splitlines())==2 and max(map(int,used.splitlines()))<100
        workers=[]
        for gpu,label in enumerate(['B2','B16']):
            command=[sys.executable,'-u','scripts/verify_qwen3_mild_conflict_route_gpu.py','--batch-label',label]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
            with (ROOT/f'{label}.log').open('x') as h:p=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT)
            children.append(p);workers.append(dict(pid=p.pid,label=label,gpu=gpu,identity=identity(p.pid),command=command))
        write(ROOT/'launch.json',dict(started=time.time(),workers=workers,request_sha256=sha(ROOT/'request.json')),True)
        write(ROOT/'run_status.json',dict(status='running',pid=os.getpid(),workers=workers))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Mild route preflight failed; retain sources and logs')
            time.sleep(3)
        assert all(p.returncode==0 for p in children);check();parts={}
        for label in ['B2','B16']:
            part=read(ROOT/f'verification.{label}.json');assert part['passed'] and part['request_sha256']==sha(ROOT/'request.json')
            assert set(part['batches'])=={label};parts.update(part['batches'])
        write(ROOT/'verification.json',dict(passed=True,batches=parts,request_sha256=sha(ROOT/'request.json'),
            source_sha256={n:sha(REPO/n) for n in FILES},all_frozen_inputs_verified=len(hashes),
            terminal_workers={str(p.pid):identity(p.pid) for p in children},finished=time.time()),True)
        write(ROOT/'run_status.json',dict(status='complete',pid=os.getpid(),finished=time.time()));print('MILD_ROUTE_PREFLIGHT_PASSED',flush=True)
    except BaseException as exc:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:p.wait()
        write(ROOT/'run_status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),failed=time.time()));raise


if __name__=='__main__':run()
