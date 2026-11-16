import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path('/root/rivermind-data/ic-ac')
previous = repo/'outputs/generation_time_evidence_drift_qwen3_coco500'
out = repo/'outputs/generation_time_evidence_drift_qwen3_single500'
python = '/root/rivermind-data/envs/ic-feature-extract/bin/python'
out.mkdir(exist_ok=True)
(out/'logs').mkdir(exist_ok=True)
(out/'coco500').mkdir(exist_ok=True)
manifest=json.loads((previous/'cohort.json').read_text())
assert len(manifest['coco500']) == len(set(manifest['coco500'])) == 500
dest=out/'cohort.json'
if dest.exists():
    assert json.loads(dest.read_text()) == manifest
else:
    dest.write_text(json.dumps(manifest,indent=2)+'\n')
protocol={
 'model':'Qwen3-VL-8B-Instruct',
 'variants':['svar','svar_sar_ems_attention'],
 'cohort_source':str(previous/'cohort.json'),
 'sample':'same frozen 500 COCO images; nested prefixes use original seed43 draw',
 'prompt':'Describe this image.',
 'max_new_tokens':512,'dtype':'bfloat16','batch_size':1,'do_sample':False,
 'shards':2,'recipe':{'base_layers':[5,18],'svar_strength':0.5,'write_layer':18,'ffn_layer':20,
                     'late_layers':[21,22],'strength':5,'tau':0.03},
 'source_sha256':{},
 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
}
for rel in ['features/qwen3_sar_ems_attention.py','features/llava_sar_ems_attention.py',
            'scripts/generation_time_evidence_drift_qwen3.py','scripts/generation_time_sar_ems_seed43.py',
            'configs/model_configs_coco4000_512_endac_baselines_v2.yaml']:
    protocol['source_sha256'][rel]=hashlib.sha256((repo/rel).read_bytes()).hexdigest()
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
environment=os.environ.copy()
environment.update(NLTK_DATA=str(repo/'outputs/generation_time_evidence_drift_llava/nltk_data'),
                   TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
processes=[]
for shard in (0,1):
    environment['CUDA_VISIBLE_DEVICES']=str(shard)
    command=[python,'-u','scripts/generation_time_evidence_drift_qwen3.py',
             '--output-root',str(out),'--cohort','coco500','--shards','2','--shard',str(shard),
             '--variants','svar','svar_sar_ems_attention']
    with (out/f'logs/gpu{shard}.log').open('a') as log:
        p=subprocess.Popen(command,cwd=repo,env=environment.copy(),stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT)
    processes.append(p)
    (out/f'logs/gpu{shard}.pid').write_text(str(p.pid))
codes=[p.wait() for p in processes]
if any(codes):
    raise RuntimeError(f'Generation workers exited {codes}')
environment['CUDA_VISIBLE_DEVICES']=''
subprocess.run([python,'-u',str(out/'evaluate.py')],cwd=repo,env=environment,check=True)
print('SINGLE500_COMPLETE',flush=True)
