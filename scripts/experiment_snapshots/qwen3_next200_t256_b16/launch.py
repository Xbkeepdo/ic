"""Detach the prepared next-layer 200-image experiment once."""
import os
from pathlib import Path
import subprocess
REPO=Path('/root/rivermind-data/ic-ac')
ROOT=REPO/'outputs/generation_time_evidence_drift_qwen3_next200_t256_b16'
assert (ROOT/'cohort.json').is_file()
if (ROOT/'driver.pid').exists(): raise FileExistsError('Driver already launched; inspect before resume')
(ROOT/'logs').mkdir(exist_ok=True)
with (ROOT/'logs/driver.log').open('w') as log:
    p=subprocess.Popen(['/root/rivermind-data/envs/ic-feature-extract/bin/python',
        str(REPO/'scripts/experiment_snapshots/qwen3_next200_t256_b16/run.py')],cwd=REPO,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
(ROOT/'driver.pid').write_text(str(p.pid)+'\n')
print('NEXT200_T256_DRIVER_PID',p.pid)
