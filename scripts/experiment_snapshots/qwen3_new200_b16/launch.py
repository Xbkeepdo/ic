"""Detach the already prepared experiment driver and record its PID."""
import os
from pathlib import Path
import subprocess

REPO=Path('/root/rivermind-data/ic-ac')
ROOT=REPO/'outputs/generation_time_evidence_drift_qwen3_new200_b16'
if not (ROOT/'cohort.json').is_file(): raise FileNotFoundError('Prepare the frozen cohort first')
if (ROOT/'driver.pid').exists(): raise FileExistsError('A driver was already launched; inspect it before relaunching')
(ROOT/'logs').mkdir(exist_ok=True)
with (ROOT/'logs/driver.log').open('w') as log:
    process=subprocess.Popen(['/root/rivermind-data/envs/ic-feature-extract/bin/python',str(REPO/'scripts/experiment_snapshots/qwen3_new200_b16/run.py')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
(ROOT/'driver.pid').write_text(str(process.pid)+'\n')
print('NEW200_DRIVER_PID',process.pid)
