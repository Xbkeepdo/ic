import subprocess
from pathlib import Path
out=Path('/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_single500')
with (out/'logs/driver.log').open('a') as log:
 p=subprocess.Popen(['/root/rivermind-data/envs/ic-feature-extract/bin/python','-u',str(out/'run.py')],
                    cwd='/root/rivermind-data/ic-ac',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'logs/driver.pid').write_text(str(p.pid))
print('SINGLE500_LAUNCHED',p.pid)
