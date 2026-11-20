"""Supply byte-certified copied-control metadata without changing the protocol."""

import argparse
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.run_qwen3_live_preference_contrast10 import check
from scripts import audit_qwen3_caption_with_native_cache_v2 as independent


def prepare(root):
    plan=read(root/'protocol.json');check(plan)
    assert read(root/'pipeline_complete.json')['passed'] and not read(root/'analysis_complete.json')['passed']
    assert "KeyError: 'copied_controls'" in (root/'independent_audit.log').read_text()
    assert not (root/'analysis_recovery_request.json').exists() and not (root/'independent_summary.json').exists()
    controls=[]
    for name in ['sadi','svar']:
        for shard in range(plan['shards']):
            source=Path(plan['source'])/f'{plan["cohort"]}/{name}.shard{shard}.jsonl'
            target=root/f'{plan["cohort"]}/{source.name}'
            assert source.read_bytes()==target.read_bytes()
            controls.append(dict(source=str(source),target=str(target),sha256=sha(source)))
    files=[root/f'{plan["cohort"]}/{name}.shard{shard}.jsonl'
        for name in plan['variants'] for shard in range(plan['shards'])]
    files += list((root/'evaluations').glob('**/metrics.*.json'))
    files += list((root/'evaluations').glob('**/chair_per_image*.jsonl'))
    assert len(files)==len(plan['variants'])*plan['shards']+4
    write(root/'analysis_recovery_request.json',dict(fixed_at=time.time(),
        protocol_sha256=sha(root/'protocol.json'),original_request_sha256=sha(root/'analysis_request.json'),
        source_sha256={str(Path(__file__)):sha(Path(__file__))},
        preserved_sha256={str(p):sha(p) for p in files},copied_controls=['sadi','svar'],
        control_receipts=controls,original_failure_preserved=True,before_new_metric_inspection=True,
        change='Add independently byte-certified copied_controls to the auditor in-memory plan only. Original protocol, raw captions, parser, metrics and intervals unchanged.'))


def run(root):
    plan=read(root/'protocol.json');check(plan);request=read(root/'analysis_recovery_request.json')
    assert request['protocol_sha256']==sha(root/'protocol.json')
    assert request['original_request_sha256']==sha(root/'analysis_request.json')
    assert not (root/'independent_summary.json').exists() and not (root/'analysis_provenance.json').exists()
    for name,wanted in {**request['source_sha256'],**request['preserved_sha256']}.items():assert sha(Path(name))==wanted,name
    for e in request['control_receipts']:
        assert sha(Path(e['source']))==sha(Path(e['target']))==e['sha256']
        assert Path(e['source']).read_bytes()==Path(e['target']).read_bytes()
    assert 'copied_controls' not in plan
    original_read=independent.read
    def read_with_receipts(path):
        value=original_read(path)
        return dict(value,copied_controls=request['copied_controls']) if Path(path).resolve()==(root/'protocol.json').resolve() else value
    with patch.object(independent,'read',read_with_receipts):independent.audit(root)
    assert read(root/'independent_summary.json')['passed'];check(plan)
    for name,wanted in request['preserved_sha256'].items():assert sha(Path(name))==wanted,name
    write(root/'analysis_provenance.json',dict(passed=True,finished=time.time(),
        protocol_sha256=sha(root/'protocol.json'),request_sha256=sha(root/'analysis_request.json'),
        recovery_request_sha256=sha(root/'analysis_recovery_request.json'),physical_review='pending'))
    write(root/'analysis_recovery_complete.json',dict(passed=True,completed_at=time.time()))
    with (root/'quality.log').open('x') as log:
        result=subprocess.run([sys.executable,'scripts/analyze_qwen3_clva_quality.py',str(root)],cwd=REPO,
            stdout=log,stderr=subprocess.STDOUT)
    write(root/'quality_complete.json',dict(passed=result.returncode==0,returncode=result.returncode,completed_at=time.time()))
    assert result.returncode==0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root)
