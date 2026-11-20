"""Reuse the frozen two-group harness for attention plus FFN alignment."""

import argparse
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_supported_rival_preflight as original
from scripts.run_qwen3_signed_feedback import read,write,sha

ATTENTION=REPO/'outputs/qwen3_supported_rival_preflight10_20261120'
MATH=REPO/'outputs/qwen3_rival_ffn_alignment_math_20261120.json'
MODES=['aux','aux_shifted']


def prepare(root):
    proof=read(MATH);assert proof['passed'] and not proof['cuda_initialized']
    for n,v in proof['source_sha256'].items():assert sha(Path(n))==v,n
    assert read(ATTENTION/'pipeline_complete.json')['passed']
    with patch.object(original,'MODES',MODES):original.prepare(root)
    plan=read(root/'protocol.json')
    files=[Path(__file__),REPO/'features/qwen3_rival_ffn_alignment.py',MATH,ATTENTION/'protocol.json',ATTENTION/'pipeline_complete.json']
    for name in ['calibrated','calibrated_shifted']:
        for shard in [0,1]:
            path=ATTENTION/f'coco10/{name}.shard{shard}.jsonl';target=root/'coco10'/path.name
            assert not target.exists();target.write_bytes(path.read_bytes());files.extend([path,target])
    plan['source_sha256'].update({str(p):sha(p) for p in files})
    plan.update(variants=['sadi','svar','calibrated','calibrated_shifted',*MODES],candidates=MODES,
        ffn_alignment=dict(layer=19,primary_attention='Unchanged calibrated rival ROI attention in both new arms.',
            evidence='Normalized positive rival-target semantic contrast times native write norm, before selecting a3x3 region.',
            objective='HellingerSquared(Q_support,softmax(cos(visual_hpre19,u19)/.03)).',
            operation='Analytic tangent gradient changes only FFN update direction; FP32 norm preserved before BF16 rounding. Step min(attention strength,1), at most4 halvings; retain only actual objective decrease.',
            control='Aux_shifted rolls only FFN evidence by M//2; primary attention ROI and dose stay unchanged.',
            replay='Exact source FFN residual captured before its norm, assert residual+native_update equals native19 anchor. Replace update then replay20--35 current token. Source19 KV and all past slots retained.',
            cost='No extra model, MLP, LM-head, backward or encoder calls relative to the attention proposal.'),
        purpose='Old10 complete-caption and implementation screen, not independent evidence or a500 decision.',
        math_verification_sha256=sha(MATH),attention_controls_source=str(ATTENTION))
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    for p in files:
        if p.suffix=='.py':(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',plan)


def worker(root,shard):
    from features import qwen3_supported_rival_guidance as attention
    from features.qwen3_rival_ffn_alignment import Qwen3RivalFFNAlignment
    with patch.object(original,'MODES',MODES),patch.object(attention,'Qwen3SupportedRivalGuidance',Qwen3RivalFFNAlignment):
        original.worker(root,shard)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:prepare(a.root) if a.prepare else worker(a.root,a.worker)
    except BaseException:
        if a.root.exists():write(a.root/f'aux_failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
