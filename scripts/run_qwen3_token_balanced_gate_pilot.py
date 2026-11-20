"""Use a token-balanced source detector with the unchanged attention action."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_native_sadi_gate_pilot as original

FIT=original.shared.OUT/'qwen3_token_balanced_source_probe_20261120'
DIAG=original.shared.OUT/'qwen3_token_balanced_source_gate_transfer_20261120'
PLANNED=original.shared.OUT/'qwen3_token_balanced_gate_pilot_plan_20261120'
OUT=original.shared.OUT/'qwen3_token_balanced_gate_pilot_20261120'


def config():
    assert original.shared.read(FIT/'verification.json')['passed']
    assert original.shared.read(FIT/'sampling_verification.json')['passed']
    analysis=original.shared.read(DIAG/'analysis.json')
    assert analysis['passed'] and analysis['primary_seed']==43 and analysis['initial_pilot_retention_screen']
    return DIAG,'native_seed43',FIT/'seed43',analysis['thresholds']['native_seed43']


def execute(declare,worker):
    with ExitStack() as stack:
        extras=['scripts/run_qwen3_token_balanced_gate_pilot.py',
                'scripts/fit_qwen3_token_balanced_source_probe.py',
                'scripts/analyze_qwen3_token_balanced_source_gate.py']
        replacements=dict(FIT=FIT,NATIVE_DIAG=DIAG,PLANNED=PLANNED,ACTIVE='token_balanced',
            GATES=('token_balanced',),FILES=[*original.FILES,*extras],
            gate_config=config,output=lambda:OUT,__file__=__file__)
        for name,value in replacements.items():stack.enter_context(patch.object(original,name,value))
        original.declare() if declare else original.execute(worker)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1])
    args=parser.parse_args();execute(args.declare,args.worker)
