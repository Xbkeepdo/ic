"""Test predeclared native-caption gates with the unchanged post-SADI action."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_native_sadi_gate_pilot as original

FIT = original.shared.OUT / 'qwen3_native_caption_probe_recovery_20261120'
DIAG = original.shared.OUT / 'qwen3_native_caption_gate_transfer_recovery_20261120'
PLANNED = original.shared.OUT / 'qwen3_native_caption_gate_pilot_plan_20261120'
GATES = ('caption_seed43', 'caption_original108')
ACTIVE = GATES[0]


def output():
    return original.shared.OUT / f'qwen3_native_{ACTIVE}_gate_pilot_20261120'


def config():
    assert original.shared.read(FIT / 'verification.json')['passed']
    analysis = original.shared.read(DIAG / 'analysis.json'); assert analysis['passed']
    key = {'caption_seed43': 'seed43', 'caption_original108': 'original108'}[ACTIVE]
    passed = analysis['fork_retention_and_real_fp_screens'][key]
    if not passed:
        review = original.shared.read(DIAG / 'human_review.json')
        assert review['analysis_sha256'] == original.shared.sha(DIAG / 'analysis.json')
        assert review['new_image_confirmed_physical_targets'][key]
        assert analysis['counts'][key]['automatic_real_triggered'] <= 17
    calibration = original.shared.read(FIT / 'calibration.json')[key]
    assert calibration['threshold'] == analysis['thresholds'][key]
    return DIAG, key, Path(calibration['model']), calibration['threshold']


def execute(declare, worker):
    extras = ['scripts/run_qwen3_native_caption_gate_pilot.py',
        'scripts/trace_qwen3_native_caption_source_recovery.py',
        'scripts/fit_qwen3_native_caption_probe_recovery.py',
        'scripts/analyze_qwen3_native_caption_gate_recovery.py']
    with ExitStack() as stack:
        replacements = dict(FIT=FIT, NATIVE_DIAG=DIAG, PLANNED=PLANNED, ACTIVE=ACTIVE,
            GATES=GATES, FILES=[*original.FILES, *extras], gate_config=config,
            output=output, __file__=__file__)
        for name, value in replacements.items(): stack.enter_context(patch.object(original, name, value))
        original.declare() if declare else original.execute(worker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', action='store_true')
    parser.add_argument('--worker', type=int, choices=[0, 1])
    parser.add_argument('--gate', choices=GATES, default=os.environ.get('IC_NATIVE_SOURCE_GATE', GATES[0]))
    args = parser.parse_args(); ACTIVE = args.gate
    execute(args.declare, args.worker)
