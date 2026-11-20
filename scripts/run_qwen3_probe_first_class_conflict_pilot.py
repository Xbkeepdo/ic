"""Queue a prospective sign-consistency gate while independent500 finishes."""

import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_probe_first_class_pilot as base
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-first-class-conflict-pilot')
SOURCE = base.OUT
PREVIOUS = base.NAME
NAME = 'sadi_probe_first_class_conflict_19'


def prepare():
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', PREVIOUS), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for filename in ('features/qwen3_probe_first_class_conflict.py',
                     'scripts/run_qwen3_probe_first_class_conflict_pilot.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    path = OUT / 'object_lexicon.json'
    plan['source_sha256'][str(path)] = shared.sha(path)
    previous = shared.read(SOURCE / 'protocol.json')
    plan.update(stage='head-gradient-probe-first-class-conflict-pilot', source_trace=previous['source_trace'],
                probe=previous['probe'], actuation_preflight=previous['actuation_preflight'],
                selection='Prospective V<0 and G>0 gate, fixed before inspecting independent500 metrics.')
    validation = shared.OUT / 'generation_time_evidence_drift_qwen3_fresh_first_class500/run_status.json'
    plan['independent500_status_at_design'] = shared.read(validation)['status']
    plan['recipe'].update(exclude_prior_token_id=True,
        gate='Existing first-class p_real<.5 gate plus layer19 visual write contrast<0 and FFN target gain>0.',
        readout_reuse='Compute sign readouts once and reuse their selected rows in the unchanged attention action.',
        rationale='Avoid acting when the FFN is already reducing the target, including man-made in the old development captions.',
        thresholds='Natural signs only; no fitted threshold, training or method-strength change.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    from features import qwen3_probe_first_class as first
    from features.qwen3_probe_first_class_conflict import Qwen3BatchFirstClassConflict
    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchFirstClassConflict):
        base.worker(gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare):
            driver.run()
    else:
        worker(args.worker)
