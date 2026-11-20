"""One gate ablation on the existing100 images; reuse controls and image semantics."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_probe_first_class_pilot as base
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-ffn-promotion-pilot')
SOURCE = base.OUT
PREDECESSOR = shared.OUT / 'generation_time_evidence_drift_qwen3_fresh_svar_first_class500'
NAME = 'sadi_first_class_ffn_promotion_19'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', base.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    previous = shared.read(SOURCE / 'protocol.json')
    for filename in ('features/qwen3_probe_ffn_promotion.py', 'scripts/audit_qwen3_ffn_promotion.py',
                     'scripts/run_qwen3_ffn_promotion_pilot.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    for path in (OUT / 'object_lexicon.json', SOURCE / 'object_lexicon.json'):
        plan['source_sha256'][str(path)] = shared.sha(path)
    trace = Path(previous['source_trace'])
    for shard, groups in enumerate(shared.read(trace / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = trace / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-ffn-promotion-pilot', source_trace=str(trace),
        probe=previous['probe'], actuation_preflight=previous['actuation_preflight'],
        predecessor=str(PREDECESSOR), fixed_at=time.time(), max_batch=16,
        selection='Development ablation after inspecting earlier100 and two500 runs. Require local FFN target gain>0, without the old V<0 restriction. No held-out claim for this100.',
        development_screen='Relative to native SADI: Cs, Ci and false mentions strictly decrease; F1 loss<=1.5pp and recall loss<=2pp; Ci does not worsen after jointly removing the largest false-mention reduction image. Also require at least one image-reviewed genuine unsupported-object correction, beyond lexical/category artifacts. Otherwise no500 promotion.',
        numerical_preflight='Inherited exact early-replay disabled/zero/reject logits and current KV; action and rollback unchanged. New gate checked for every saved event by the independent promotion auditor.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gate='First lexical token and first prefix object class, p_real<.5, and local layer19 FFN target logit gain>0.',
        visual_support_gate='No requirement on the sign of the aggregated local visual logit contrast.',
        readout_reuse='Read local contrasts once for risk-positive rows; reuse selected readouts in the existing action.',
        unchanged='Keep SADI5--18, source/action19, final-gradient direction, FFN spatial drift, strength, probability-decrease acceptance and tail20--35 replay.',
        thresholds='Natural zero sign; no fitted threshold, new training or per-image exception.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    from features import qwen3_probe_first_class as first
    from features.qwen3_probe_ffn_promotion import Qwen3BatchFFNPromotion
    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchFFNPromotion):
        base.worker(gpu)


def run():
    plan = prepare()
    try:
        shared.write(OUT / 'run_status.json', dict(status='waiting_for_predecessor', pid=os.getpid(),
                     predecessor=str(PREDECESSOR), queued=time.time()))
        while True:
            previous = shared.read(PREDECESSOR / 'run_status.json')
            if previous['status'] == 'complete':
                break
            assert previous['status'] != 'failed', 'Inspect failed predecessor before starting another GPU job.'
            os.kill(previous['pid'], 0)
            time.sleep(10)
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', lambda: plan):
            driver.run()
    except BaseException as exc:
        if not (OUT / 'run_status.json').exists() or shared.read(OUT / 'run_status.json')['status'].startswith('waiting'):
            shared.write(OUT / 'run_status.json', dict(status='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
