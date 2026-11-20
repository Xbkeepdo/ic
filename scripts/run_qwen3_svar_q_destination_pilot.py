"""Queue one destination-only ablation after the frozen fourth500 generation."""

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
from scripts import run_qwen3_svar_first_class_pilot as original

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-svar-q-destination-pilot')
SOURCE = original.OUT
PREDECESSOR = shared.OUT / 'generation_time_evidence_drift_qwen3_fresh_svar_midpoint500'
CPU = shared.OUT / 'qwen3_svar_q_destination_cpu/cpu_verification.json'
NAME = 'svar_first_class_q_destination_19'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed'] and shared.read(CPU)['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', original.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for shard in (0, 1):
        path = SOURCE / f'coco100/svar.shard{shard}.jsonl'
        plan['source_sha256'][str(path)] = shared.sha(path)
        shutil.copyfile(path, OUT / f'coco100/{path.name}')
    plan['copied_controls'].append('svar')
    previous = shared.read(SOURCE / 'protocol.json')
    trace = Path(previous['source_trace'])
    for path in (CPU, SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json',
                 Path(previous['semantic_preflight']), PREDECESSOR / 'protocol.json',
                 *(REPO / name for name in ('features/qwen3_probe_svar_first_class.py',
                    'features/qwen3_svar_q_destination.py', 'scripts/verify_qwen3_svar_q_destination.py',
                    'scripts/verify_qwen3_svar_q_destination_gpu.py',
                    'scripts/run_qwen3_svar_q_destination_pilot.py', 'scripts/audit_qwen3_svar_q_destination.py'))):
        plan['source_sha256'][str(path)] = shared.sha(path)
    for shard, groups in enumerate(shared.read(trace / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = trace / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-svar-q-destination-pilot', max_batch=16,
        variants=[*plan['copied_controls'], NAME], cached_controls=['svar'],
        primary_baseline='svar', primary_mechanism_comparator=original.NAME,
        source_trace=str(trace), probe=previous['probe'], predecessor=str(PREDECESSOR),
        actuation_preflight=str(OUT / 'q_destination_preflight.json'),
        semantic_preflight=previous['semantic_preflight'], fixed_at=time.time(),
        selection='One prospective FFN-destination ablation, prepared without reading fourth500 metrics. '
                  'Keep its adaptive D-based amount; use only per-head target-negative visual writes for destination.',
        development_screen='For a new mitigation candidate: native-SVAR Cs/Ci/false mentions strictly lower, '
                           'F1 loss<=1.5pp, recall loss<=2pp, leave-largest Ci no worse, and one image-reviewed '
                           'genuine correction. Compare the original risk arm fully; no threshold retuning.',
        interpretation='This tests the destination factor only, not whether FFN is wholly unnecessary. '
                       'Per-head dose is matched conditional on the same state; free-generation totals may differ.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gradient_scope={NAME: previous['recipe']['gradient_scope'][original.NAME]},
        destination='Normalize target-negative attention writes per head; remove the Q-P spatial deficit factor only.',
        preserved='Original D-based amount, donors, gate, gradient, actual-probability acceptance, tail replay.',
        prediction='Record a nonpositive first-order bound and label the parent numeric prediction as reference.',
        extra_cost='Read an already computed per-head negative distribution; no extra gradient/model/MLP/LM-head forwards.',
        saved='Scalar checks only; no full attention or hidden-state tensors are written.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import models
    from features import qwen3_probe_first_class as first
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_svar_q_destination import Qwen3BatchSvarQDestination
    from scripts.verify_qwen3_svar_q_destination_gpu import verify
    build = models.build_model

    def checked_build(*args, **kwargs):
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        if gpu == 0:
            plan = shared.read(OUT / 'protocol.json')
            probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
            allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
            shared.write(OUT / 'q_destination_preflight.json', verify(wrapper, probe, allowed))
        return wrapper

    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchSvarQDestination), \
            patch.object(models, 'build_model', checked_build):
        base.worker(gpu)


def run():
    plan = prepare()
    try:
        shared.write(OUT / 'run_status.json', dict(status='waiting_for_predecessor_generation', pid=os.getpid(),
                     predecessor=str(PREDECESSOR), queued=time.time()))
        while True:
            state = shared.read(PREDECESSOR / 'run_status.json')
            assert state['status'] != 'failed', 'Inspect the failed predecessor before more GPU work.'
            if state['status'] in ('evaluating', 'complete'):
                break
            pid = state['pid']
            raw = Path(f'/proc/{pid}/stat').read_text()
            assert raw[raw.rfind(')') + 2:].split()[0] != 'Z'
            time.sleep(5)
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
