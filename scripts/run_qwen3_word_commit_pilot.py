"""Queue unchanged-action word-commit validation after the hash controls."""

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

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-word-commit-pilot')
SOURCE = base.OUT
PREDECESSOR = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-matched-hash-pilot')
NAME = 'sadi_first_class_word_commit_19'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', base.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    previous = shared.read(SOURCE / 'protocol.json')
    for name in ('features/qwen3_probe_word_commit.py', 'scripts/verify_qwen3_word_commit.py',
                 'scripts/run_qwen3_word_commit_pilot.py', 'scripts/audit_qwen3_word_commit.py'):
        plan['source_sha256'][str(REPO / name)] = shared.sha(REPO / name)
    trace = Path(previous['source_trace'])
    for shard, groups in enumerate(shared.read(trace / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = trace / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan['source_sha256'][str(OUT / 'object_lexicon.json')] = shared.sha(OUT / 'object_lexicon.json')
    validation = shared.OUT / 'generation_time_evidence_drift_qwen3_replication_first_class500/run_status.json'
    plan.update(stage='head-gradient-word-commit-pilot', max_batch=16,
        source_trace=str(trace), probe=previous['probe'], predecessor=str(PREDECESSOR),
        actuation_preflight=str(OUT / 'word_commit_preflight.json'),
        fixed_at=time.time(), replication500_status_at_design=shared.read(validation)['status'],
        selection='Prospective acceptance ablation, developed from first500 behavior; evaluate old100 first. Do not alter the active independent replication.',
        development_screen='CHAIRs/Ci and false mentions no worse than original first-class old100; F1 loss<=0.5pp. A second500 follow-up remains an ablation, not a newly sampled independent cohort.')
    plan['recipe'].update(exclude_prior_token_id=True,
        acceptance='Actual original-target log probability decreases AND corrected greedy token differs; otherwise restore hidden and current KV20--35.',
        acceptance_readout='Observe the corrected LM-head call already used by the original acceptance test; no additional LM-head forward.',
        unchanged='Detector, vocabulary, prefix-class gate, gradients, attention direction, FFN drift and strength unchanged.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import models
    from features import qwen3_probe_first_class as first
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_word_commit import Qwen3BatchWordChangingCommit
    from scripts.verify_qwen3_word_commit import verify
    build = models.build_model

    def checked_build(*args, **kwargs):
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        if gpu == 0:
            plan = shared.read(OUT / 'protocol.json')
            probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
            allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
            shared.write(OUT / 'word_commit_preflight.json', verify(wrapper, probe, allowed))
        return wrapper

    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchWordChangingCommit), \
            patch.object(models, 'build_model', checked_build):
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
            assert previous['status'] != 'failed', 'Inspect failed hash controls before starting another job.'
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
    if args.worker is None:
        run()
    else:
        worker(args.worker)
