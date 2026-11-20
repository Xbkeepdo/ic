"""Queue one fixed-evidence acceptance ablation, reusing all caption controls."""

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

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-svar-ffn-alignment-pilot')
SOURCE = original.OUT
PREDECESSOR = shared.OUT / 'generation_time_evidence_drift_qwen3_svar_hash_cached500'
NAME = 'svar_first_class_ffn_alignment_19'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', original.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for shard in (0, 1):
        path = SOURCE / 'coco100' / f'svar.shard{shard}.jsonl'
        plan['source_sha256'][str(path)] = shared.sha(path)
        shutil.copyfile(path, OUT / 'coco100' / path.name)
    plan['copied_controls'].append('svar')
    plan['cached_controls'] = ['svar']
    previous = shared.read(SOURCE / 'protocol.json')
    for filename in ('features/qwen3_probe_svar_first_class.py', 'features/qwen3_svar_ffn_alignment.py',
                     'scripts/run_qwen3_svar_ffn_alignment_pilot.py', 'scripts/verify_qwen3_svar_ffn_alignment.py',
                     'scripts/audit_qwen3_svar_ffn_alignment.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    for path in (SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json', Path(previous['semantic_preflight'])):
        plan['source_sha256'][str(path)] = shared.sha(path)
    trace = Path(previous['source_trace'])
    for shard, groups in enumerate(shared.read(trace / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = trace / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-svar-ffn-alignment-pilot', max_batch=16,
        variants=[*plan['copied_controls'], NAME], primary_baseline='svar',
        source_trace=str(trace), probe=previous['probe'], predecessor=str(PREDECESSOR),
        actuation_preflight=str(OUT / 'ffn_alignment_preflight.json'),
        semantic_preflight=previous['semantic_preflight'], fixed_at=time.time(),
        selection='Prospective fixed-evidence acceptance ablation on the old development100. '
                  'Use the unchanged attention proposal, additionally requiring its source FFN preference '
                  'to approach the pre-proposal negative-write evidence. Do not inspect intermediate500 controls.',
        development_screen='Relative to native SVAR: CHAIRs/Ci/false mentions strictly decrease, F1 loss<=1.5pp, '
                           'recall loss<=2pp, leave-largest-reduction Ci no worse; also require an image-reviewed '
                           'genuine unsupported-object correction. Report the original risk arm even if stronger. '
                           'Otherwise no500 promotion; no post-hoc threshold change.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gradient_scope={NAME: previous['recipe']['gradient_scope'][original.NAME]},
        acceptance='Actual original-target log probability decreases AND D_after<D_before; '
                   'otherwise restore hidden and current KV20--35.',
        fixed_evidence='Reuse exactly Q_negative computed once by the unchanged proposal. '
                       'P_after uses the actual source19 FFN output from that proposal and the same visual hpre.',
        drift='D=1-sum sqrt(Q_negative_before * softmax(cos(visual_hpre, FFN_update)/.03)).',
        extra_cost='One post-proposal spatial-preference dot product. Capture Q from its existing normalization '
                   'and the FFN update from its existing call; no additional model/MLP/LM-head forwards.',
        saved_vectors='Only Q_before and P_after for actual proposals, for independent drift recomputation. '
                      'No full attention matrices or hidden states are written.',
        unchanged='SVAR5--18, first-token/class risk gate, target gradient, layer19 attention action, '
                  'FFN drift budget/strength, replay and image-only semantic caches.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import models
    from features import qwen3_probe_first_class as first
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_svar_ffn_alignment import Qwen3BatchSvarFFNAlignment
    from scripts.verify_qwen3_svar_ffn_alignment import verify
    build = models.build_model

    def checked_build(*args, **kwargs):
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        if gpu == 0:
            plan = shared.read(OUT / 'protocol.json')
            probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
            allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
            shared.write(OUT / 'ffn_alignment_preflight.json', verify(wrapper, probe, allowed))
        return wrapper

    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchSvarFFNAlignment), \
            patch.object(models, 'build_model', checked_build):
        base.worker(gpu)


def run():
    plan = prepare()
    try:
        shared.write(OUT / 'run_status.json', dict(status='waiting_for_predecessor', pid=os.getpid(),
                     predecessor=str(PREDECESSOR), queued=time.time()))
        while True:
            state = shared.read(PREDECESSOR / 'run_status.json')
            if state['status'] == 'complete':
                break
            assert state['status'] != 'failed', 'Inspect the failed predecessor before starting more GPU work.'
            os.kill(state['pid'], 0)
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
