"""Test supported-rival margin recycling against the lexical probe control."""

import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-grounded-rival-pilot')
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-lexical-pilot')
TRACE = shared.OUT / 'qwen3_sadi_pre_attention_i_el_trace'
PREVIOUS = 'sadi_probe_lexical_recycle_31'
NAME = 'sadi_probe_grounded_rival_31'


def prepare():
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', PREVIOUS), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for filename in ('features/qwen3_probe_grounded_rival.py', 'scripts/run_qwen3_probe_grounded_rival_pilot.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    path = OUT / 'object_lexicon.json'
    plan['source_sha256'][str(path)] = shared.sha(path)
    previous = shared.read(SOURCE / 'protocol.json')
    plan.update(stage='head-gradient-probe-grounded-rival-pilot', source_trace=str(TRACE), probe=previous['probe'],
                selection='Prospective supported-rival margin objective; same lexical trigger and frozen probe.')
    plan['recipe'].update(exclude_prior_token_id=True,
        objective='Final native logit(target) minus logit(rival). One gradient through source31 FFN and tail32--35.',
        rival='Highest native-probability other lexical object in global top16 and logit gap<=2; p_real>=.5, V_rival>0 and V_rival>V_target.',
        guided_destination='Use negative margin contributions in the existing spatially guided recycling operation.',
        trials='Fractions .5*D, D, min(1,2*D); one shared gradient and source KV, at most three tail trials.',
        acceptance='Keep first proposal whose final argmax equals the chosen rival and margin strictly decreases; otherwise rollback.',
        diagnostic_force_rival=False, preflight='Exact disabled/zero/forced rejection logits,tokens,current tail KV; no prefix KV append.',
        scope='I/E_l detector, target-conditioned visual effects and FFN spatial deviation; not complete SAM/SAM-V2.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    import models
    from features import qwen3_sar_ems_head_gradient as head
    from features import qwen3_probe_attention_recycle as recycle_module
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_grounded_rival import Qwen3BatchProbeGroundedRival
    from scripts.run_qwen3_probe_recycle_pilot import preflight
    build = models.build_model
    plan = shared.read(OUT / 'protocol.json')
    trace = shared.read(TRACE / 'protocol.json')
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    probe, index = None, 0

    def checked_build(*args, **kwargs):
        nonlocal probe
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
        if gpu == 0:
            cached = None

            def diagnostic(*args, **options):
                nonlocal cached
                options['diagnostic_force_probe_gate'] = False
                controller = Qwen3BatchProbeGroundedRival(*args, diagnostic_force_rival=True,
                    cached_semantics=cached, **options)
                original = controller.statistics

                def collect(responses):
                    nonlocal cached
                    if len(controller.reader.semantics) == 36:
                        cached = torch.stack([controller.reader.semantics[i] for i in range(36)]).cpu()
                    return original(responses)

                controller.statistics = collect
                return controller

            with patch.object(recycle_module, 'Qwen3BatchProbeAttentionRecycle', diagnostic):
                report = preflight(wrapper, probe, allowed)
            report.update(controller='Qwen3BatchProbeGroundedRival', diagnostic_force_rival=True,
                          production_force_rival=False, semantic_cache_reused=True)
            shared.write(OUT / 'rival_preflight.json', report)
        return wrapper

    def factory(model, span, nouns, *, final_gradient):
        nonlocal index
        assert final_gradient
        with np.load(TRACE / f'semantics/shard{gpu}.batch{index}.npz', allow_pickle=False) as saved:
            cache = {key: saved[key] for key in saved.files}
        assert cache['image_ids'].tolist() == trace['groups'][gpu][index]['ids']
        assert cache['span'].tolist() == list(span)
        index += 1
        return Qwen3BatchProbeGroundedRival(model, span, allowed, probe=probe,
            semantic_tokens=cache['token_ids'].tolist(), cached_semantics=torch.from_numpy(cache['E_l']), guided=True)

    with patch.object(shared, 'HEAD_FINAL', NAME), patch.object(head, 'Qwen3BatchHeadGradient', factory), \
            patch.object(models, 'build_model', checked_build):
        shared.worker(OUT, gpu)
    assert index == len(trace['groups'][gpu])


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
