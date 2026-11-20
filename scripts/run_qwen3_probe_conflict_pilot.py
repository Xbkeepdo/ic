"""Add a visual/FFN sign guard to the frozen recycling proposal."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-conflict-pilot')
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-recycle-pilot-v2')
TRACE = shared.OUT / 'qwen3_sadi_pre_attention_i_el_trace'
PREVIOUS = 'sadi_probe_spatial_recycle_31'
NAME = 'sadi_probe_conflict_recycle_31'


def prepare():
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', PREVIOUS), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    previous = shared.read(SOURCE / 'protocol.json')
    for filename in ('features/qwen3_probe_conflict_guard.py', 'scripts/run_qwen3_probe_conflict_pilot.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    checked = SOURCE / 'recycle_preflight.json'
    plan['source_sha256'][str(checked)] = shared.sha(checked)
    plan.update(stage='head-gradient-probe-conflict-pilot', source_trace=str(TRACE), probe=previous['probe'],
                actuation_preflight=str(checked),
                selection='Single prospective guard change: require V<0 and FFN gain>0 in addition to original probe risk.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gate='First training token, original probe p_real<0.5, visual write contrast<0 and FFN target gain>0.',
        guard='Natural signs, not a fitted threshold; remaining model, budget and attention destination are unchanged.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    import models
    from features import qwen3_sar_ems_head_gradient as head
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_conflict_guard import Qwen3BatchProbeConflictGuard
    build = models.build_model
    plan = shared.read(OUT / 'protocol.json')
    trace = shared.read(TRACE / 'protocol.json')
    allowed = frozenset(shared.read(OUT / 'training_vocabulary.json')['token_ids'])
    probe, index = None, 0

    def checked_build(*args, **kwargs):
        nonlocal probe
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
        return wrapper

    def factory(model, span, nouns, *, final_gradient):
        nonlocal index
        assert final_gradient
        with np.load(TRACE / f'semantics/shard{gpu}.batch{index}.npz', allow_pickle=False) as saved:
            cache = {key: saved[key] for key in saved.files}
        assert cache['image_ids'].tolist() == trace['groups'][gpu][index]['ids']
        assert cache['span'].tolist() == list(span)
        index += 1
        return Qwen3BatchProbeConflictGuard(model, span, allowed, probe=probe,
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
