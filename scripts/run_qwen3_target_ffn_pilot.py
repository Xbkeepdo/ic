"""Preserve complete SADI and condition only the later FFN spatial readout."""

import argparse
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
from scripts import run_qwen3_svar_first_class_pilot as svar

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-target-ffn-pilot')
SOURCE, TRACE = base.OUT, base.TRACE
CPU = shared.OUT / 'qwen3_target_ffn_cpu/cpu_verification.json'
NAME = 'sadi_probe_target_ffn_19'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    assert shared.read(CPU)['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', base.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for shard in (0, 1):
        path = svar.OUT / f'coco100/svar.shard{shard}.jsonl'
        plan['source_sha256'][str(path)] = shared.sha(path)
        shutil.copyfile(path, OUT / f'coco100/{path.name}')
    previous = shared.read(SOURCE / 'protocol.json')
    for path in (CPU, OUT / 'object_lexicon.json', SOURCE / 'object_lexicon.json',
            *(REPO / name for name in ('features/qwen3_target_ffn.py', 'scripts/verify_qwen3_target_ffn.py',
                'scripts/verify_qwen3_target_ffn_gpu.py', 'scripts/run_qwen3_target_ffn_pilot.py',
                'scripts/audit_qwen3_target_ffn.py', 'scripts/audit_qwen3_retained_prefix_state.py',
                'scripts/audit_qwen3_svar_first_class.py',
                'scripts/audit_qwen3_caption_denominator.py', 'scripts/audit_qwen3_cached_caption_pilot.py',
                'scripts/audit_qwen3_head_feedback.py'))):
        plan['source_sha256'][str(path)] = shared.sha(path)
    for shard, groups in enumerate(shared.read(TRACE / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = TRACE / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-target-ffn-pilot', max_batch=16,
        variants=[*plan['copied_controls'], 'svar', NAME], source_trace=str(TRACE),
        probe=previous['probe'], primary_baseline='sadi', primary_mechanism_comparator=base.NAME,
        additional_copied_controls={'svar': str(svar.OUT)}, fixed_at=time.time(),
        actuation_preflight=str(OUT / 'target_ffn_preflight.json'),
        selection='Preserve full SADI5--18 and the existing layer19 attention recovery. '
                  'Replace only the FFN spatial direction with its locally target-promoting component.',
        development_screen='Against SADI: Cs/Ci/absolute false mentions strictly lower; F1 loss<=1.5pp, '
            'recall loss<=2pp, leave-largest Ci no worse, and image-confirmed correction. '
            'Also compare full old guidance and matched SVAR. No threshold or layer tuning on results.',
        interpretation='The gradient coordinate mask is a local derivative attribution, not a calibrated '
            'truth label or an edit of the FFN. Old100 is development data, not independent validation.')
    plan['recipe'].update(exclude_prior_token_id=True, base_variant='sadi', base_layers=[5, 18],
        intervention_layer=19, replayed_tail_layers=list(range(20, 36)),
        target_ffn='u_plus[d]=u[d] if u[d]*dlogp_final(y)/du[d]>0, else0; '
            'P_y=softmax(cos(hpre_visual,u_plus)/.03). Empty component gives uniform P.',
        unchanged='Risk/class gate, full SADI5--18, target-residual gradient Q, rho=.5D, '
            'signed history-to-visual action, actual probability acceptance, KV rollback.',
        extra_cost='Collect residual and FFN-output derivatives in ONE existing autograd.grad call. '
            'No extra model/MLP/LM-head calls; extra vector attribution and spatial diagnostics only.',
        saved='Scalar diagnostics, no full attention matrices or image hidden-state arrays.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import models
    from features import qwen3_probe_first_class as first
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_target_ffn import Qwen3BatchTargetFFN
    from scripts.verify_qwen3_target_ffn_gpu import verify
    build = models.build_model

    def checked_build(*args, **kwargs):
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        if gpu == 0:
            plan = shared.read(OUT / 'protocol.json')
            probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
            allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
            shared.write(OUT / 'target_ffn_preflight.json', verify(wrapper, probe, allowed))
        return wrapper

    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchTargetFFN), \
            patch.object(models, 'build_model', checked_build):
        base.worker(gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    if args.worker is None:
        prepare_run = prepare
        if args.prepared:
            plan = shared.read(OUT / 'protocol.json')
            check = shared.read(OUT / 'preparation_check.json')
            assert check['passed'] and check['protocol_sha256'] == shared.sha(OUT / 'protocol.json')
            for filename, expected in plan['source_sha256'].items():
                assert shared.sha(Path(filename)) == expected, filename
            prepare_run = lambda: plan
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare_run):
            driver.run()
    else:
        worker(args.worker)
