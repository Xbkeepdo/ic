"""Screen one selected-layer policy after complete SADI on cached100 batches."""

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

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-ffn-layer-selection-pilot')
SOURCE, TRACE = base.OUT, base.TRACE
CPU = shared.OUT / 'qwen3_ffn_layer_selection_cpu/cpu_verification.json'
PREFLIGHT = shared.OUT / 'qwen3_ffn_layer_selection_preflight/verification.json'
NAME = 'sadi_ffn_layer_select_19_31'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed'] and shared.read(CPU)['passed']
    preflight = shared.read(PREFLIGHT)
    assert preflight['passed'] and preflight['base_layers_inclusive'] == [5, 18]
    assert preflight['eligible_layers'] == [19, 23, 27, 31]
    for name, expected in preflight['source_sha256'].items():
        assert shared.sha(REPO / name) == expected, name
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', base.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for shard in (0, 1):
        path = svar.OUT / f'coco100/svar.shard{shard}.jsonl'
        plan['source_sha256'][str(path)] = shared.sha(path)
        shutil.copyfile(path, OUT / f'coco100/{path.name}')
    previous = shared.read(SOURCE / 'protocol.json')
    for path in (CPU, PREFLIGHT, PREFLIGHT.parent / 'status.json',
            OUT / 'object_lexicon.json', SOURCE / 'object_lexicon.json',
            *(REPO / name for name in ('features/qwen3_ffn_layer_selection.py',
                'scripts/verify_qwen3_ffn_layer_selection.py', 'scripts/verify_qwen3_ffn_layer_selection_gpu.py',
                'scripts/run_qwen3_ffn_layer_selection_pilot.py', 'scripts/audit_qwen3_ffn_layer_selection.py',
                'scripts/audit_qwen3_retained_prefix_state.py', 'scripts/audit_qwen3_svar_first_class.py',
                'scripts/audit_qwen3_caption_denominator.py', 'scripts/audit_qwen3_cached_caption_pilot.py',
                'scripts/audit_qwen3_head_feedback.py'))):
        plan['source_sha256'][str(path)] = shared.sha(path)
    for shard, groups in enumerate(shared.read(TRACE / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = TRACE / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-ffn-layer-selection-pilot', max_batch=16,
        variants=[*plan['copied_controls'], 'svar', NAME], source_trace=str(TRACE),
        probe=previous['probe'], primary_baseline='sadi', primary_mechanism_comparator=base.NAME,
        additional_copied_controls={'svar': str(svar.OUT)}, fixed_at=time.time(),
        actuation_preflight=str(PREFLIGHT), selection_mode='adaptive',
        planned_followup_controls=dict(fixed19='Same positive-FFN-push and recycle eligibility, force layer19.',
            fixed31='Same positive-FFN-push and recycle eligibility, force layer31.'),
        execution_order='Run adaptive100 first; if numerical and genuine-object checks support it, '
            'execute the already-defined fixed-layer controls before500. Report all controls that are run.',
        selection='Preserve complete SADI5--18. Select at most one of19/23/27/31 per image/token '
            'using positive final-target FFN attribution times negative-visual fraction times FFN drift.',
        development_screen='Against SADI: Cs/Ci/absolute false mentions lower; F1 loss<=1.5pp, '
            'recall loss<=2pp, leave-largest Ci no worse, and an image-confirmed unsupported-object correction. '
            'Also report old19 guidance and SVAR; no retuning on this100.',
        interpretation='Local first-order attribution is comparable across layers because all read the '
            'same final log probability. This is a whole selection policy, not a dose-matched causal test.')
    plan['recipe'].update(exclude_prior_token_id=True, base_variant='sadi', base_layers=[5, 18],
        intervention_layer='selected_per_image_and_step', eligible_layers=[19, 23, 27, 31],
        selection_score='relu(dlogp(y)/du_l dot u_l) * target_negative_visual_absolute_fraction_l * D_l',
        selection='Argmax of positive scores, smaller-layer tie break; no action if all scores zero.',
        gradient_scope={NAME: 'One existing19--35 tail backward collects attention-residual and FFN-output gradients at all4 layers.'},
        attention_action='Original rho=.5D signed history-to-negative-visual probability transfer at selected layer; original full-update cosine P.',
        replay='At most one attention change per image; same full batch GEMM shapes and one20--35 current-token replay, selected-row KV rollback on rejection.',
        unchanged='Original risk/class gate, complete SADI5--18, tau=.03, signed destinations and actual original-target probability-decrease acceptance.',
        extra_cost='No extra model or backward calls relative to the old19 proposal; collect4 source gradients in one backward and reuse their cached native states.',
        saved='Scalar4-layer selection diagnostics only; no full hidden or attention tensors written.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    from features import qwen3_probe_first_class as first
    from features.qwen3_ffn_layer_selection import Qwen3BatchFFNLayerSelection
    with patch.object(base, 'OUT', OUT), patch.object(base, 'NAME', NAME), \
            patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchFFNLayerSelection):
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
