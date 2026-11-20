"""Keep full SADI and compare adjacent evidence at matched visual mass."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_bounded_visual_tilt_pilot as previous
from scripts import run_qwen3_causal_visual_tilt_pilot as causal
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE = shared.OUT/'qwen3_causal_visual_tilt_pilot_20261119'
OUT = shared.OUT/'qwen3_mass_matched_visual_pilot_20261119'
PREFLIGHT = shared.OUT/'qwen3_mass_matched_visual_preflight_20261119'
RESERVATION = shared.OUT/'qwen3_mass_matched_visual_reserved500_20261119'
MODES = {'mass_native': ('evidence', False), 'mass_refresh': ('evidence', True),
         'mass_roll': ('evidence', True)}
PERMUTATIONS = {'mass_native': 'none', 'mass_refresh': 'none', 'mass_roll': 'half_roll'}
CONTROLS = ['sadi', 'svar', 'causal_fixed']
FILES = ['scripts/run_qwen3_mass_matched_visual_pilot.py',
         'scripts/audit_qwen3_mass_matched_visual_pilot.py',
         'scripts/analyze_qwen3_mass_matched_visual_pilot.py']
AUDITS = [FILES[1], 'scripts/audit_qwen3_caption_with_native_cache_v2.py', FILES[2]]


def prepare():
    assert not OUT.exists(), 'Inspect existing workers and outputs before any resume'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    proof = shared.read(PREFLIGHT/'verification.json')
    request = shared.read(PREFLIGHT/'request.json')
    assert proof['passed'] and proof['request_sha256'] == shared.sha(PREFLIGHT/'request.json')
    assert proof['all_frozen_inputs_verified'] == len(request['source_sha256'])
    assert set(proof['batches']) == {'B2', 'B16'}
    reserve = shared.read(RESERVATION/'reservation.json')
    assert reserve['before_mass_matched_pilot_metrics'] and not reserve['neural_generation_started']
    assert reserve['preflight_verification_sha256'] == shared.sha(PREFLIGHT/'verification.json')
    assert not (RESERVATION/'consumed.json').exists()
    old = shared.read(SOURCE/'protocol.json')
    hashes = dict(request['source_sha256'])
    hashes.update(old['source_sha256'])
    hashes.update(shared.read(SOURCE/'analysis_protocol.json')['source_sha256'])
    certificate = shared.read(previous.CERTIFICATE)
    assert certificate['passed'] and certificate['joint17_agreement_verified']
    hashes.update(certificate['source_sha256'])
    paths = [PREFLIGHT/n for n in ('request.json', 'verification.json')]
    paths += [RESERVATION/n for n in ('cohort.json', 'selection.json', 'reservation.json')]
    paths += [previous.CERTIFICATE, SOURCE/'protocol.json', SOURCE/'final_provenance.json']
    paths += [REPO/n for n in FILES + ['scripts/run_qwen3_bounded_visual_tilt_pilot.py',
        'scripts/run_qwen3_history_route_pilot.py', 'scripts/run_qwen3_causal_visual_tilt_pilot.py',
        'scripts/analyze_qwen3_scalar_visual_pilot.py', 'scripts/qwen3_caption_groups_with_cache.py',
        'scripts/audit_qwen3_caption_with_native_cache_v2.py']]
    for path in paths:
        hashes[str(path)] = shared.sha(path)
    for name, value in hashes.items():
        assert shared.sha(REPO/name) == value, name
    OUT.mkdir(); (OUT/'logs').mkdir(); (OUT/'coco100').mkdir()
    for name in ('cohort.json', 'object_lexicon.json'):
        shutil.copyfile(SOURCE/name, OUT/name)
    assert not set(shared.read(OUT/'cohort.json')['coco100']) & set(
        shared.read(RESERVATION/'cohort.json')['coco500'])
    for shard in (0, 1):
        for name in CONTROLS:
            path = SOURCE/f'coco100/{name}.shard{shard}.jsonl'
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT/'coco100'/path.name)
    plan = dict(stage='mass-matched-visual-development', images=100, shards=2, cohort='coco100',
        max_batch=16, max_new_tokens=512, variants=[*CONTROLS, *MODES],
        candidates=list(MODES), copied_controls=CONTROLS, cached_controls=['sadi'],
        base_by_candidate={name: 'sadi' for name in MODES},
        mode_by_candidate={name: mode for name, (mode, _) in MODES.items()},
        refresh_by_candidate={name: refresh for name, (_, refresh) in MODES.items()},
        permutation_by_candidate=PERMUTATIONS,
        source=str(SOURCE), source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),
        source_sha256=hashes, fixed_at=time.time(), probe=old['probe'], real_threshold=old['real_threshold'],
        cache_certificate=str(previous.CERTIFICATE), cache_certificate_sha256=shared.sha(previous.CERTIFICATE),
        eos_token_ids=old['eos_token_ids'],
        recipe=dict(base_layers=[5, 18], unchanged_anchor=19,
            budget_source_layers=list(range(19, 35)), intervention_layers=list(range(20, 36)), gamma=1.,
            gate='Unchanged original108 SAM-V2 threshold and first-token/class scope.',
            source_contrast='u=D*Q/(Q+P); r=2*u-D; D=HellingerSquared(Q,P).',
            coefficient='c_hj=sigmoid(r_j+b_h), solve b_h in 32 bisections.',
            matching='Per-head FP32 visual partition equals local .5*abs(current replay logit) reference; nonvisual logits unchanged. BF16 rounding measured separately.',
            shift='abs(current replay visual logit)*c_hj; c in [0,1]; gamma=1.',
            signals='Q=Norm(write_norm*(1+log_s)); P=softmax(cos(uFFN,hpre_visual)/.03).',
            native='Native source k-1; actual replay query at k.',
            refresh='Actual modified attention and FFN source k-1; one final target retained.',
            roll='Same refreshed construction with contrast rolled by half the visual span.',
            acceptance='Actual original-target final logp decreases; otherwise restore hidden/current20--35 KV; history untouched.',
            cost='No backward or extra encoder; shared image hpre/GQA/semantic RAM caches; solver/reference-softmax overhead measured.',
            cache='Original22 groups retained. Seven certified zero-positive-gate batches: nine images per candidate,91 live.',
            precision='Matching is local at the actual query; it does not equate entire trajectories across versions.',
            eos='Actual emitted token IDs determine EOS; legacy terminal_eos preserved.'),
        development_screen='Lower sentence Cs/Ci and false count than SADI; F1 loss<=1.5pp, recall loss<=2pp, leave-largest Ci no worse. Compare native/refresh against fixed and permuted controls, and SVAR. Review original images before interpreting physical corrections.',
        followup='Exploratory reused100. New reserved500 remains untouched unless a useful new candidate is identified; no redraw or validation tuning.')
    assert len('chair_per_image.first100.'+'-'.join(sorted(plan['variants']))+'.jsonl') <= 255
    shared.write(OUT/'protocol.json', plan)
    analysis_files = [*AUDITS, 'scripts/analyze_qwen3_scalar_visual_pilot.py',
        'scripts/analyze_qwen3_causal_visual_tilt_pilot.py',
        'scripts/audit_qwen3_visual_feedback_mentions.py', 'scripts/audit_qwen3_fresh_caption500.py',
        'scripts/qwen3_caption_groups_with_cache.py']
    shared.write(OUT/'analysis_protocol.json', dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'), before_inspecting_pilot_metrics=True,
        source_sha256={str(REPO/name): shared.sha(REPO/name) for name in analysis_files},
        primary='Full100 sentence and legacy, counts/recall/F1, paired intervals; all six arms retained.',
        cases='All hallucinated-image fixes/introductions and top3 nonzero mention/recall changes; ties by ID; inspect original photos.',
        scope='Old development100, no independent validation claim. Shared uncapped subset is secondary.'))
    return plan


def decorate_statistics(stats, refresh, eos, permutation):
    for stat in stats:
        if stat.get('event_log_from_certified_cache'):
            causal.decorate_statistics([stat], refresh, eos)
            stat.update(mass_matched_visual=True, contrast_permutation=permutation,
                diagnostic_flat_contrast=False, mass_matching_solver_iterations=32,
                mass_matching_reference_coefficient=.5,
                mass_matching_precision='FP32 per-head partition; BF16 rounding measured separately.',
                maximum_fp32_visual_mass_difference=0., maximum_bf16_visual_mass_difference=0.,
                transport_mode='adjacent_mass_matched_visual_logit_tilt',
                acg_scope='Certified no-positive-gate generation reused; no enhancement executed.')
        for event in stat['full_joint_events']:
            event['emitted_is_eos'] = event['emitted_token'] in eos
        stat['actual_eos_accepts'] = sum(e['accepted'] and e['emitted_is_eos'] for e in stat['full_joint_events'])


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_mass_matched_feedback import Qwen3BatchMassMatchedFeedback
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    plan = shared.read(OUT/'protocol.json')
    for name, value in plan['source_sha256'].items():
        assert shared.sha(REPO/name) == value, name
    groups = source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    cached = source_groups(previous.ORIGINAL/f'coco100/sadi_sam_acg_guided_19_35.shard{gpu}.jsonl')
    assert shared.group_keys(cached) == shared.group_keys(groups)
    eligible = {e['group']: e for e in shared.read(previous.CERTIFICATE)['eligible_groups'] if e['shard'] == gpu}
    paths = {name: OUT/f'coco100/{name}.shard{gpu}.jsonl' for name in MODES}
    assert not any(path.exists() for path in paths.values())
    wrapper = build_wrapper(); probe = FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed = frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    eos = plan['eos_token_ids']; eos = set(eos if isinstance(eos, list) else [eos])
    verified = False; memory = []
    with torch.inference_mode():
        for index, group in enumerate(groups):
            ids = [r['image_id'] for r in group]; static = {}
            if index not in eligible:
                (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
                assert [image for image, _ in inputs] == ids
                assert span[1]-span[0] == group[0]['visual_token_count']
                if not verified:
                    tokens, _, _ = common.generate_group(wrapper, span, inputs, allowed, 'sadi', 32)
                    assert tokens == [r['response_token_ids'][:32] for r in group]
                    shared.write(OUT/f'verification.gpu{gpu}.json', dict(passed=True, ids=ids, native32_exact=True))
                    gc.collect(); torch.cuda.empty_cache(); verified = True
            for name, (mode, refresh) in MODES.items():
                permutation = plan['permutation_by_candidate'][name]
                if index in eligible:
                    assert eligible[index]['image_ids'] == ids
                    rows = previous.cache_rows(cached[index], plan, gpu, index, name, mode)
                    decorate_statistics([row['sar_ems_statistics'] for row in rows], refresh, eos, permutation)
                else:
                    holder = {}
                    def factory(model, current_span, nouns, mode, strength):
                        controller = Qwen3BatchMassMatchedFeedback(model, current_span, allowed, probe=probe,
                            semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer, history_parser=parser,
                            shared_static=static, real_threshold=plan['real_threshold'],
                            tilt_mode='evidence', refresh=refresh, gamma=1., permutation=permutation)
                        holder['controller'] = controller
                        return controller
                    torch.cuda.reset_peak_memory_stats(wrapper.device)
                    with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                        tokens, stats, seconds = original.generate_group(wrapper, span, inputs, allowed, shared.POSITIVE, 512)
                    calls = holder['controller'].steps; holder.clear()
                    decorate_statistics(stats, refresh, eos, permutation)
                    texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    rows = [dict(image_id=image, model=MODEL, variant=name, response_token_ids=t,
                        generated_text=text.strip(), seconds=seconds/len(ids), batch_wall_seconds=seconds,
                        batch_size=len(ids), visual_token_count=span[1]-span[0], sar_ems_statistics=stat,
                        decoder='native_qwen3_generate_same_span_batch', generation_reused=False,
                        actual_model_calls_in_batch=calls)
                        for image, t, text, stat in zip(ids, tokens, texts, stats, strict=True)]
                with paths[name].open('a', encoding='utf-8') as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                gc.collect(); torch.cuda.empty_cache()
                memory.append(dict(group=index, variant=name, allocated=torch.cuda.memory_allocated(wrapper.device),
                    peak=torch.cuda.max_memory_allocated(wrapper.device), reused=rows[0]['generation_reused']))
                shared.write(OUT/f'memory.gpu{gpu}.json', memory)
                print(f'[{gpu}] {name} group={index} B={len(ids)} seconds={rows[0]["batch_wall_seconds"]:.3f}', flush=True)
            static.clear(); gc.collect(); torch.cuda.empty_cache()
    assert all(shared.group_keys(source_groups(path)) == shared.group_keys(groups) for path in paths.values())


def run():
    with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), patch.object(driver, 'prepare', prepare):
        driver.run()
    import os
    commands = [[name, str(OUT)] for name in AUDITS]
    shared.write(OUT/'post_audit_status.json', dict(status='running', commands=commands, started=time.time()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
    try:
        for index, args in enumerate(commands):
            with (OUT/f'post_audit{index}.log').open('x') as handle:
                subprocess.run([sys.executable, *args], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        assert all(shared.read(OUT/name)['passed'] for name in
                   ('mechanism_audit.json', 'independent_summary.json', 'analysis_provenance.json'))
        shared.write(OUT/'post_audit_status.json', dict(status='passed', commands=commands, finished=time.time()))
    except BaseException as exc:
        shared.write(OUT/'post_audit_status.json', dict(status='failed', error=repr(exc), commands=commands, failed=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
