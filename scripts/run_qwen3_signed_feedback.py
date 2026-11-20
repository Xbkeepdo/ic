"""Evaluate signed SAR/EMS next-layer feedback on exact cached SADI batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / 'outputs'
PREFIX = 'generation_time_evidence_drift_qwen3_'
BASE = 'sadi'
POSITIVE = 'sadi_sar_ems_next_19_20'
SIGNED = 'sadi_sar_ems_signed_19_20'
DIRECTIONAL = 'sadi_sar_ems_directional_19_20'
LATE_DIRECTIONAL = 'sadi_sar_ems_directional_25_26'
CONSENSUS = 'sadi_sar_ems_consensus_26'
ALTERNATIVE = 'sadi_sar_ems_alternative_25_26'
SVAR_ALTERNATIVE = 'svar_sar_ems_alternative_25_26'
MARGIN = 'sadi_sar_ems_margin_25_26'
FFN_AUX = 'sadi_sar_ems_ffn_aux_25_26'
LOW_UNIFORM = 'sadi_low_visual_uniform_29'
LOW_SPATIAL = 'sadi_low_visual_spatial_29'
REMOVAL_GATE = 'sadi_visual_removal_gate_32'
VERIFIED_RIVAL = 'sadi_visual_verified_rival_32'
NEGATIVE_WRITE = 'sadi_negative_write_32'
REMOVAL_SAME31 = 'sadi_visual_removal_same_31'
REMOVAL_SAME35 = 'sadi_visual_removal_same_35'
FINAL_TARGET = 'sadi_final_target_negative_31'
HEAD_FINAL = 'sadi_head_final_gradient_31'
HEAD_SOURCE = 'sadi_head_source_gradient_31'
CONTROLS = (BASE, POSITIVE)
CANDIDATES = (SIGNED, DIRECTIONAL)
RUNS = {
    'pilot': ('layer_feedback100_b16', 'coco100', 512),
    '500': ('next500_t256_b16_combined', 'coco500', 256),
    'svar-pilot': ('svar100', 'coco100', 512),
    'svar-500': ('svar_next500_t256_b32', 'coco500', 256),
    'svar-mass-pilot': ('signed_feedback_svar-pilot', 'coco100', 512),
    'svar-mass-500': ('svar_next500_t256_b32', 'coco500', 256),
    'late-pilot': ('signed_feedback_pilot', 'coco100', 512),
    'late-500': ('signed_feedback_500', 'coco500', 256),
    'consensus-pilot': ('signed_feedback_late-pilot', 'coco100', 512),
    'consensus-500': ('signed_feedback_500', 'coco500', 256),
    'alternative-pilot': ('signed_feedback_late-pilot', 'coco100', 512),
    'alternative-500': ('signed_feedback_500', 'coco500', 256),
    'reference-500': ('signed_feedback_alternative-500', 'coco500', 256),
    'svar-alternative-pilot': ('signed_feedback_svar-mass-pilot', 'coco100', 512),
    'svar-alternative-500': ('svar_next500_t256_b32', 'coco500', 256),
    'margin-pilot': ('signed_feedback_alternative-pilot', 'coco100', 512),
    'margin-500': ('fresh_alternative500', 'coco500', 256),
    'ffn-aux-pilot': ('signed_feedback_alternative-pilot', 'coco100', 512),
    'ffn-aux-500': ('fresh_alternative500', 'coco500', 256),
    'low-visual-pilot': ('signed_feedback_alternative-pilot', 'coco100', 512),
    'low-visual-500': ('next500_t256_b16_combined', 'coco500', 256),
    'removal-gate-pilot': ('signed_feedback_alternative-pilot', 'coco100', 512),
    'removal-gate-500': ('signed_feedback_reference-500', 'coco500', 256),
    'verified-rival-pilot': ('signed_feedback_removal-gate-pilot', 'coco100', 512),
    'verified-rival-500': ('signed_feedback_reference-500', 'coco500', 256),
    'negative-write-pilot': ('signed_feedback_removal-gate-pilot', 'coco100', 512),
    'negative-write-500': ('signed_feedback_reference-500', 'coco500', 256),
    'removal-same-pilot': ('signed_feedback_removal-gate-pilot', 'coco100', 512),
    'removal-same-500': ('signed_feedback_reference-500', 'coco500', 256),
    'final-target-pilot': ('signed_feedback_removal-gate-pilot', 'coco100', 512),
    'final-target-500': ('signed_feedback_reference-500', 'coco500', 256),
    'head-gradient-pilot': ('signed_feedback_final-target-pilot', 'coco100', 512),
    'head-gradient-500': ('signed_feedback_final-target-500', 'coco500', 256),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_sources():
    return (
        'features/qwen3_sar_ems_layer_feedback.py', 'features/qwen3_sar_ems_attention_batch.py',
        'features/llava_sar_ems_attention.py', 'features/llava_sadi.py', 'features/llava_decoding_evidence.py',
        'scripts/generation_time_qwen3_layer_feedback.py', 'scripts/generation_time_evidence_drift_qwen3_batched.py',
        'scripts/generation_time_sar_ems_seed43.py', 'models/qwen3_vl_wrapper.py',
        'configs/model_configs_coco4000_512_endac_baselines_v2.yaml',
    )


def source_groups(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    offset = 0
    groups = []
    while offset < len(rows):
        first = rows[offset]
        size = first['batch_size']
        group = rows[offset:offset + size]
        assert 1 <= size <= 32 and len(group) == size
        assert all(row['batch_size'] == size and row['batch_wall_seconds'] == first['batch_wall_seconds']
                   and row['visual_token_count'] == first['visual_token_count']
                   and row['decoder'] == 'native_qwen3_generate_same_span_batch' for row in group)
        groups.append(group)
        offset += size
    return groups


def group_keys(groups):
    return [(tuple(row['image_id'] for row in group), group[0]['visual_token_count']) for group in groups]


def prepare(stage, validate_directional_tradeoff=False):
    source_name, cohort, cap = RUNS[stage]
    source = OUT / (PREFIX + source_name)
    output = OUT / (PREFIX + f'signed_feedback_{stage}')
    svar = stage.startswith('svar-')
    visual_mass = stage.startswith('svar-mass-')
    late = stage.startswith('late-')
    consensus = stage.startswith('consensus-')
    alternative = stage.startswith('alternative-')
    margin = stage.startswith('margin-')
    ffn_aux = stage.startswith('ffn-aux-')
    low_visual = stage.startswith('low-visual-')
    removal_gate = stage.startswith('removal-gate-')
    verified_rival = stage.startswith('verified-rival-')
    negative_write = stage.startswith('negative-write-')
    removal_same = stage.startswith('removal-same-')
    final_target = stage.startswith('final-target-')
    head_gradient = stage.startswith('head-gradient-')
    svar_alternative = stage.startswith('svar-alternative-')
    reference = stage == 'reference-500'
    pilot_stage = stage.endswith('pilot')
    controls = CONTROLS
    if svar:
        controls = ('svar', 'svar_sar_ems_attention' if pilot_stage else 'svar_sar_ems_next_19_20')
    if visual_mass and pilot_stage:
        controls += ('svar_sar_ems_directional_19_20',)
    if svar_alternative and pilot_stage:
        controls += ('svar_sar_ems_directional_19_20', 'svar_sar_ems_visual_mass_19_20')
    if late or consensus or alternative:
        controls += (DIRECTIONAL,)
    if (consensus or alternative) and pilot_stage:
        controls += (LATE_DIRECTIONAL,)
    if reference:
        controls += (DIRECTIONAL, ALTERNATIVE)
    if margin or ffn_aux:
        controls += (ALTERNATIVE,)
        if not pilot_stage:
            controls += ('svar',)
    if (removal_gate or verified_rival or negative_write or removal_same or final_target or head_gradient) and not pilot_stage:
        controls += ('svar',)
    if head_gradient:
        controls += (FINAL_TARGET,)
    if (verified_rival or negative_write or removal_same) and pilot_stage:
        controls += (REMOVAL_GATE,)
    ids = read(source / 'cohort.json')[cohort]
    assert len(ids) == len(set(ids)) == (100 if pilot_stage else 500)
    cached = read(source / 'protocol.json')
    assert cached['max_new_tokens'] == cap and cached.get('model', 'Qwen3-VL-8B-Instruct') == 'Qwen3-VL-8B-Instruct'
    for path in generation_sources():
        recorded = cached['source_sha256'].get(path, cached['source_sha256'].get(str(REPO / path)))
        if recorded is not None and recorded != sha(REPO / path):
            raise RuntimeError(f'Cached generation source differs: {path}')
    candidates = CANDIDATES
    selection = 'Compare both candidates on the original pilot batches.'
    if late:
        candidates = (LATE_DIRECTIONAL,)
        selection = 'Test later source layers with identical directional evidence, FFN preference and per-layer strength.'
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_late-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert LATE_DIRECTIONAL in read(pilot / 'screening.json')['advance_to_500']
    if consensus:
        candidates = (CONSENSUS,)
        selection = 'Only layer 26 receives the guided budget, and only when layers 24 and 25 both nominate the same supported token.'
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_consensus-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert CONSENSUS in read(pilot / 'screening.json')['advance_to_500']
    if alternative:
        candidates = (ALTERNATIVE,)
        selection = 'Use a nearby alternative noun when its actual signed visual support is stronger and its FFN preference drift is smaller.'
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_alternative-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert ALTERNATIVE in read(pilot / 'screening.json')['advance_to_500']
    if reference:
        candidates = ('svar',)
        selection = 'Generate a plain SVAR reference on the exact four batches/shards used by the SADI variants.'
    if margin:
        candidates = (MARGIN,)
        selection = 'Guide next-layer attention when visual writes favor a plausible rival and FFN gain favors the primary over that rival.'
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_margin-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert MARGIN in read(pilot / 'screening.json')['advance_to_500']
    if ffn_aux:
        candidates = (FFN_AUX,)
        selection = 'Keep the frozen alternative attention step and selectively reduce the FFN primary-rival margin gain.'
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_ffn-aux-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert FFN_AUX in read(pilot / 'screening.json')['advance_to_500']
    if low_visual:
        candidates = (LOW_UNIFORM, LOW_SPATIAL)
        selection = 'At layer 28, gate on low visual mass and a nearby unseen entity token. Compare uniform restoration and restoration with target-conditioned spatial guidance at layer 29.'
        trace = OUT / 'qwen3_sadi_first_object_trace'
        assert read(trace / 'status.json')['status'] == 'complete'
        trace_ids = set(read(trace / 'cohort.json')['trace'])
        assert not set(ids).intersection(trace_ids)
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_low-visual-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            pilot_rows = read(pilot / 'independent_summary.json')['eval']['sentence']
            spatial, uniform = pilot_rows[LOW_SPATIAL], pilot_rows[LOW_UNIFORM]
            assert spatial['chair_s'] < uniform['chair_s'] and spatial['chair_i'] < uniform['chair_i']
            assert spatial['audit_false_mentions'] < uniform['audit_false_mentions']
            assert spatial['object_f1'] >= uniform['object_f1']
            selection += (' 500-image follow-up tests the spatial increment against the uniform control. '
                          'The pilot did not lower CHAIRs against SADI. Frozen coefficients from the disjoint diagnostic cohort.')
    if removal_gate or verified_rival or negative_write or removal_same:
        candidates = (NEGATIVE_WRITE if negative_write else VERIFIED_RIVAL if verified_rival else REMOVAL_GATE,)
        selection = 'Gate layer 32 feedback on visual removal contrast of the primary and a plausible rival at layer 31.'
        if negative_write:
            selection = 'Restore visual attention to patches with negative write contributions to a weakly supported first entity.'
        if removal_same:
            candidates = (REMOVAL_SAME31, REMOVAL_SAME35)
            selection = 'Use the frozen visual-removal gate to recompute the last query of layer 31 or layer 35, reusing projected KV.'
        trace = OUT / 'qwen3_sadi_visual_removal_trace'
        label_cohort = OUT / 'qwen3_sadi_first_object_trace/cohort.json'
        assert read(trace / 'status.json')['status'] == 'complete'
        assert not set(ids).intersection(read(label_cohort)['trace'])
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_' + stage.replace('-500', '-pilot'))
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            promoted = read(pilot / 'screening.json')['advance_to_500']
            candidates = tuple(name for name in candidates if name in promoted)
            assert candidates
    if final_target:
        candidates = (FINAL_TARGET,)
        selection = ('Use the final candidate token with layer 31 visual-removal contrast and negative-write '
                     'evidence versus FFN spatial preference. Replay the current query through layers 31--35 '
                     'and replace the current cache positions of layers 32--35. Keep the previous negative-write coefficients.')
        trace = OUT / 'qwen3_sadi_terminal_write_trace'
        assert read(trace / 'status.json')['status'] == 'complete'
        assert read(trace / 'analysis.json')['replay_matches'] == 1707
        assert not set(ids).intersection(read(trace / 'cohort.json')['trace'])
        assert read(OUT / 'qwen3_final_target_preflight/verification.json')['passed']
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_final-target-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert FINAL_TARGET in read(pilot / 'screening.json')['advance_to_500']
    if head_gradient:
        candidates = (HEAD_FINAL, HEAD_SOURCE)
        selection = ('Compare source-only and final-target gradients of log probability at the same layer 31 '
                     'visual-write gate. Use a separate visual logit shift per head, FFN spatial drift, '
                     'and keep only proposals that lower the original final candidate probability. '
                     'Restore current tail KV and hidden states for rejected rows.')
        trace = OUT / 'qwen3_sadi_terminal_write_trace'
        preflight = OUT / 'qwen3_head_gradient_preflight/verification.json'
        batch_preflight = OUT / 'qwen3_head_gradient_preflight/batch16_verification.json'
        checked = read(preflight)
        assert checked['final']['passed'] and checked['source']['passed']
        assert read(batch_preflight)['passed']
        assert all(c['exact_baseline_logits_and_tokens'] for c in checked['forced_reject'].values())
        for filename, digest in checked['source_sha256'].items():
            assert sha(REPO / filename) == digest
        assert not set(ids).intersection(read(trace / 'cohort.json')['trace'])
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_head-gradient-pilot')
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            promoted = read(pilot / 'screening.json')['advance_to_500']
            candidates = tuple(name for name in candidates if name in promoted)
            if not candidates:
                result = read(pilot / 'independent_summary.json')['eval']['sentence']
                base, candidate = result[BASE], result[HEAD_FINAL]
                mentions = read(pilot / 'object_mention_audit.json')['sentence']['metrics']
                assert candidate['chair_s'] < base['chair_s'] and candidate['chair_i'] < base['chair_i']
                assert candidate['object_f1'] > base['object_f1'] and candidate['object_recall'] > base['object_recall']
                assert mentions[HEAD_FINAL]['false_mentions'] <= mentions[BASE]['false_mentions']
                assert mentions[HEAD_FINAL]['false_image_classes'] < mentions[BASE]['false_image_classes']
                candidates = (HEAD_FINAL,)
                selection += (' Post-pilot follow-up of fewer hallucinated images/classes and higher recall/F1. '
                              'Total false mentions were unchanged, so the earlier strict count-decrease screen did not promote it. '
                              'CHAIRi also benefits from more real object mentions. Validate the frozen recipe on 500 images.')
            assert candidates
    if svar:
        candidates = ('svar_sar_ems_visual_mass_19_20',) if visual_mass else ('svar_sar_ems_directional_19_20',)
        if svar_alternative:
            candidates = (SVAR_ALTERNATIVE,)
        selection = 'Test the frozen SADI directional-evidence recipe on the SVAR base.'
        if visual_mass:
            selection = 'Permit a mild visual attention mass increase only when the actual writes have net positive target support.'
        if svar_alternative:
            selection = 'Test the frozen nearby-candidate attention guidance on the SVAR base.'
        if not pilot_stage:
            pilot = OUT / (PREFIX + 'signed_feedback_' + stage.replace('-500', '-pilot'))
            assert not set(ids).intersection(read(pilot / 'cohort.json')['coco100'])
            assert tuple(read(pilot / 'screening.json')['advance_to_500']) == candidates
    if stage == '500':
        pilot = OUT / (PREFIX + 'signed_feedback_pilot')
        pilot_ids = read(pilot / 'cohort.json')['coco100']
        assert not set(ids).intersection(pilot_ids)
        candidates = tuple(read(pilot / 'screening.json')['advance_to_500'])
        selection = 'Candidates satisfying the preregistered pilot screening rule.'
        if validate_directional_tradeoff:
            pilot_result = read(pilot / 'independent_summary.json')['eval']['sentence']
            base = pilot_result[BASE]
            candidate = pilot_result[DIRECTIONAL]
            pair = candidate['paired_against'][BASE]
            assert pair['delta_ci95']['chair_i'][1] < 0
            assert candidate['audit_false_mentions'] < base['audit_false_mentions']
            assert candidate['object_f1'] >= base['object_f1']
            candidates = (DIRECTIONAL,)
            selection = ('Separate CHAIRi follow-up after the pilot. Neither candidate passed the joint screening rule. '
                         'Directional evidence reduced false mentions and CHAIRi, with the paired CHAIRi interval below zero and higher mean F1. '
                         'Its pilot CHAIRs increased by 1 pp. Validate the tradeoff on the disjoint 500-image cohort at the cached 256-token cap.')
        assert candidates and set(candidates).issubset(CANDIDATES)
    shards = 4 if stage in ('500', 'late-500', 'consensus-500', 'alternative-500', 'reference-500',
                           'low-visual-500', 'removal-gate-500', 'verified-rival-500', 'negative-write-500',
                           'removal-same-500', 'final-target-500', 'head-gradient-500') else 2
    source_sha = {}
    seen = []
    for shard in range(shards):
        paths = [source / cohort / f'{name}.shard{shard}.jsonl' for name in controls]
        groups = [source_groups(path) for path in paths]
        keys = group_keys(groups[0])
        assert all(keys == group_keys(group) for group in groups[1:])
        seen.extend(i for batch, _ in keys for i in batch)
        source_sha.update({str(path): sha(path) for path in paths})
    assert sorted(seen) == sorted(ids)
    new_files = ('features/qwen3_sar_ems_signed.py', 'scripts/run_qwen3_signed_feedback.py',
                 'scripts/evaluate_caption_chair.py', 'scripts/evaluate_generation_time_evidence_drift_llava.py',
                 'scripts/evaluate_qwen3_next200_sentence_tokenized.py', 'coco-labeling/coco_chair.py')
    if svar:
        new_files += ('features/qwen3_svar_layer_feedback.py', 'features/qwen3_sar_ems_attention.py')
    if visual_mass or svar_alternative:
        new_files += ('features/qwen3_sar_ems_visual_mass.py',)
    if consensus:
        new_files += ('features/qwen3_sar_ems_consensus.py',)
    if alternative or reference or svar_alternative or margin or ffn_aux:
        new_files += ('features/qwen3_sar_ems_alternative.py',)
    if margin:
        new_files += ('features/qwen3_sar_ems_margin.py',)
    if ffn_aux:
        new_files += ('features/qwen3_sar_ems_ffn_aux.py',)
    if low_visual:
        new_files += ('features/qwen3_sar_ems_low_visual.py',)
        for path in (trace / 'protocol.json', trace / 'cohort.json', trace / 'analysis.json'):
            source_sha[str(path)] = sha(path)
    if removal_gate or verified_rival or negative_write or removal_same:
        new_files += ('features/qwen3_sar_ems_removal_gate.py', 'features/qwen3_visual_write_contrast.py')
        for path in (trace / 'protocol.json', trace / 'analysis.json', label_cohort):
            source_sha[str(path)] = sha(path)
    if verified_rival:
        new_files += ('features/qwen3_sar_ems_verified_rival.py',)
    if negative_write:
        new_files += ('features/qwen3_sar_ems_negative_write.py',)
    if removal_same:
        new_files += ('features/qwen3_sar_ems_removal_same.py',)
    if final_target:
        new_files += ('features/qwen3_sar_ems_final_target.py',
                      'features/qwen3_sar_ems_removal_gate.py', 'features/qwen3_visual_write_contrast.py',
                      'scripts/verify_qwen3_final_target_replay.py')
        for path in (trace / 'protocol.json', trace / 'cohort.json', trace / 'analysis.json',
                     OUT / 'qwen3_final_target_preflight/verification.json'):
            source_sha[str(path)] = sha(path)
    if head_gradient:
        new_files += ('features/qwen3_sar_ems_head_gradient.py', 'features/qwen3_sar_ems_final_target.py',
                      'features/qwen3_sar_ems_removal_gate.py', 'features/qwen3_visual_write_contrast.py')
        source_sha[str(preflight)] = sha(preflight)
        source_sha[str(batch_preflight)] = sha(batch_preflight)
    source_sha.update({str(REPO / path): sha(REPO / path) for path in (*generation_sources(), *new_files)})
    output.mkdir()
    (output / cohort).mkdir()
    (output / 'logs').mkdir()
    for shard in range(shards):
        for name in controls:
            filename = f'{name}.shard{shard}.jsonl'
            shutil.copyfile(source / cohort / filename, output / cohort / filename)
    write(output / 'cohort.json', {cohort: ids})
    protocol = {
        'source': str(source), 'cohort': cohort, 'stage': stage, 'images': len(ids), 'max_new_tokens': cap,
        'model': 'Qwen3-VL-8B-Instruct',
        'variants': [*controls, *candidates], 'candidates': list(candidates),
        'cached_controls': list(controls), 'shards': shards, 'source_sha256': source_sha,
        'batch_policy': 'Replay exact cached ordered batches, preserve original GPU assignment.', 'selection': selection,
        'recipe': {'signed': '5 * HellingerSquared(Q,P) * tanh(gain/2) * (Q-P) / max(range(Q-P),1e-6)',
                   'directional': 'Original positive-only correction; replace Q with normalized positive target-logit projections of actual patch writes.',
                   'visual_mass': 'Add uniform visual logit shift 0.5 * drift * tanh(gain/2) * clamp(sum(signed_writes)/sum(positive_writes),0,1) after within-vision rebalancing.',
                   'sources': [24, 25] if late or consensus or alternative or svar_alternative or margin or ffn_aux else [18, 19],
                   'targets': ([26] if consensus else [25, 26]) if late or consensus or alternative or svar_alternative or margin or ffn_aux else [19, 20],
                   'base_layers': [5, 18], 'tau': 0.03},
        'screening_rule': 'Both sentence CHAIR scores and false mention count decrease versus the base, with at most 1.5pp F1 and 2pp recall loss. CHAIRi must remain no worse without the biggest single-image false-mention reduction.',
    }
    if low_visual:
        protocol['recipe'] = {'sources': [28], 'targets': [29], 'base_layers': [5, 18],
                              'mass_threshold': 0.12, 'maximum_shared_logit_shift': 0.5,
                              'top_k': 8, 'logit_gap_limit': 2, 'exclude_prior_token_id': True,
                              'maximum_spatial_scale': 2, 'tau': 0.03}
    if removal_gate or verified_rival or removal_same:
        protocol['recipe'] = {'sources': [31], 'targets': [32], 'base_layers': [5, 18],
                              'top_k': 8, 'logit_gap_limit': 2, 'support_threshold': 0.35,
                              'minimum_support_advantage': 0.35, 'exclude_prior_primary_token_id': True,
                              'budget_scale': '5 * HellingerSquared(Q_rival,P) * tanh(visual_advantage)', 'tau': 0.03}
    if verified_rival:
        protocol['recipe']['final_layer_check'] = 'On layer 35, require stronger positive rival visual write and contrary FFN gain, then penalize an unchanged unsupported near-tie by at most 0.5 logit.'
    if removal_same:
        protocol['recipe']['sources'] = protocol['recipe']['targets'] = [31, 35]
        protocol['recipe']['layer_assignment'] = {name: (31 if name == REMOVAL_SAME31 else 35) for name in candidates}
        protocol['recipe']['replay'] = 'One last-query attention and FFN replay at the assigned layer; no extra KV updates.'
    if negative_write:
        protocol['recipe'] = {'sources': [31], 'targets': [32], 'base_layers': [5, 18], 'support_threshold': 0.35,
                              'exclude_prior_primary_token_id': True, 'tau': 0.03,
                              'budget_scale': '3 * HellingerSquared(Q_negative,P) * negative_write_share * tanh(0.35-support)'}
    if final_target:
        protocol['recipe'] = {'sources': [31], 'targets': [31], 'final_target_read_layer': 35,
                              'replayed_tail_layers': [32, 33, 34, 35], 'base_layers': [5, 18],
                              'skip_prefill': True, 'support_threshold': 0.35,
                              'exclude_prior_primary_token_id': True, 'tau': 0.03,
                              'budget_scale': '3 * HellingerSquared(Q_negative,P) * negative_write_share * tanh(0.35-support)',
                              'cache_policy': 'Replace selected current positions in layers 32--35; preserve earlier positions.'}
    if head_gradient:
        protocol['recipe'] = {'sources': [31], 'targets': [31], 'final_target_read_layer': 35,
                              'gradient_scope': {HEAD_FINAL: 'source FFN and tail layers 32--35', HEAD_SOURCE: 'source FFN only'},
                              'objective': 'log probability of the original final candidate',
                              'support_threshold': .35, 'skip_prefill': True, 'exclude_prior_token_id': True,
                              'base_layers': [5, 18], 'tau': .03, 'maximum_scale': 3, 'spatial_floor': .25,
                              'correction': 'Negative centered per-head visual logit derivative, weighted by the positive Q_negative-P deficit.',
                              'acceptance': 'Final candidate probability strictly decreases after the proposal.',
                              'cache_policy': 'Read-only cached-prefix gradient probe; replace current tail positions for accepted rows and restore rejected rows.'}
    write(output / 'protocol.json', protocol)
    return output, protocol


def worker(output, gpu):
    import torch
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_sar_ems_signed import Qwen3BatchSignedLayerFeedback
    from features import qwen3_sar_ems_layer_feedback as feedback
    from features.qwen3_svar_layer_feedback import Qwen3BatchSvarLayerFeedback, svar_visual_logits
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as shared
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config

    protocol = read(output / 'protocol.json')
    cohort = output / protocol['cohort']
    cap = protocol['max_new_tokens']
    controls = protocol['cached_controls']
    svar = controls[0] == 'svar'
    if protocol['stage'].startswith(('svar-mass-', 'svar-alternative-')):
        from features.qwen3_sar_ems_visual_mass import Qwen3BatchSupportedVisualMass
    if protocol['stage'].startswith('consensus-'):
        from features.qwen3_sar_ems_consensus import Qwen3BatchConsensusFeedback
    if protocol['stage'].startswith(('alternative-', 'svar-alternative-', 'margin-', 'ffn-aux-')) or protocol['stage'] == 'reference-500':
        from features.qwen3_sar_ems_alternative import Qwen3BatchSupportedAlternative
    if protocol['stage'].startswith('margin-'):
        from features.qwen3_sar_ems_margin import Qwen3BatchMarginFeedback
    if protocol['stage'].startswith('ffn-aux-'):
        from features.qwen3_sar_ems_ffn_aux import Qwen3BatchAlternativeWithFFNAux
    if protocol['stage'].startswith('low-visual-'):
        from features.qwen3_sar_ems_low_visual import Qwen3BatchLowVisualFeedback
    if protocol['stage'].startswith(('removal-gate-', 'verified-rival-', 'negative-write-', 'removal-same-')):
        from features.qwen3_sar_ems_removal_gate import Qwen3BatchWriteRemovalGate
    if protocol['stage'].startswith('verified-rival-'):
        from features.qwen3_sar_ems_verified_rival import Qwen3BatchVerifiedRival
    if protocol['stage'].startswith('negative-write-'):
        from features.qwen3_sar_ems_negative_write import Qwen3BatchNegativeWrite
    if protocol['stage'].startswith('removal-same-'):
        from features.qwen3_sar_ems_removal_same import Qwen3BatchWriteRemovalSame
    if protocol['stage'].startswith(('final-target-', 'head-gradient-')):
        from features.qwen3_sar_ems_final_target import Qwen3BatchFinalTargetReplay
    if protocol['stage'].startswith('head-gradient-'):
        from features.qwen3_sar_ems_head_gradient import Qwen3BatchHeadGradient
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config['hf_name'] = str(MODEL_ROOT)
    torch.set_num_threads(1)
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    assert nouns

    def generate_candidate(span, group, name, max_tokens=cap):
        if name == 'svar':
            return shared.generate_group(wrapper, span, group, nouns, name, max_tokens)
        def factory(model, span, nouns, mode, strength):
            assert mode == 'next'
            if 'visual_mass' in name:
                return Qwen3BatchSupportedVisualMass(model, span, nouns, strength=strength)
            if name == CONSENSUS:
                return Qwen3BatchConsensusFeedback(model, span, nouns, strength=strength)
            if name in (ALTERNATIVE, SVAR_ALTERNATIVE):
                return Qwen3BatchSupportedAlternative(model, span, nouns, strength=strength)
            if name == MARGIN:
                return Qwen3BatchMarginFeedback(model, span, nouns, strength=strength)
            if name == FFN_AUX:
                return Qwen3BatchAlternativeWithFFNAux(model, span, nouns, strength=strength)
            if name in (LOW_UNIFORM, LOW_SPATIAL):
                return Qwen3BatchLowVisualFeedback(model, span, nouns, spatial=name == LOW_SPATIAL)
            if name == REMOVAL_GATE:
                return Qwen3BatchWriteRemovalGate(model, span, nouns)
            if name == VERIFIED_RIVAL:
                return Qwen3BatchVerifiedRival(model, span, nouns)
            if name == NEGATIVE_WRITE:
                return Qwen3BatchNegativeWrite(model, span, nouns)
            if name == FINAL_TARGET:
                return Qwen3BatchFinalTargetReplay(model, span, nouns)
            if name in (HEAD_FINAL, HEAD_SOURCE):
                return Qwen3BatchHeadGradient(model, span, nouns, final_gradient=name == HEAD_FINAL)
            if name in (REMOVAL_SAME31, REMOVAL_SAME35):
                return Qwen3BatchWriteRemovalSame(model, span, nouns, layer=31 if name == REMOVAL_SAME31 else 35)
            controller = Qwen3BatchSignedLayerFeedback(model, span, nouns, strength=strength,
                                                       directional_evidence=(name != SIGNED))
            if name == LATE_DIRECTIONAL:
                controller.sources = (24, 25)
                controller.targets = (25, 26)
            return controller
        with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
            if svar:
                with patch.object(feedback, 'sadi_visual_logits', svar_visual_logits):
                    return original.generate_group(wrapper, span, group, nouns, POSITIVE, max_tokens)
            return original.generate_group(wrapper, span, group, nouns, POSITIVE, max_tokens)

    def generate_control(span, inputs, name, cap):
        if name == 'svar':
            return shared.generate_group(wrapper, span, inputs, nouns, name, cap)
        if name in (DIRECTIONAL, LATE_DIRECTIONAL, ALTERNATIVE, REMOVAL_GATE, FINAL_TARGET):
            return generate_candidate(span, inputs, name, cap)
        if not svar:
            return original.generate_group(wrapper, span, inputs, nouns, name, cap)
        if name in ('svar_sar_ems_directional_19_20', 'svar_sar_ems_visual_mass_19_20'):
            return generate_candidate(span, inputs, name, cap)
        if name == 'svar_sar_ems_next_19_20':
            def factory(model, span, nouns, mode, strength):
                assert mode == 'next'
                return Qwen3BatchSvarLayerFeedback(model, span, nouns, strength=strength)
            with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                return original.generate_group(wrapper, span, inputs, nouns, POSITIVE, cap)
        return shared.generate_group(wrapper, span, inputs, nouns, name, cap)

    def inputs_for(group):
        ids = [row['image_id'] for row in group]
        (span, inputs), = shared.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
        assert [i for i, _ in inputs] == ids
        assert span[1] - span[0] == group[0]['visual_token_count']
        return span, inputs

    def compare_control(shard):
        control_groups = [source_groups(cohort / f'{name}.shard{shard}.jsonl') for name in controls]
        base_groups = control_groups[0]
        index = min(range(len(base_groups)), key=lambda i: (len(base_groups[i]) != 2, len(base_groups[i])))
        span, inputs = inputs_for(base_groups[index])
        checked = {}
        for name, groups in zip(controls, control_groups, strict=True):
            rows = groups[index]
            reference = [row['response_token_ids'] for row in rows]
            prefix_len = min(64, max(map(len, reference)))
            tokens, _, _ = generate_control(span, inputs, name, prefix_len)
            assert tokens == [tokens[:prefix_len] for tokens in reference], ('Cached control replay changed', shard, name)
            checked[name] = {'replayed_tokens': prefix_len, 'image_ids': [i for i, _ in inputs]}
        return checked

    with torch.inference_mode():
        checked = {str(shard): compare_control(shard) for shard in range(gpu, protocol['shards'], 2)}
        write(output / f'verification.gpu{gpu}.json', checked)
        for shard in range(gpu, protocol['shards'], 2):
            old_groups = source_groups(cohort / f'{controls[0]}.shard{shard}.jsonl')
            paths = {name: cohort / f'{name}.shard{shard}.jsonl' for name in protocol['candidates']}
            if any(path.exists() for path in paths.values()):
                raise FileExistsError('Existing candidate captions')
            for group in old_groups:
                span, inputs = inputs_for(group)
                for name, path in paths.items():
                    token_ids, stats, seconds = generate_candidate(span, inputs, name)
                    texts = wrapper.processor.batch_decode(token_ids, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
                    with path.open('a', encoding='utf-8') as stream:
                        for (image_id, _), tokens, text, info in zip(inputs, token_ids, texts, stats, strict=True):
                            assert 0 < len(tokens) <= cap
                            if name == 'svar':
                                assert not info
                            else:
                                expected_replay = info['guided_attention_calls'] if name in (REMOVAL_SAME31, REMOVAL_SAME35, FINAL_TARGET, HEAD_FINAL, HEAD_SOURCE) else 0
                                assert info['local_recompute_events'] == expected_replay and not info['diagnostic_force_guidance']
                                assert not info.get('diagnostic_reject_all', False)
                            row = dict(image_id=image_id, model=MODEL, variant=name, generated_text=text.strip(),
                                       response_token_ids=tokens, seconds=seconds / len(group),
                                       batch_size=len(group), batch_wall_seconds=seconds,
                                       visual_token_count=span[1] - span[0], sar_ems_statistics=info,
                                       decoder='native_qwen3_generate_same_span_batch')
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    print(f'[{shard}] {name} batch={len(group)} seconds={seconds:.3f}', flush=True)
            assert all(group_keys(source_groups(path)) == group_keys(old_groups) for path in paths.values())


def evaluate(output, protocol):
    cohort, n = protocol['cohort'], protocol['images']
    names = protocol['variants']
    for filename, expected in protocol['source_sha256'].items():
        assert sha(Path(filename)) == expected, filename
    tag = '-'.join(sorted(names))
    for mode in ('sentence', 'legacy'):
        subprocess.run([sys.executable, str(REPO / 'scripts/evaluate_caption_chair.py'),
                        '--generation-root', str(output), '--cohort', cohort, '--limit', str(n),
                        '--variants', *names, '--evaluator', mode], cwd=REPO, check=True)
    target = output / f'evaluations/sentence_v1/{cohort}'
    result = read(target / f'metrics.first{n}.{tag}.json')['methods']
    base_name = protocol['cached_controls'][0]
    base = result[base_name]
    scores = [json.loads(line)['methods'] for line in
              (target / f'chair_per_image.first{n}.{tag}.jsonl').read_text(encoding='utf-8').splitlines()]
    false_key, mention_key = 'hallucinated_mentions', 'object_mentions'
    false_counts = {name: sum(row[name][false_key] for row in scores) for name in names}
    if protocol['stage'] == 'reference-500':
        write(output / 'screening.json', {'advance_to_500': [], 'purpose': 'Matched SVAR reference',
                                          'hallucinated_mentions': false_counts})
        return
    advance, by_candidate = [], {}
    for candidate in protocol['candidates']:
        enhanced = result[candidate]
        largest = max(range(n), key=lambda i: scores[i][base_name][false_key] - scores[i][candidate][false_key])
        trimmed_ci = {name: sum(row[name][false_key] for i, row in enumerate(scores) if i != largest) /
                      max(1, sum(row[name][mention_key] for i, row in enumerate(scores) if i != largest))
                      for name in (base_name, candidate)}
        checks = {
            'CHAIRs_decreases': enhanced['chair_s'] < base['chair_s'],
            'CHAIRi_decreases': enhanced['chair_i'] < base['chair_i'],
            'false_mentions_decrease': false_counts[candidate] < false_counts[base_name],
            'F1_loss_at_most_1_5pp': enhanced['object_f1'] >= base['object_f1'] - 0.015,
            'recall_loss_at_most_2pp': enhanced['object_recall'] >= base['object_recall'] - 0.02,
            'CHAIRi_no_worse_without_largest_reduction': trimmed_ci[candidate] <= trimmed_ci[base_name],
        }
        by_candidate[candidate] = {'checks': checks, 'leave_one_out_chair_i': trimmed_ci}
        if all(checks.values()):
            advance.append(candidate)
    write(output / 'screening.json', {'advance_to_500': advance, 'candidates': by_candidate,
                                      'hallucinated_mentions': false_counts})


def run(stage, validate_directional_tradeoff=False):
    output, protocol = prepare(stage, validate_directional_tradeoff)
    children = []
    try:
        for gpu in range(2):
            with (output / f'logs/gpu{gpu}.log').open('w') as log:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                           PYTHONUNBUFFERED='1')
                proc = subprocess.Popen([sys.executable, __file__, '--worker', str(output), '--gpu', str(gpu)],
                                        cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append(proc)
        write(output / 'run_status.json', {'status': 'generating', 'pids': [p.pid for p in children]})
        codes = [p.wait() for p in children]
        if any(codes):
            raise RuntimeError(f'GPU workers failed: {codes}')
        write(output / 'run_status.json', {'status': 'evaluating'})
        evaluate(output, protocol)
        write(output / 'run_status.json', {'status': 'complete'})
    except Exception as error:
        write(output / 'run_status.json', {'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=RUNS)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    parser.add_argument('--validate-directional-tradeoff', action='store_true')
    args = parser.parse_args()
    if args.worker is not None and args.gpu is not None:
        worker(args.worker, args.gpu)
    elif args.stage is not None:
        if args.validate_directional_tradeoff and args.stage != '500':
            parser.error('Directional tradeoff follow-up requires --stage 500')
        run(args.stage, args.validate_directional_tradeoff)
    else:
        parser.error('Specify --stage or --worker with --gpu')
