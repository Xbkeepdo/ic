"""Evaluate native single-image SVAR captions on the fixed 500-image cohort."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


REPO = Path('/root/rivermind-data/ic-ac')
ROOT = REPO / 'outputs/generation_time_evidence_drift_qwen3_single500'
ORIGINAL = REPO / 'outputs/generation_time_evidence_drift_qwen3_coco500'
FOLDER = ROOT / 'coco500'
COCO = REPO.parent / 'dataset/coco'
NAMES = ('svar', 'svar_sar_ems_attention')
METRICS = ('chair_s', 'chair_i', 'object_recall', 'object_f1')
COUNTS = ('hallucinated_images', 'hallucinated_mentions', 'real_mentions', 'object_mentions')
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))

from data.qa_benchmark import atomic_write_json
from scripts.evaluate_generation_time_evidence_drift_llava import (
    _bootstrap_metrics, evaluate, summarize,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def compact_summary(rows: list[dict]) -> dict:
    summary = summarize(rows)
    return {key: summary[key] for key in (
        'images', *METRICS, *COUNTS, 'tokens_per_image', 'words_per_image',
    )}


def paired_summary(rows: list[dict], *, bootstrap: bool = False) -> dict:
    scores = {name: [row['methods'][name] for row in rows] for name in NAMES}
    methods = {name: compact_summary(scores[name]) for name in NAMES}
    base, late = NAMES
    pairs = list(zip(scores[base], scores[late], strict=True))
    delta = {key: methods[late][key] - methods[base][key] for key in METRICS}
    comparison = {
        'delta': delta,
        'fixed_hallucinated_images': sum(a['chair_s'] > b['chair_s'] for a, b in pairs),
        'introduced_hallucinated_images': sum(a['chair_s'] < b['chair_s'] for a, b in pairs),
    }
    if bootstrap:
        draws = np.random.default_rng(43).integers(0, len(rows), size=(5000, len(rows)))
        boot = {name: _bootstrap_metrics(scores[name], draws) for name in NAMES}
        comparison['delta_ci95'] = {
            key: np.quantile(boot[late][key] - boot[base][key], [.025, .975]).tolist()
            for key in METRICS
        }
    return {'methods': methods, 'late_paired_against_svar': comparison}


def main() -> None:
    ids = list(map(int, read_json(ROOT / 'cohort.json')['coco500']))
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError('Expected the frozen cohort of 500 unique images')
    if ids != list(map(int, read_json(ORIGINAL / 'cohort.json')['coco500'])):
        raise ValueError('Single-image cohort must match the original frozen cohort order')
    raw = {name: {} for name in NAMES}
    for name in NAMES:
        for shard in range(2):
            path = FOLDER / f'{name}.shard{shard}.jsonl'
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(records) != 250 or {int(row['image_id']) for row in records} != set(ids[shard::2]):
                raise ValueError(f'Incomplete, duplicated, or unexpected images in {path}')
            for row in records:
                image_id = int(row['image_id'])
                if image_id in raw[name] or row['variant'] != name or not row['response_token_ids']:
                    raise ValueError(f'Invalid raw caption for {image_id} in {path}')
                if row['decoder'] != 'native_qwen3_generate' or row.get('batch_size', 1) != 1:
                    raise ValueError(f'Expected native single-image generation for {image_id}')
                if row['model'] != 'qwen3_vl_8b':
                    raise ValueError(f'Unexpected model for image {image_id}')
                raw[name][image_id] = row

    base, late = NAMES
    for image_id in ids:
        left, right = raw[base][image_id], raw[late][image_id]
        for key in ('visual_token_count', 'decoder', 'model'):
            if left[key] != right[key]:
                raise ValueError(f'Paired protocol mismatch: image {image_id}, field {key}')
        stats = right['sar_ems_statistics']
        if stats['guided_attention_calls'] != 2 * stats['guided_steps']:
            raise ValueError(f'Late attention calls do not match triggers for image {image_id}')

    metrics_path = evaluate(ROOT, 'coco500', list(NAMES), COCO, 500, 5000, None)
    report = read_json(metrics_path)
    per_image_path = Path(report['per_image'])
    rows = [json.loads(line) for line in per_image_path.read_text(encoding='utf-8').splitlines()]
    if [row['image_id'] for row in rows] != ids:
        raise ValueError('Scored image order does not match the frozen cohort')
    by_id = {row['image_id']: row for row in rows}

    uncapped = [row for row in rows if all(row['methods'][name]['tokens'] < 512 for name in NAMES)]
    changes = [{
        'image_id': row['image_id'],
        'late_minus_svar': row['methods'][late]['hallucinated_mentions'] - row['methods'][base]['hallucinated_mentions'],
        'by_method': {name: row['methods'][name]['hallucinated_mentions'] for name in NAMES},
    } for row in rows]
    largest_absolute = max(changes, key=lambda row: (abs(row['late_minus_svar']), -row['image_id']))
    largest_reduction = min(changes, key=lambda row: (row['late_minus_svar'], row['image_id']))
    diagnostics = {}
    for label, case in (('largest_absolute_change', largest_absolute), ('largest_reduction', largest_reduction)):
        remaining = [row for row in rows if row['image_id'] != case['image_id']]
        diagnostics[label] = {'case': case, 'without_image': paired_summary(remaining)}

    guidance = [raw[late][image_id]['sar_ems_statistics'] for image_id in ids]
    validation = {
        'primary': str(metrics_path),
        'paired_images': len(ids),
        'captions': sum(len(records) for records in raw.values()),
        'native_single_image_generation': True,
        'same_visual_count_decoder_model': True,
        'frozen_cohort_matches_previous_500': True,
        'caps': {name: sum(row['methods'][name]['tokens'] >= 512 for row in rows) for name in NAMES},
        'n_common_uncapped': len(uncapped),
        'common_uncapped': paired_summary(uncapped, bootstrap=True) if uncapped else None,
        'guidance': {
            'guided_images': sum(stats['guided_steps'] > 0 for stats in guidance),
            'guided_steps': sum(stats['guided_steps'] for stats in guidance),
            'target_matches_generated': sum(stats['target_matches_generated'] for stats in guidance),
            'guided_attention_calls': sum(stats['guided_attention_calls'] for stats in guidance),
            'changed_token_sequences': sum(raw[base][image_id]['response_token_ids'] != raw[late][image_id]['response_token_ids'] for image_id in ids),
        },
        'hallucinated_mention_changes': {
            'decreased_images': sum(case['late_minus_svar'] < 0 for case in changes),
            'increased_images': sum(case['late_minus_svar'] > 0 for case in changes),
            'unchanged_images': sum(case['late_minus_svar'] == 0 for case in changes),
        },
        'influence': diagnostics,
    }
    validation_path = ROOT / 'validation.single500.json'
    atomic_write_json(validation_path, validation)

    previous_nested = read_json(ORIGINAL / 'metrics.nested_seed43.json')
    draw = list(map(int, previous_nested['draw_ids']))
    if len(draw) != 500 or len(set(draw)) != 500 or set(draw) != set(ids):
        raise ValueError('Saved original sampling order does not match the frozen cohort')
    nested = {
        'selection': previous_nested['selection'],
        'draw_ids': draw,
        'bootstrap_seed': 43,
        'bootstrap_draws': 5000,
        'subsets': {},
    }
    for count in (100, 200, 300, 400, 500):
        selected = [by_id[image_id] for image_id in sorted(draw[:count])]
        nested['subsets'][str(count)] = paired_summary(selected, bootstrap=True)
    nested_path = ROOT / 'metrics.nested_seed43.single500.json'
    atomic_write_json(nested_path, nested)
    print(json.dumps({
        'metrics': str(metrics_path),
        'validation': str(validation_path),
        'nested': str(nested_path),
        'methods': {name: {key: report['methods'][name][key] for key in (*METRICS, *COUNTS)} for name in NAMES},
        'paired': report['methods'][late]['paired_against'][base],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
