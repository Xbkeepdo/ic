"""Validate and score the fresh paired Qwen3 COCO200 experiment (maximum batch 16)."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np


REPO = Path('/root/rivermind-data/ic-ac')
ROOT = REPO / 'outputs/generation_time_evidence_drift_qwen3_new200_b16'
FOLDER = ROOT / 'coco200'
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


def load_and_validate(ids: list[int]) -> tuple[dict, dict]:
    """Recover complete ordered batches from consecutive, append-only shard records."""
    raw = {name: {} for name in NAMES}
    plans = {name: {} for name in NAMES}
    for name in NAMES:
        for shard in range(2):
            path = FOLDER / f'{name}.shard{shard}.jsonl'
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            expected = set(ids[shard::2])
            if len(records) != 100 or {int(row['image_id']) for row in records} != expected:
                raise ValueError(f'{path}: expected exactly 100 unique images from the frozen shard')
            for row in records:
                image_id = int(row['image_id'])
                if image_id in raw[name] or row['variant'] != name or row['model'] != 'qwen3_vl_8b':
                    raise ValueError(f'{path}: duplicate image or wrong variant/model for {image_id}')
                if row['decoder'] != 'native_qwen3_generate_same_span_batch':
                    raise ValueError(f'{path}: unexpected decoder for {image_id}')
                size = row['batch_size']
                if type(size) is not int or not 1 <= size <= 16:
                    raise ValueError(f'{path}: invalid batch size for {image_id}')
                tokens = row['response_token_ids']
                if not isinstance(tokens, list) or not tokens or len(tokens) > 512:
                    raise ValueError(f'{path}: empty or over-limit generation for {image_id}')
                if not isinstance(row['generated_text'], str):
                    raise ValueError(f'{path}: invalid caption for {image_id}')
                if type(row['visual_token_count']) is not int or row['visual_token_count'] <= 0:
                    raise ValueError(f'{path}: invalid visual token count for {image_id}')
                for field in ('seconds', 'batch_wall_seconds'):
                    if not math.isfinite(float(row[field])) or float(row[field]) < 0:
                        raise ValueError(f'{path}: invalid {field} for {image_id}')
                if name == NAMES[1]:
                    stats = row['sar_ems_statistics']
                    if (not 0 <= stats['target_matches_generated'] <= stats['guided_steps'] <= len(tokens)
                            or stats['guided_attention_calls'] != 2 * stats['guided_steps']):
                        raise ValueError(f'{path}: inconsistent guidance statistics for {image_id}')
                raw[name][image_id] = row

            batches = []
            offset = 0
            while offset < len(records):
                first = records[offset]
                size = first['batch_size']
                group = records[offset:offset + size]
                if len(group) != size:
                    raise ValueError(f'{path}: incomplete final batch at row {offset}')
                for row in group:
                    if (row['batch_size'] != size
                            or row['visual_token_count'] != first['visual_token_count']
                            or row['batch_wall_seconds'] != first['batch_wall_seconds']):
                        raise ValueError(f'{path}: inconsistent batch metadata at row {offset}')
                    if not math.isclose(row['seconds'] * size, row['batch_wall_seconds'], rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f'{path}: inconsistent per-image batch timing at row {offset}')
                batches.append({
                    'image_ids': [int(row['image_id']) for row in group],
                    'batch_size': size,
                    'visual_token_count': first['visual_token_count'],
                })
                offset += size
            plans[name][str(shard)] = batches

    base, late = NAMES
    if plans[base] != plans[late]:
        raise ValueError('The two methods used different ordered batches, batch sizes, or visual token counts')
    for image_id in ids:
        left, right = raw[base][image_id], raw[late][image_id]
        for field in ('model', 'decoder', 'visual_token_count', 'batch_size'):
            if left[field] != right[field]:
                raise ValueError(f'Paired protocol mismatch for image {image_id}: {field}')
    return raw, plans[base]


def paired_summary(rows: list[dict], *, bootstrap: bool = False) -> dict | None:
    if not rows:
        return None
    scores = {name: [row['methods'][name] for row in rows] for name in NAMES}
    fields = ('images', *METRICS, *COUNTS, 'tokens_per_image', 'words_per_image')
    methods = {}
    for name in NAMES:
        summary = summarize(scores[name])
        methods[name] = {key: summary[key] for key in fields}
    base, late = NAMES
    pairs = list(zip(scores[base], scores[late], strict=True))
    comparison = {
        'delta': {key: methods[late][key] - methods[base][key] for key in METRICS},
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


def log_summary() -> dict:
    result = {}
    for shard in range(2):
        path = ROOT / 'logs' / f'gpu{shard}.log'
        if not path.is_file():
            result[str(shard)] = {'path': str(path), 'available': False, 'oom_split_count': None, 'traceback_count': None}
            continue
        content = path.read_text(encoding='utf-8', errors='replace')
        result[str(shard)] = {
            'path': str(path), 'available': True,
            'oom_split_count': len(re.findall(rf'^\[oom:{shard}\] splitting\b', content, re.MULTILINE)),
            'traceback_count': content.count('Traceback (most recent call last):'),
            'cuda_oom_error_lines': sum('OutOfMemoryError' in line or 'CUDA out of memory' in line for line in content.splitlines()),
        }
    return result


def main() -> None:
    ids = list(map(int, read_json(ROOT / 'cohort.json')['coco200']))
    if len(ids) != len(set(ids)) or len(ids) != 200:
        raise ValueError('Expected a frozen cohort of 200 unique images')
    selection_ids_sha256 = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()
    if selection_ids_sha256 != read_json(ROOT / 'protocol.json')['selected_sha256']:
        raise ValueError('Frozen image manifest differs from the protocol selection hash')
    raw, batches = load_and_validate(ids)

    # A new cohort needs its own ground-truth scope. This reads the full COCO
    # annotations once and stores the scoped cache under this new output root;
    # later invocations reuse it. Never pass the old 500-image cache here.
    metrics_path = evaluate(ROOT, 'coco200', list(NAMES), COCO, 200, 5000, None)
    primary = read_json(metrics_path)
    rows = [json.loads(line) for line in Path(primary['per_image']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['image_id'] for row in rows] != ids:
        raise ValueError('Scored image IDs/order differ from the frozen cohort')
    base, late = NAMES
    uncapped = [row for row in rows if all(row['methods'][name]['tokens'] < 512 for name in NAMES)]
    actual_batch16 = [row for row in rows if raw[base][row['image_id']]['batch_size'] == 16]
    changes = [{
        'image_id': row['image_id'],
        'late_minus_svar': row['methods'][late]['hallucinated_mentions'] - row['methods'][base]['hallucinated_mentions'],
        'by_method': {name: row['methods'][name]['hallucinated_mentions'] for name in NAMES},
    } for row in rows]
    cases = {
        'largest_absolute_change': max(changes, key=lambda row: (abs(row['late_minus_svar']), -row['image_id'])),
        'largest_reduction': min(changes, key=lambda row: (row['late_minus_svar'], row['image_id'])),
        'largest_increase': max(changes, key=lambda row: (row['late_minus_svar'], -row['image_id'])),
    }
    influence = {
        label: {'case': case, 'without_image': paired_summary([row for row in rows if row['image_id'] != case['image_id']])}
        for label, case in cases.items()
    }
    distributions = {}
    for shard, groups in batches.items():
        counts = Counter(group['batch_size'] for group in groups)
        distributions[shard] = {
            'batches': len(groups),
            'batch_counts_by_size': {str(size): count for size, count in sorted(counts.items())},
            'image_counts_by_batch_size': {str(size): size * count for size, count in sorted(counts.items())},
        }
    all_counts = Counter(group['batch_size'] for groups in batches.values() for group in groups)
    guidance = [raw[late][image_id]['sar_ems_statistics'] for image_id in ids]
    validation = {
        'primary': str(metrics_path), 'paired_images': len(ids),
        'captions': sum(len(records) for records in raw.values()),
        'image_ids_sha256': hashlib.sha256(json.dumps(ids).encode('ascii')).hexdigest(),
        'image_ids_sha256_encoding': 'json.dumps(ids) with default separators; matches the shared evaluator.',
        'selection_ids_sha256': selection_ids_sha256,
        'selection_ids_sha256_encoding': "json.dumps(ids, separators=(',', ':')); matches protocol.selected_sha256.",
        'same_ordered_batches_both_methods': True,
        'same_model_decoder_visual_count_batch_size': True,
        'batch_size_setting': 16, 'batch_size_semantics': 'Maximum size; same-span tail groups and recorded OOM splits may be smaller.',
        'batches_by_shard': batches, 'batch_distribution_by_shard': distributions,
        'batch_counts_by_size': {str(size): count for size, count in sorted(all_counts.items())},
        'image_counts_by_batch_size': {str(size): size * count for size, count in sorted(all_counts.items())},
        'generation_logs': log_summary(),
        'caps': {name: sum(row['methods'][name]['tokens'] >= 512 for row in rows) for name in NAMES},
        'n_common_uncapped': len(uncapped),
        'common_uncapped': paired_summary(uncapped, bootstrap=True),
        'n_actual_batch16': len(actual_batch16),
        'actual_batch16_selection': 'Images whose matched realized batch size is 16. Membership follows preprocessing span/prompt-length buckets and any recorded OOM splits, not scores; this is not a new random sample. The complete 200-image result remains primary.',
        'actual_batch16_image_ids': [row['image_id'] for row in actual_batch16],
        'actual_batch16': paired_summary(actual_batch16, bootstrap=True),
        'guidance': {
            'guided_images': sum(stats['guided_steps'] > 0 for stats in guidance),
            'guided_steps': sum(stats['guided_steps'] for stats in guidance),
            'target_matches_generated': sum(stats['target_matches_generated'] for stats in guidance),
            'guided_attention_calls': sum(stats['guided_attention_calls'] for stats in guidance),
            'changed_token_sequences': sum(raw[base][i]['response_token_ids'] != raw[late][i]['response_token_ids'] for i in ids),
        },
        'hallucinated_mention_changes': {
            'decreased_images': sum(case['late_minus_svar'] < 0 for case in changes),
            'increased_images': sum(case['late_minus_svar'] > 0 for case in changes),
            'unchanged_images': sum(case['late_minus_svar'] == 0 for case in changes),
        },
        'influence': influence,
    }
    validation_path = ROOT / 'validation.new200_b16.json'
    atomic_write_json(validation_path, validation)
    print(json.dumps({
        'metrics': str(metrics_path), 'validation': str(validation_path),
        'methods': {name: {key: primary['methods'][name][key] for key in (*METRICS, *COUNTS)} for name in NAMES},
        'paired': primary['methods'][late]['paired_against'][base],
    }, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_write_json(ROOT / 'evaluation_error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
