"""Validate and score both matched next-layer feedback arms on the frozen 200 images."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np

REPO = Path('/root/rivermind-data/ic-ac')
ROOT = REPO / 'outputs/generation_time_evidence_drift_qwen3_next200_t256_b16'
FOLDER = ROOT / 'coco200'
COCO = REPO.parent / 'dataset/coco'
NAMES = ('sadi', 'sadi_sar_ems_next_19_20')
METRICS = ('chair_s', 'chair_i', 'object_recall', 'object_f1')
COUNTS = ('hallucinated_images', 'hallucinated_mentions', 'real_mentions', 'object_mentions')
TIMING = ('Per-method generate time includes its own prefill and controller cache construction; '
          'excludes model loading, CPU image preprocessing and verification. Sum batch_wall_seconds '
          'once per batch; totals across both GPUs are accumulated GPU-worker elapsed time, not experiment wall time.')
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from data.qa_benchmark import atomic_write_json
from scripts.evaluate_generation_time_evidence_drift_llava import _bootstrap_metrics, evaluate, summarize
from scripts import evaluate_generation_time_evidence_drift_llava as shared_evaluator


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_statistics(stats, name, token_count):
    if name == NAMES[0]:
        return
    if not isinstance(stats, dict):
        raise ValueError('Missing layer-feedback statistics')
    for key in ('guided_steps', 'guided_attention_calls', 'target_matches_generated',
                'target_match_events', 'local_recompute_events'):
        if type(stats.get(key)) is not int or stats[key] < 0:
            raise ValueError(f'Invalid statistics counter: {key}')
    steps, calls = stats['guided_steps'], stats['guided_attention_calls']
    if not (0 <= stats['target_matches_generated'] <= steps <= token_count
            and steps <= calls <= 2 * steps and stats['target_match_events'] <= calls):
        raise ValueError('Inconsistent step/event statistics')
    if stats.get('diagnostic_force_guidance'):
        raise ValueError('Diagnostic forced guidance must never enter scored generation')
    expected_replays = 0
    if stats['local_recompute_events'] != expected_replays:
        raise ValueError('Unexpected local recomputation count')
    by_layer = stats['by_layer']
    if set(by_layer) != {'19', '20'}:
        raise ValueError('Expected statistics for target layers 19 and 20')
    for target in (19, 20):
        entry = by_layer[str(target)]
        source = target - 1
        if entry['source_layer'] != source or entry['target_layer'] != target:
            raise ValueError('Wrong source-to-target layer mapping')
        if not 0 <= entry['target_matches_generated'] <= entry['guided_steps'] <= token_count:
            raise ValueError('Invalid layer-specific step counts')
    if sum(entry['guided_steps'] for entry in by_layer.values()) != calls:
        raise ValueError('Layer event totals do not match attention call count')
    if sum(entry['target_matches_generated'] for entry in by_layer.values()) != stats['target_match_events']:
        raise ValueError('Layer target-match totals do not match event count')


def load_and_validate(ids):
    raw, plans, times = ({name: {} for name in NAMES} for _ in range(3))
    for name in NAMES:
        for shard in range(2):
            path = FOLDER / f'{name}.shard{shard}.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if len(rows) != 100 or {int(row['image_id']) for row in rows} != set(ids[shard::2]):
                raise ValueError(f'{path}: expected 100 unique images from the frozen shard')
            for row in rows:
                image_id = int(row['image_id'])
                if image_id in raw[name] or row['variant'] != name or row['model'] != 'qwen3_vl_8b':
                    raise ValueError(f'{path}: duplicate image or wrong model/variant')
                if row['decoder'] != 'native_qwen3_generate_same_span_batch':
                    raise ValueError(f'{path}: wrong decoder')
                if type(row['batch_size']) is not int or not 1 <= row['batch_size'] <= 16:
                    raise ValueError(f'{path}: invalid batch size')
                tokens = row['response_token_ids']
                if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256:
                    raise ValueError(f'{path}: empty or over-limit caption')
                if not isinstance(row['generated_text'], str):
                    raise ValueError(f'{path}: invalid generated text')
                if type(row['visual_token_count']) is not int or row['visual_token_count'] <= 0:
                    raise ValueError(f'{path}: invalid visual token count')
                for field in ('seconds', 'batch_wall_seconds'):
                    if not math.isfinite(float(row[field])) or float(row[field]) < 0:
                        raise ValueError(f'{path}: invalid timing')
                validate_statistics(row.get('sar_ems_statistics'), name, len(tokens))
                raw[name][image_id] = row
            batches, seconds, offset = [], 0.0, 0
            while offset < len(rows):
                first = rows[offset]
                size = first['batch_size']
                group = rows[offset:offset + size]
                if len(group) != size:
                    raise ValueError(f'{path}: incomplete final batch')
                for row in group:
                    if any(row[key] != first[key] for key in ('batch_size', 'visual_token_count', 'batch_wall_seconds')):
                        raise ValueError(f'{path}: inconsistent metadata within batch')
                    if not math.isclose(row['seconds'] * size, row['batch_wall_seconds'], rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f'{path}: invalid per-image share of batch timing')
                batches.append({'image_ids': [int(row['image_id']) for row in group],
                                'batch_size': size, 'visual_token_count': first['visual_token_count']})
                seconds += first['batch_wall_seconds']
                offset += size
            plans[name][str(shard)] = batches
            times[name][str(shard)] = seconds
    for name in NAMES[1:]:
        if plans[name] != plans[NAMES[0]]:
            raise ValueError(f'{name}: ordered image batches differ from SADI')
    return raw, plans[NAMES[0]], times


def paired_summary(rows, *, bootstrap=False):
    if not rows:
        return None
    scores = {name: [row['methods'][name] for row in rows] for name in NAMES}
    fields = ('images', *METRICS, *COUNTS, 'tokens_per_image', 'words_per_image')
    methods = {name: {key: value for key, value in summarize(scores[name]).items() if key in fields} for name in NAMES}
    boot = {}
    if bootstrap:
        draws = np.random.default_rng(43).integers(0, len(rows), size=(5000, len(rows)))
        boot = {name: _bootstrap_metrics(scores[name], draws) for name in NAMES}
    comparisons = {}
    for name, reference in ((NAMES[1], NAMES[0]),):
        pairs = list(zip(scores[reference], scores[name], strict=True))
        entry = {'variant': name, 'reference': reference,
                 'delta': {key: methods[name][key] - methods[reference][key] for key in METRICS},
                 'fixed_hallucinated_images': sum(a['chair_s'] > b['chair_s'] for a, b in pairs),
                 'introduced_hallucinated_images': sum(a['chair_s'] < b['chair_s'] for a, b in pairs)}
        if bootstrap:
            entry['delta_ci95'] = {key: np.quantile(boot[name][key] - boot[reference][key], [.025, .975]).tolist() for key in METRICS}
        comparisons[f'{name}_vs_{reference}'] = entry
    return {'methods': methods, 'comparisons': comparisons}


def log_summary():
    result = {}
    for shard in range(2):
        path = ROOT / 'logs' / f'gpu{shard}.log'
        if not path.is_file():
            raise FileNotFoundError(f'Missing generation log: {path}')
        content = path.read_text(encoding='utf-8', errors='replace')
        result[str(shard)] = {'path': str(path),
                             'oom_split_count': len(re.findall(rf'^\[oom:{shard}\] splitting\b', content, re.MULTILINE)),
                             'traceback_count': content.count('Traceback (most recent call last):'),
                             'cuda_oom_error_lines': sum('OutOfMemoryError' in line or 'CUDA out of memory' in line for line in content.splitlines())}
    return result


def aggregate_guidance(raw, ids):
    output = {}
    counters = ('guided_steps', 'guided_attention_calls', 'target_matches_generated',
                'target_match_events', 'local_recompute_events')
    for name in NAMES[1:]:
        stats = [raw[name][i]['sar_ems_statistics'] for i in ids]
        total = {key: sum(s[key] for s in stats) for key in counters}
        total['guided_images'] = sum(s['guided_steps'] > 0 for s in stats)
        total['changed_token_sequences_vs_sadi'] = sum(raw[name][i]['response_token_ids'] != raw[NAMES[0]][i]['response_token_ids'] for i in ids)
        total['unique_step_target_match_rate'] = total['target_matches_generated'] / total['guided_steps'] if total['guided_steps'] else None
        total['layer_event_target_match_rate'] = total['target_match_events'] / total['guided_attention_calls'] if total['guided_attention_calls'] else None
        total['by_layer'] = {}
        for layer in ('19', '20'):
            entries = [s['by_layer'][layer] for s in stats]
            count = sum(e['guided_steps'] for e in entries)
            entry = {'source_layer': entries[0]['source_layer'], 'target_layer': int(layer),
                     'guided_steps': count, 'target_matches_generated': sum(e['target_matches_generated'] for e in entries)}
            for field in ('mean_budget_max', 'mean_drift', 'mean_gain'):
                entry[field] = sum(e['guided_steps'] * e[field] for e in entries if e['guided_steps']) / count if count else None
            total['by_layer'][layer] = entry
        output[name] = total
    return output


def repetition_and_denominator(rows, raw):
    """Describe repetition and output-dependent denominator sensitivity, never replace primary scores."""
    base, enhanced = NAMES
    repetition = {}
    for name in NAMES:
        cases = []
        for row in rows:
            score = row['methods'][name]
            counts = score['canonical_object_counts']
            hallucinated = Counter(item[1] for item in score['hallucinated_objects'])
            cases.append({'image_id': row['image_id'], 'object_mentions': score['object_mentions'],
                          'distinct_object_categories': len(counts),
                          'repeated_object_mentions': sum(max(0, count - 1) for count in counts.values()),
                          'repeated_hallucinated_mentions': sum(max(0, count - 1) for count in hallucinated.values()),
                          'maximum_category_count': max(counts.values(), default=0),
                          'object_counts': counts, 'hallucinated_counts': dict(hallucinated),
                          'tokens': score['tokens']})
        top = sorted(cases, key=lambda x: (-x['repeated_object_mentions'], x['image_id']))[:10]
        repetition[name] = {
            'images_with_repeated_object_category': sum(c['repeated_object_mentions'] > 0 for c in cases),
            'object_mentions': sum(c['object_mentions'] for c in cases),
            'distinct_object_category_image_pairs': sum(c['distinct_object_categories'] for c in cases),
            'repeated_object_mentions': sum(c['repeated_object_mentions'] for c in cases),
            'repeated_hallucinated_mentions': sum(c['repeated_hallucinated_mentions'] for c in cases),
            'top10_repetition_images': top,
        }
    growth = []
    for row in rows:
        a, b = (row['methods'][name] for name in NAMES)
        growth.append({'image_id': row['image_id'],
                       'delta_object_mentions': b['object_mentions'] - a['object_mentions'],
                       'delta_hallucinated_mentions': b['hallucinated_mentions'] - a['hallucinated_mentions'],
                       'by_method': {name: {key: row['methods'][name][key] for key in
                                           ('object_mentions', 'hallucinated_mentions', 'tokens', 'canonical_object_counts')}
                                     for name in NAMES}})
    growth.sort(key=lambda x: (-x['delta_object_mentions'], x['image_id']))
    positive = [item for item in growth if item['delta_object_mentions'] > 0]
    net = sum(item['delta_object_mentions'] for item in growth)
    gross = sum(item['delta_object_mentions'] for item in positive)
    excluded = {}
    for top_k in (1, 3, 5):
        selected = positive[:top_k]
        removed = {item['image_id'] for item in selected}
        contribution = sum(item['delta_object_mentions'] for item in selected)
        excluded[str(top_k)] = {
            'removed_image_ids': sorted(removed), 'removed_images': len(removed),
            'removed_denominator_increase': contribution,
            'share_of_gross_positive_denominator_increase': contribution / gross if gross else None,
            'share_of_net_denominator_increase': contribution / net if net > 0 else None,
            'remaining': paired_summary([row for row in rows if row['image_id'] not in removed]),
        }
    return {
        'definition': 'Repeated mentions count each canonical COCO object category after its first mention within a caption; repetition can be legitimate and is not automatically a generation loop.',
        'by_method': repetition,
        'denominator_growth': {
            'variant': enhanced, 'reference': base, 'net_object_mention_increase': net,
            'gross_positive_object_mention_increase': gross,
            'net_hallucinated_mention_change': sum(item['delta_hallucinated_mentions'] for item in growth),
            'top10_increases': growth[:10], 'remove_top_positive_growth': excluded,
            'interpretation': 'Output-conditioned post-hoc sensitivity only. Fractions of the net increase can exceed one when other images decrease; all 200 images remain primary.',
        },
    }


def main():
    ids = list(map(int, read_json(ROOT / 'cohort.json')['coco200']))
    if len(ids) != 200 or len(set(ids)) != 200 or ids != sorted(ids):
        raise ValueError('Expected 200 sorted unique frozen images')
    protocol = read_json(ROOT / 'protocol.json')
    ids_sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode('ascii')).hexdigest()
    if protocol['selected_sha256'] != ids_sha or tuple(protocol['variants']) != NAMES:
        raise ValueError('Cohort or methods differ from frozen protocol')
    if protocol['max_new_tokens'] != 256 or protocol['images_per_shard'] != 100:
        raise ValueError('Wrong token limit or shard size in protocol')
    for rel, expected in protocol['source_sha256'].items():
        if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed after protocol freeze: {rel}')
    reuse = protocol['verification_reuse']
    for field, hash_field in (('path', 'sha256'), ('protocol_path', 'protocol_sha256')):
        if hashlib.sha256(Path(reuse[field]).read_bytes()).hexdigest() != reuse[hash_field]:
            raise ValueError('Reused verification provenance changed after freeze')
    raw, batches, times = load_and_validate(ids)
    original_score = shared_evaluator._score_caption
    def score_with_objects(chair, row, cache):
        result = original_score(chair, row, cache)
        # Reuse the CHAIR result just computed by the shared scorer, rather than
        # tokenizing/scoring the caption a second time for repetition diagnostics.
        scored = cache[(int(row['image_id']), row['generated_text'])]
        result['canonical_object_counts'] = dict(Counter(scored['mscoco_generated_words']))
        if sum(result['canonical_object_counts'].values()) != result['object_mentions']:
            raise ValueError('Canonical object counts disagree with CHAIR denominator')
        return result
    with patch.object(shared_evaluator, '_score_caption', score_with_objects):
        metrics_path = evaluate(ROOT, 'coco200', list(NAMES), COCO, 200, 5000, None)
    primary = read_json(metrics_path)
    primary['timing'] = TIMING
    atomic_write_json(metrics_path, primary)
    rows = [json.loads(line) for line in Path(primary['per_image']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['image_id'] for row in rows] != ids:
        raise ValueError('Scored image IDs/order differ from frozen cohort')
    uncapped = [row for row in rows if all(row['methods'][name]['tokens'] < 256 for name in NAMES)]
    batch16 = [row for row in rows if raw[NAMES[0]][row['image_id']]['batch_size'] == 16]
    influence = {}
    for name in NAMES[1:]:
        changes = [{'image_id': row['image_id'], 'delta_hallucinated_mentions': row['methods'][name]['hallucinated_mentions'] - row['methods'][NAMES[0]]['hallucinated_mentions']} for row in rows]
        cases = {'largest_reduction': min(changes, key=lambda x: (x['delta_hallucinated_mentions'], x['image_id'])),
                 'largest_increase': max(changes, key=lambda x: (x['delta_hallucinated_mentions'], -x['image_id']))}
        influence[name] = {'mention_changes': {'decreased_images': sum(x['delta_hallucinated_mentions'] < 0 for x in changes),
                                               'increased_images': sum(x['delta_hallucinated_mentions'] > 0 for x in changes),
                                               'unchanged_images': sum(x['delta_hallucinated_mentions'] == 0 for x in changes)},
                           'cases': {label: {'case': case, 'without_image': paired_summary([row for row in rows if row['image_id'] != case['image_id']])} for label, case in cases.items()}}
    counts = Counter(batch['batch_size'] for groups in batches.values() for batch in groups)
    timing = {'definition': TIMING, 'generation_seconds_by_shard': times,
              'generation_seconds_sum': {name: sum(times[name].values()) for name in NAMES},
              'generation_seconds_per_image': {name: sum(times[name].values()) / 200 for name in NAMES}}
    if (ROOT / 'run_status.json').is_file():
        status = read_json(ROOT / 'run_status.json')
        timing['driver_status_at_evaluation'] = status
        if status.get('started_utc') and status.get('generation_finished_utc'):
            timing['generation_wall_seconds_including_load'] = (datetime.fromisoformat(status['generation_finished_utc']) - datetime.fromisoformat(status['started_utc'])).total_seconds()
    validation = {
        'primary': str(metrics_path), 'paired_images': 200, 'captions': sum(len(r) for r in raw.values()),
        'selection_ids_sha256': ids_sha, 'same_ordered_batches_both_methods': True,
        'source_sha256_verified': True, 'batch_size_setting': 16,
        'verification_reuse': reuse,
        'batch_size_semantics': 'Maximum; same-span tail groups and paired OOM splits may be smaller.',
        'batches_by_shard': batches,
        'batch_counts_by_size': {str(k): v for k, v in sorted(counts.items())},
        'image_counts_by_batch_size': {str(k): k * v for k, v in sorted(counts.items())},
        'generation_logs': log_summary(), 'timing': timing,
        'caps': {name: sum(row['methods'][name]['tokens'] >= 256 for row in rows) for name in NAMES},
        'n_common_uncapped': len(uncapped),
        'common_uncapped_selection': 'Outcome-conditioned subset, secondary only; all 200 images remain primary.',
        'common_uncapped': paired_summary(uncapped, bootstrap=True),
        'n_actual_batch16': len(batch16),
        'actual_batch16_selection': 'Preprocessing/batch-plan-defined subset, declared before scoring; not a new random sample.',
        'actual_batch16_image_ids': [row['image_id'] for row in batch16],
        'actual_batch16': paired_summary(batch16, bootstrap=True),
        'guidance': aggregate_guidance(raw, ids), 'influence': influence,
        'repetition_and_denominator': repetition_and_denominator(rows, raw),
    }
    validation_path = ROOT / 'validation.next200_t256_b16.json'
    atomic_write_json(validation_path, validation)
    print(json.dumps({'metrics': str(metrics_path), 'validation': str(validation_path),
                      'methods': {name: {key: primary['methods'][name][key] for key in (*METRICS, *COUNTS)} for name in NAMES},
                      'paired': {name: primary['methods'][name]['paired_against'] for name in NAMES[1:]}}, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_write_json(ROOT / 'evaluation_error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
