#!/usr/bin/env python3
"""Score fixed generation-order prefixes while paired single-image decoding runs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.qa_benchmark import atomic_write_json
from scripts.evaluate_generation_time_evidence_drift_llava import (
    _bootstrap_metrics, _score_caption, load_chair, summarize,
)

NAMES = ('svar', 'svar_sar_ems_attention')
METRICS = ('chair_s', 'chair_i', 'object_recall', 'object_f1')
MILESTONES = (100, 200, 300, 400, 500)


def complete_rows(path: Path) -> list[dict]:
    """Ignore only an unfinished trailing write; malformed complete rows fail."""
    if not path.exists():
        return []
    with path.open('rb') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            data = handle.read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    complete = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def read_captions(root: Path, ids: list[int]) -> dict:
    result = {name: {} for name in NAMES}
    for name in NAMES:
        for shard in range(2):
            allowed = set(ids[shard::2])
            for row in complete_rows(root / 'coco500' / f'{name}.shard{shard}.jsonl'):
                image_id = int(row['image_id'])
                if image_id not in allowed or image_id in result[name]:
                    raise ValueError(f'{name}/{shard}: duplicate or wrong-shard image {image_id}')
                if row['variant'] != name or row['model'] != 'qwen3_vl_8b':
                    raise ValueError(f'Unexpected variant/model for {image_id}')
                if row['decoder'] != 'native_qwen3_generate' or row.get('batch_size', 1) != 1:
                    raise ValueError(f'Expected native single-image decoding for {image_id}')
                if not row['response_token_ids']:
                    raise ValueError(f'Empty generation for {image_id}')
                if name == NAMES[1]:
                    stats = row['sar_ems_statistics']
                    if stats['guided_attention_calls'] != 2 * stats['guided_steps']:
                        raise ValueError(f'Guidance count mismatch for {image_id}')
                result[name][image_id] = row
    for image_id in set(result[NAMES[0]]) & set(result[NAMES[1]]):
        if result[NAMES[0]][image_id]['visual_token_count'] != result[NAMES[1]][image_id]['visual_token_count']:
            raise ValueError(f'Paired visual token mismatch for {image_id}')
    return result


def report(ids: list[int], raw: dict, scored: dict, cache_path: Path) -> dict:
    rows = {name: [scored[f'{name}/{image_id}']['score'] for image_id in ids] for name in NAMES}
    methods = {name: summarize(rows[name]) for name in NAMES}
    draws = np.random.default_rng(43).integers(0, len(ids), size=(5000, len(ids)))
    boot = {name: _bootstrap_metrics(rows[name], draws) for name in NAMES}
    base, late = NAMES
    pairs = list(zip(rows[base], rows[late], strict=True))
    comparison = {
        'delta': {key: methods[late][key] - methods[base][key] for key in METRICS},
        'delta_ci95': {key: np.quantile(boot[late][key] - boot[base][key], [.025, .975]).tolist() for key in METRICS},
        'fixed_hallucinated_images': sum(a['chair_s'] > b['chair_s'] for a, b in pairs),
        'introduced_hallucinated_images': sum(a['chair_s'] < b['chair_s'] for a, b in pairs),
        'changed_token_sequences': sum(raw[base][i]['response_token_ids'] != raw[late][i]['response_token_ids'] for i in ids),
    }
    return {
        'images': len(ids), 'image_ids': ids, 'image_ids_sha256': fingerprint(ids),
        'selection': 'Prefix of the frozen generation manifest (ascending image ID), not completion order or original random-draw order.',
        'decoder': 'native_qwen3_generate', 'batch_size': 1,
        'bootstrap_draws': 5000, 'bootstrap_seed': 43,
        'object_metric_aggregation': 'CHAIRi pools mention counts; recall and F1 average per-image scores.',
        'chair_cache': str(cache_path), 'methods': methods,
        'late_paired_against_svar': comparison,
        'caps': {name: sum(row['tokens'] >= 512 for row in rows[name]) for name in NAMES},
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--coco-root', type=Path, required=True)
    parser.add_argument('--ground-truth-cache-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--driver-pid', type=int)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    root = args.output_root
    folder = root / 'milestones'
    folder.mkdir(exist_ok=True)
    ids = list(map(int, json.loads((root / 'cohort.json').read_text())['coco500']))
    if len(ids) != 500 or len(set(ids)) != 500 or ids != sorted(ids):
        raise ValueError('Expected the frozen sorted 500-image manifest')
    chair, cache_path = load_chair(args.coco_root, args.ground_truth_cache_root, ids, None)
    provenance = {
        'cohort': fingerprint(ids),
        'chair_source': hashlib.sha256((REPO / 'coco-labeling/coco_chair.py').read_bytes()).hexdigest(),
        'scorer_source': hashlib.sha256((REPO / 'scripts/evaluate_generation_time_evidence_drift_llava.py').read_bytes()).hexdigest(),
        'ground_truth': hashlib.sha256(cache_path.read_bytes()).hexdigest(),
    }
    score_path = folder / 'score_cache.json'
    saved = json.loads(score_path.read_text()) if score_path.exists() else {'provenance': provenance, 'rows': {}}
    if saved['provenance'] != provenance:
        raise ValueError('Score cache provenance changed')
    scored = saved['rows']
    lexical_cache = {}
    try:
        while True:
            raw = read_captions(root, ids)
            common = set(raw[NAMES[0]]) & set(raw[NAMES[1]])
            prefix = next((index for index, image_id in enumerate(ids) if image_id not in common), len(ids))
            changed = False
            for name in NAMES:
                for image_id, row in raw[name].items():
                    key, digest = f'{name}/{image_id}', fingerprint(row)
                    if key in scored:
                        if scored[key]['source_sha256'] != digest:
                            raise ValueError(f'Previously scored caption changed: {key}')
                    else:
                        scored[key] = {'source_sha256': digest, 'score': _score_caption(chair, row, lexical_cache)}
                        changed = True
            if changed:
                atomic_write_json(score_path, saved)
            ready = []
            for count in MILESTONES:
                if prefix < count:
                    break
                path = folder / f'metrics.generation_prefix{count}.json'
                if not path.exists():
                    result = report(ids[:count], raw, scored, cache_path)
                    atomic_write_json(path, result)
                    print(json.dumps({'event': 'milestone_ready', 'images': count, 'metrics': str(path)}, sort_keys=True), flush=True)
                ready.append(count)
            status = {
                'paired_images': len(common), 'fixed_prefix_complete': prefix,
                'captions': {name: len(raw[name]) for name in NAMES},
                'ready_milestones': ready, 'scored_captions': len(scored),
                'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
            atomic_write_json(folder / 'status.json', status)
            if ready == list(MILESTONES):
                print('ALL_MILESTONES_READY', flush=True)
                return
            if args.driver_pid:
                try:
                    os.kill(args.driver_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError('Generation driver exited before all milestones were available')
            time.sleep(args.poll_seconds)
    except Exception as exc:
        atomic_write_json(folder / 'error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
