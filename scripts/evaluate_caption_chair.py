#!/usr/bin/env python3
"""Explicitly select legacy or sentence-tokenized CHAIR for saved captions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
VERSIONS = {'legacy': 'legacy_v1', 'sentence': 'sentence_v1'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator', choices=tuple(VERSIONS), required=True,
                        help='legacy: original whole-caption Treebank; sentence: sentence-first Treebank')
    parser.add_argument('--generation-root', type=Path, required=True, help='Existing cohort.json and caption shards')
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--coco-root', type=Path, default=REPO.parent / 'dataset/coco')
    parser.add_argument('--evaluation-root', type=Path, help='Default: GENERATION_ROOT/evaluations/{evaluator}_v1')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--bootstrap-draws', type=int, default=5000)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.cohort):
        parser.error('cohort must be a simple manifest key')
    if len(set(args.variants)) != len(args.variants) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in args.variants):
        parser.error('variants must be distinct simple names')
    source = args.generation_root.resolve()
    version = VERSIONS[args.evaluator]
    output = (args.evaluation_root or source / 'evaluations' / version).resolve()
    manifest = read_json(source / 'cohort.json')
    ids = manifest.get(args.cohort)
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        parser.error('cohort must name a nonempty unique image list in cohort.json')
    if args.limit is not None and not 1 <= args.limit <= len(ids):
        parser.error('limit outside cohort range')
    if args.bootstrap_draws < 0:
        parser.error('bootstrap-draws must be nonnegative')
    if output == source or output in source.parents or output == (source / args.cohort).resolve():
        parser.error('evaluation-root must be separate from generation data')
    output_cohort = output / args.cohort
    if output_cohort.is_symlink() or output_cohort.resolve() == (source / args.cohort).resolve():
        parser.error('evaluation cohort directory must not link to the original cohort')
    raw_paths = []
    for variant in args.variants:
        paths = sorted((source / args.cohort).glob(f'{variant}.shard*.jsonl'))
        if not paths:
            parser.error(f'No saved caption shards for {variant}')
        raw_paths.extend(paths)
    os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
    from data.qa_benchmark import atomic_write_json
    from scripts import evaluate_generation_time_evidence_drift_llava as shared
    import nltk
    module_name = 'coco_chair'
    chair_source = REPO / 'coco-labeling/coco_chair.py'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, chair_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module = sys.modules[module_name]
    provenance = {'evaluator': args.evaluator, 'version': version,
                  'chair_source_sha256': sha(chair_source), 'shared_evaluator_sha256': sha(Path(shared.__file__)),
                  'nltk_version': nltk.__version__, 'ground_truth_and_predictions_share_parser': True}
    sentence_class = None
    if args.evaluator == 'sentence':
        from scripts import evaluate_qwen3_next200_sentence_tokenized as adapter
        pointer = nltk.data.find('tokenizers/punkt/english.pickle')
        with pointer.open() as stream:
            resource_sha = hashlib.sha256(stream.read()).hexdigest()
        provenance.update(adapter_sha256=sha(Path(adapter.__file__)), punkt_sha256=resource_sha)
        sentence_class = adapter.make_sentence_chair(module.CocoChairEvaluator,
                                                    nltk.data.load('tokenizers/punkt/english.pickle'))
    provenance_path = output / 'evaluator.json'
    if output.exists():
        if not provenance_path.is_file() or read_json(provenance_path) != provenance:
            parser.error('evaluation-root has different/missing evaluator provenance; use a new directory')
    else:
        output.mkdir(parents=True)
        atomic_write_json(provenance_path, provenance)
    manifest_path = output / 'cohort.json'
    if manifest_path.exists() and manifest_path.read_bytes() != (source / 'cohort.json').read_bytes():
        parser.error('Saved cohort manifest differs; use a new evaluation-root')
    if not manifest_path.exists():
        shutil.copyfile(source / 'cohort.json', manifest_path)
    output_cohort.mkdir(exist_ok=True)
    original_hashes = {str(path): sha(path) for path in raw_paths}
    for path in raw_paths:
        target = output_cohort / path.name
        if target.exists() or target.is_symlink():
            if sha(target) != original_hashes[str(path)]:
                parser.error('Existing evaluation input differs from saved captions')
        else:
            try:
                target.symlink_to(path.resolve())
            except OSError:
                shutil.copyfile(path, target)
    original_load = shared.load_chair
    def versioned_load(coco_root, output_root, selected_ids, full_cache_path):
        if full_cache_path is not None or Path(output_root).resolve() != output:
            raise ValueError('Only version-isolated GT caches are supported')
        # The shared cache validates annotation fingerprints, image scope and
        # CHAIR source; this wrapper additionally validates parser identity.
        for cached in output.glob('chair_ground_truth.*.json'):
            if read_json(cached).get('evaluator_provenance') != provenance:
                raise ValueError('GT cache has missing/different evaluator provenance')
        evaluator, cache_path = original_load(coco_root, output_root, selected_ids, None)
        cache = read_json(cache_path)
        cache['evaluator_provenance'] = provenance
        atomic_write_json(cache_path, cache)
        return evaluator, cache_path
    with ExitStack() as stack:
        stack.enter_context(patch.object(shared, 'load_chair', versioned_load))
        if sentence_class is not None:
            stack.enter_context(patch.object(module, 'CocoChairEvaluator', sentence_class))
        metrics_path = shared.evaluate(output, args.cohort, args.variants, args.coco_root,
                                       args.limit, args.bootstrap_draws, None)
    if {str(path): sha(path) for path in raw_paths} != original_hashes:
        raise RuntimeError('Original caption shards changed during evaluation')
    metrics = read_json(metrics_path)
    metrics['evaluator'] = provenance
    metrics['evaluation_inputs'] = {'generation_root': str(source), 'caption_sha256': original_hashes,
                                    'cohort_sha256': sha(source / 'cohort.json'),
                                    'selector_sha256': sha(__file__), 'original_caption_files_unchanged': True}
    atomic_write_json(metrics_path, metrics)
    print(json.dumps({'evaluator': args.evaluator, 'version': version, 'metrics': str(metrics_path)}, indent=2))


if __name__ == '__main__':
    main()
