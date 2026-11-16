#!/usr/bin/env python3
"""CPU-only, isolated sentence-tokenized CHAIR rescore of the frozen Qwen3 next200 run."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / 'outputs/generation_time_evidence_drift_qwen3_next200_t256_b16'
OUTPUT = ORIGINAL / 'sentence_tokenized_evaluation_v1'
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

from data.qa_benchmark import atomic_write_json
from scripts import evaluate_generation_time_evidence_drift_llava as shared
from scripts.experiment_snapshots.qwen3_next200_t256_b16 import evaluate as snapshot


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class SentenceTreebankSpans:
    """Keep original character coordinates while applying Treebank within each sentence."""
    def __init__(self, sentence_tokenizer, word_tokenizer):
        self.sentences = sentence_tokenizer
        self.words = word_tokenizer

    def span_tokenize(self, text):
        previous_end = 0
        for sentence_start, sentence_end in self.sentences.span_tokenize(text):
            for start, end in self.words.span_tokenize(text[sentence_start:sentence_end]):
                absolute = (sentence_start + start, sentence_start + end)
                if not previous_end <= absolute[0] < absolute[1] <= len(text):
                    raise ValueError('Invalid or unordered absolute token span')
                previous_end = absolute[1]
                yield absolute


def load_original_chair_module():
    if 'coco_chair' not in sys.modules:
        source = REPO / 'coco-labeling/coco_chair.py'
        spec = importlib.util.spec_from_file_location('coco_chair', source)
        if spec is None or spec.loader is None:
            raise ImportError(source)
        module = importlib.util.module_from_spec(spec)
        sys.modules['coco_chair'] = module
        spec.loader.exec_module(module)
    return sys.modules['coco_chair']


def make_sentence_chair(base, punkt):
    class SentenceChair(base):
        def _caption_to_units(self, caption):
            # This override is already active during base.__init__ ->
            # get_annotations_from_captions, so GT and predictions share it.
            if not isinstance(self._tokenizer, SentenceTreebankSpans):
                self._tokenizer = SentenceTreebankSpans(punkt, self._tokenizer)
            return super()._caption_to_units(caption)
    return SentenceChair


def microtests(module, sentence_class, punkt):
    module.ensure_nltk_data(download=False)
    def bare(cls):
        obj = cls.__new__(cls)
        obj._tokenizer = TreebankWordTokenizer()
        obj._lemmatizer = WordNetLemmatizer()
        obj.double_word_dict = obj._build_double_word_dict()
        return obj
    original, corrected = bare(module.CHAIR), bare(sentence_class)
    multi = 'There are two chairs. A person stands near a table.'
    old = original._caption_to_units(multi)
    new = corrected._caption_to_units(multi)
    if 'chair' in [unit['word'] for unit in old] or 'chair' not in [unit['word'] for unit in new]:
        raise AssertionError('The demonstrated intermediate-sentence chairs. defect was not repaired')
    single_sentences = ('Two chairs beside a table.', 'A traffic light stands beside a fire hydrant.', '')
    for text in single_sentences:
        if original._caption_to_units(text) != corrected._caption_to_units(text):
            raise AssertionError(f'Single-sentence regression: {text!r}')
    samples = (multi, 'İ see chairs. “A dog sits.”', 'Mr. Smith has chairs. A dog sits.',
               'A traffic light. A fire hydrant.', 'Chairs. Chairs.',
               '  Two chairs.\n\nA dog sits.  ')
    checks = []
    for text in samples:
        spans = list(SentenceTreebankSpans(punkt, TreebankWordTokenizer()).span_tokenize(text))
        for unit in corrected._caption_to_units(text):
            if text[unit['char_start']:unit['char_end']] != unit['surface']:
                raise AssertionError('Object/word surface does not match absolute offsets')
        checks.append({'caption': text, 'spans': spans, 'tokens': [text[a:b] for a, b in spans]})
    return {'passed': True, 'original_multi_words': [u['word'] for u in old],
            'corrected_multi_words': [u['word'] for u in new],
            'single_sentence_parity_cases': list(single_sentences), 'absolute_span_cases': checks}


def link_readonly_input(source, destination):
    if destination.is_symlink() or destination.exists():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise FileExistsError(f'Unexpected existing input: {destination}')
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main():
    if read_json(ORIGINAL / 'run_status.json')['status'] != 'complete':
        raise RuntimeError('Wait for the original frozen generation and evaluation to complete')
    names = snapshot.NAMES
    raw_paths = [ORIGINAL / 'coco200' / f'{name}.shard{shard}.jsonl' for name in names for shard in range(2)]
    original_validation = read_json(ORIGINAL / 'validation.next200_t256_b16.json')
    original_metrics_path = Path(original_validation['primary'])
    original_metrics = read_json(original_metrics_path)
    original_per_image = Path(original_metrics['per_image'])
    original_cache = Path(original_metrics['chair_cache'])
    protocol = read_json(ORIGINAL / 'protocol.json')
    frozen_sources = [REPO / rel for rel in protocol['source_sha256']]
    protected = [*raw_paths, original_metrics_path, original_per_image, original_cache,
                 ORIGINAL / 'cohort.json', ORIGINAL / 'protocol.json', ORIGINAL / 'validation.next200_t256_b16.json',
                 *frozen_sources]
    hashes_before = {str(path): digest(path) for path in protected}
    for rel, expected in protocol['source_sha256'].items():
        if hashes_before[str(REPO / rel)] != expected:
            raise ValueError(f'Original source drift: {rel}')

    punkt_pointer = nltk.data.find('tokenizers/punkt/english.pickle')
    with punkt_pointer.open() as resource:
        punkt_sha = hashlib.sha256(resource.read()).hexdigest()
    punkt = nltk.data.load('tokenizers/punkt/english.pickle')
    module = load_original_chair_module()
    sentence_class = make_sentence_chair(module.CocoChairEvaluator, punkt)
    checks = microtests(module, sentence_class, punkt)
    annotation_paths = [snapshot.COCO / 'annotations' / name for name in
                        ('instances_val2014.json', 'captions_val2014.json')]
    parser = {'adapter_script': str(Path(__file__).resolve()), 'adapter_sha256': digest(__file__),
              'original_chair_sha256': digest(REPO / 'coco-labeling/coco_chair.py'),
              'punkt_resource': str(punkt_pointer), 'punkt_sha256': punkt_sha, 'nltk_version': nltk.__version__,
              'change': 'Punkt sentence spans followed by Treebank word spans with absolute offsets; original whole-caption POS/lemma/synonym rules retained for GT and predictions.'}
    provenance = {'version': 'sentence_tokenized_evaluation_v1', 'original_root': str(ORIGINAL),
                  'parser': parser, 'protected_original_sha256': hashes_before,
                  'annotations_sha256': {str(path): digest(path) for path in annotation_paths},
                  'old_gt_cache_reused': False, 'generated_captions_reused': True}
    if OUTPUT.exists():
        if not (OUTPUT / 'provenance.json').is_file() or read_json(OUTPUT / 'provenance.json') != provenance:
            raise FileExistsError('Independent evaluation directory has incompatible or incomplete provenance')
    else:
        OUTPUT.mkdir()
        atomic_write_json(OUTPUT / 'provenance.json', provenance)
    for name in ('cohort.json', 'protocol.json', 'run_status.json'):
        path, content = OUTPUT / name, (ORIGINAL / name).read_bytes()
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'Independent input copy changed: {path}')
        if not path.exists():
            path.write_bytes(content)
    (OUTPUT / 'coco200').mkdir(exist_ok=True)
    for path in raw_paths:
        link_readonly_input(path, OUTPUT / 'coco200' / path.name)
    link_readonly_input(ORIGINAL / 'logs', OUTPUT / 'logs')
    atomic_write_json(OUTPUT / 'microtests.json', checks)

    original_load = shared.load_chair
    original_score = shared._score_caption
    example_scores = {}
    new_cache_paths = []
    def load_corrected(coco_root, output_root, ids, full_cache_path):
        if Path(output_root).resolve() != OUTPUT.resolve() or full_cache_path is not None:
            raise ValueError('Corrected evaluator must build or reuse only its isolated GT cache')
        for cached in OUTPUT.glob('chair_ground_truth.*.json'):
            if read_json(cached).get('sentence_parser') != parser:
                raise ValueError('Corrected GT cache lacks matching adapter/Punkt provenance')
        evaluator, cache_path = original_load(coco_root, output_root, ids, None)
        payload = read_json(cache_path)
        payload['sentence_parser'] = parser
        atomic_write_json(cache_path, payload)
        new_cache_paths.append(cache_path)
        return evaluator, cache_path
    def capture_example(chair, row, cache):
        result = original_score(chair, row, cache)
        if int(row['image_id']) == 187169:
            scored = cache[(int(row['image_id']), row['generated_text'])]
            example_scores[row['variant']] = {
                'caption': row['generated_text'],
                'canonical_object_counts': dict(Counter(scored['mscoco_generated_words'])),
                'chair_mentions': [m for m in scored['object_mentions'] if m['canonical_object'] == 'chair'],
                'ground_truth_objects': scored['mscoco_gt_words'],
                'hallucinated_mentions': len(scored['mscoco_hallucinated_words']),
            }
        return result

    try:
        with patch.object(module, 'CocoChairEvaluator', sentence_class), \
             patch.object(shared, 'load_chair', load_corrected), \
             patch.object(shared, '_score_caption', capture_example), \
             patch.object(snapshot, 'ROOT', OUTPUT), \
             patch.object(snapshot, 'FOLDER', OUTPUT / 'coco200'):
            snapshot.main()
        validation = read_json(OUTPUT / 'validation.next200_t256_b16.json')
        corrected_metrics = read_json(validation['primary'])
        old_gt, new_gt = read_json(original_cache)['objects'], read_json(new_cache_paths[0])['objects']
        if set(old_gt) != set(new_gt):
            raise ValueError('Old/new ground-truth image scopes differ')
        differences = [{'image_id': int(i), 'added': sorted(set(new_gt[i]) - set(old_gt[i])),
                        'removed': sorted(set(old_gt[i]) - set(new_gt[i]))}
                       for i in old_gt if set(old_gt[i]) != set(new_gt[i])]
        old_rows = [json.loads(line) for line in original_per_image.read_text(encoding='utf-8').splitlines() if line.strip()]
        old_example = next(row for row in old_rows if row['image_id'] == 187169)
        example = {name: {'old': {key: old_example['methods'][name].get(key) for key in
                                  ('canonical_object_counts', 'hallucinated_mentions', 'object_mentions')},
                          'sentence_tokenized': example_scores[name]} for name in names}
        audit = {'provenance': str(OUTPUT / 'provenance.json'), 'microtests_passed': True,
                 'original_metrics': str(original_metrics_path), 'corrected_metrics': validation['primary'],
                 'gt_changed_images': len(differences), 'gt_differences': differences,
                 'example_187169': example,
                 'methods': {name: {'original': {key: original_metrics['methods'][name][key] for key in snapshot.METRICS},
                                    'sentence_tokenized': {key: corrected_metrics['methods'][name][key] for key in snapshot.METRICS}}
                             for name in names},
                 'new_gt_cache': str(new_cache_paths[0]), 'new_gt_cache_sha256': digest(new_cache_paths[0])}
    finally:
        hashes_after = {str(path): digest(path) for path in protected}
        if hashes_before != hashes_after:
            raise RuntimeError('An original frozen file changed during independent CPU scoring')
    audit['protected_original_files_unchanged'] = True
    audit['raw_sha256_before'] = {str(path): hashes_before[str(path)] for path in raw_paths}
    audit['raw_sha256_after'] = {str(path): hashes_after[str(path)] for path in raw_paths}
    atomic_write_json(OUTPUT / 'tokenizer_fix_audit.json', audit)
    print(json.dumps({'output': str(OUTPUT), 'gt_changed_images': len(differences),
                      'methods': audit['methods'], 'original_files_unchanged': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
