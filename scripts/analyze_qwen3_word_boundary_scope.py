"""Diagnose completed-word eligibility using existing SADI text, without a GPU.

The completed text is a post-hoc diagnostic. An online method would need a
verified read-only lookahead to obtain these word boundaries; this script is
not a mitigation result or an online detector evaluation.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from features.coco_object_history import ObjectClassHistory
from utils.token_alignment import build_response_token_offsets, token_indices_for_char_span


def read(path):
    return json.loads(path.read_bytes())


def summarize(decisions, labels):
    selected = [d for d in decisions if d['eligible']]
    known = [d for d in selected if (d['image_id'], d['step']) in labels]
    hall, real, true_flags, false_flags = 0, 0, 0, 0
    for d in known:
        y, p = labels[d['image_id'], d['step']]
        hall += int(y)
        real += int(not y)
        true_flags += int(y and p < .5)
        false_flags += int(not y and p < .5)
    return dict(candidates=len(decisions), eligible=len(selected), labeled_object_positions=len(known),
                non_object_start_positions=len(selected) - len(known), hall=hall, real=real,
                flagged_hall=true_flags, flagged_real=false_flags)


def analyze(output):
    from transformers import AutoTokenizer
    source = REPO / 'outputs/generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-probe-first-class-pilot'
    trace = REPO / 'outputs/qwen3_sadi_pre_attention_i_el_trace'
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    allowed = set(read(source / 'training_vocabulary.json')['token_ids'])
    lexical = set(read(source / 'object_lexicon.json')['token_ids'])
    token_class = parser.token_classes(tokenizer, lexical)
    with np.load(trace / 'predictions.npz', allow_pickle=False) as data:
        labels = {(int(i), int(s)): (bool(y), float(p)) for i, s, y, p in
                  zip(data['image_id'], data['step'], data['hallucinated'], data['p_real'], strict=True)}
    sources = [source / f'coco100/sadi.shard{shard}.jsonl' for shard in (0, 1)]
    records = [r for path in sources for r in map(json.loads, path.read_bytes().splitlines())]
    assert len(records) == 100
    completed, old = [], []
    for record in records:
        text, tokens = record['generated_text'], record['response_token_ids']
        offsets = build_response_token_offsets(tokenizer, tokens, text)
        words = list(re.finditer(r'[A-Za-z]+(?:-[A-Za-z]+)*', text))
        by_step = {}
        for i, word in enumerate(words):
            indices = token_indices_for_char_span(offsets, word.start(), word.end())
            if not indices:
                continue
            start = indices[0]
            if start in by_step:
                continue
            chosen = None
            for length in range(min(parser.maximum_words, len(words) - i), 0, -1):
                group = words[i:i + length]
                if any(not text[left.end():right.start()].isspace()
                       for left, right in zip(group, group[1:])):
                    continue
                key = parser.key(' '.join(w.group() for w in group))
                if key in parser.aliases:
                    end = group[-1].end()
                    span_tokens = token_indices_for_char_span(offsets, word.start(), end)
                    chosen = dict(canonical=parser.aliases[key], surface=text[word.start():end],
                                  last_token=span_tokens[-1], boundary_available=span_tokens[-1] + 1 < len(tokens))
                    break
            by_step[start] = chosen or dict(canonical=None, surface=word.group(), last_token=indices[-1],
                                           boundary_available=indices[-1] + 1 < len(tokens))
        seen = set()
        for step, target in enumerate(tokens):
            if target in allowed and target not in seen:
                prior = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False))
                base = dict(image_id=record['image_id'], step=step, token=target,
                            token_surface=tokenizer.decode([target]))
                old.append(dict(base, eligible=target in lexical and token_class.get(target) not in prior))
                word = by_step.get(step)
                canonical = word['canonical'] if word else None
                eligible = bool(canonical is not None and canonical not in prior and word['boundary_available'])
                completed.append(dict(base, eligible=eligible, canonical=canonical,
                    complete_surface=word['surface'] if word else None,
                    boundary_available=word['boundary_available'] if word else False,
                    minimum_lookahead_steps=word['last_token'] + 1 - step if word else None))
            seen.add(target)
    old_summary, new_summary = summarize(old, labels), summarize(completed, labels)
    assert len(old) == len(completed) == 441
    assert old_summary['eligible'] == 209 and old_summary['flagged_hall'] == 11 and old_summary['flagged_real'] == 7
    old_map = {(d['image_id'], d['step']): d for d in old}
    changes = []
    for d in completed:
        if d['eligible'] != old_map[d['image_id'], d['step']]['eligible']:
            label = labels.get((d['image_id'], d['step']))
            changes.append(dict(d, previously_eligible=old_map[d['image_id'], d['step']]['eligible'],
                                labeled_hall=label[0] if label else None, p_real=label[1] if label else None))
    histogram = Counter(d['minimum_lookahead_steps'] for d in completed if d['eligible'])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=True, images=100, current_word_form_gate=old_summary,
        posthoc_completed_word_gate=new_summary, minimum_lookahead_histogram=dict(sorted(histogram.items())),
        changed_eligibility=changes,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       [*sources, trace / 'predictions.npz', Path(__file__),
                        REPO / 'features/coco_object_history.py', REPO / 'utils/token_alignment.py']},
        limitation='Uses saved future model text only for diagnostics, not available to the current online method. No new generations, training or threshold fitting. Contextual noun/adjective disambiguation is still absent. Non-object slots have no cached detector scores.')
    (output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'changed_eligibility')}, indent=2))
    print('CHANGED_ELIGIBILITY', len(changes), json.dumps(changes[:12], indent=2))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('output', type=Path)
    analyze(args.parse_args().output)
