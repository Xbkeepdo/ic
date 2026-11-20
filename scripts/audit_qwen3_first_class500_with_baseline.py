"""Check paired batches, image caches and class-first gates in the fresh run."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from features.coco_object_history import ObjectClassHistory


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    # A running worker may still be writing the last JSONL record.
    return [json.loads(line) for line in raw[:raw.rfind(b'\n') + 1].splitlines()]


@contextmanager
def cache_file(path, partial):
    try:
        saved = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        if not partial:
            raise
        # A different, not-yet-committed batch may be writing this archive.
        yield None
    else:
        try:
            yield saved
        finally:
            saved.close()


def audit(root, partial=False, baseline='sadi'):
    from transformers import AutoTokenizer
    plan = read(root / 'protocol.json')
    if not partial:
        assert read(root / 'run_status.json')['status'] == 'complete'
    assert plan['max_new_tokens'] == 512 and plan['max_batch'] == 16
    assert read(root / 'cohort_independence.json')['passed']
    assert read(root / 'semantic_preflight.json')['exact_native_sadi_logits_and_tokens']
    prior = Path(plan['prior_protocol']).parent
    semantic_tokens = sorted(read(prior / 'training_vocabulary.json')['token_ids'])
    allowed = set(read(prior / 'object_lexicon.json')['token_ids'])
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, allowed)
    candidate = plan['candidate_method']
    methods = plan['variants']
    assert baseline in methods
    unchanged_key = f'no_accepted_action_equal_{baseline}'
    report = dict(passed=True, partial=partial, paired_images=0, paired_batches=0,
                  cache_files_checked=0, class_decisions=0, class_blocked=0,
                  proposals=0, accepted=0, **{unchanged_key: 0})
    for shard in (0, 1):
        data = {name: rows(root / f'coco500/{name}.shard{shard}.jsonl') for name in methods}
        lookup = {name: {r['image_id']: r for r in records} for name, records in data.items()}
        for name in methods:
            assert len(lookup[name]) == len(data[name])
        common = set.intersection(*(set(records) for records in lookup.values()))
        cursor = 0
        while cursor < len(data['sadi']):
            first = data['sadi'][cursor]
            size = first['batch_size']
            group = data['sadi'][cursor:cursor + size]
            cursor += size
            if len(group) != size or not all(r['image_id'] in common for r in group):
                assert partial
                continue
            ids = [r['image_id'] for r in group]
            for name in methods:
                actual = data[name]
                start = next(i for i, r in enumerate(actual) if r['image_id'] == ids[0])
                assert [r['image_id'] for r in actual[start:start + size]] == ids
                assert all(r['batch_size'] == size and r['visual_token_count'] == first['visual_token_count']
                           for r in actual[start:start + size])
            matches = []
            for path in (root / 'semantics').glob(f'shard{shard}.*.npz'):
                with cache_file(path, partial) as cached:
                    if cached is None:
                        continue
                    if cached['image_ids'].tolist() != ids:
                        continue
                    span = tuple(cached['span'].tolist())
                    key = hashlib.sha256(str((ids, span)).encode()).hexdigest()[:20]
                    assert path.name == f'shard{shard}.{key}.npz'
                    assert span[1] - span[0] == first['visual_token_count']
                    assert cached['token_ids'].tolist() == semantic_tokens
                    values = cached['E_l']
                    assert values.shape == (36, size, len(semantic_tokens)) and values.dtype == np.float32
                    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= span[1] - span[0] + 1e-4
                    matches.append(path)
            assert len(matches) == 1
            report['cache_files_checked'] += 1
            report['paired_batches'] += 1
            for image in ids:
                report['paired_images'] += 1
                row = lookup[candidate][image]
                tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
                assert stats['first_object_class_gate'] and stats['intervention_layer'] == 19
                assert stats['maximum_probe_hidden_difference'] == 0
                assert not any(stats.get(k) for k in ('diagnostic_force_probe_gate', 'diagnostic_force_guidance',
                                                    'diagnostic_zero_replay', 'diagnostic_reject_all'))
                decisions = {d['step']: d for d in stats['class_decisions']}
                assert len(decisions) == len(stats['class_decisions'])
                for step, decision in decisions.items():
                    target = decision['token']
                    assert target in allowed and target not in tokens[:step]
                    prior_classes = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False))
                    assert decision['prior_classes'] == sorted(prior_classes)
                    assert decision['canonical'] == classes[target]
                    assert decision['blocked'] == (classes[target] in prior_classes)
                    if decision['blocked']:
                        assert tokens[step] == target
                        report['class_blocked'] += 1
                    report['class_decisions'] += 1
                events = stats['recycle_events']
                assert len(events) == stats['guided_attention_calls']
                assert sum(e['accepted'] for e in events) == stats['accepted_guidance_calls']
                for event in events:
                    step, target = event['step'], event['token']
                    assert not decisions[step]['blocked'] and decisions[step]['token'] == target
                    assert event['source_layer'] == event['target_layer'] == 19
                    assert event['probe_p_real'] < .5 and event['predicted_log_probability_change'] <= 1e-6
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                    assert event['generated_token'] == tokens[step]
                    if not event['accepted']:
                        assert tokens[step] == target
                    report['proposals'] += 1
                    report['accepted'] += int(event['accepted'])
                if not stats['accepted_guidance_calls']:
                    assert tokens == lookup[baseline][image]['response_token_ids']
                    report[unchanged_key] += 1
    if not partial:
        assert report['paired_images'] == 500
    (root / 'first_class_runtime_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--baseline', choices=('sadi', 'svar'), default='sadi')
    args = parser.parse_args()
    audit(args.root, args.partial, args.baseline)
