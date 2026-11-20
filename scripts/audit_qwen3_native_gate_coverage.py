"""Reconstruct the deployed lexical/history gates on frozen native captions.

Uses cached original I/E_l probabilities, not a new model forward or a fit.
CHAIR labels are kept as automatic labels; this does not certify visual truth.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from features.coco_object_history import ObjectClassHistory
from scripts.run_qwen3_signed_feedback import source_groups, group_keys


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(trace, reference, output):
    from transformers import AutoTokenizer
    assert not output.exists()
    plan, previous = read(trace / 'protocol.json'), read(reference / 'protocol.json')
    assert read(trace / 'status.json')['status'] == 'complete'
    assert read(trace / 'analysis.json')['passed'] and read(reference / 'first_class_audit.json')['passed']
    checkpoint = Path(plan['probe']) / 'model.pt'
    assert sha(checkpoint) == read(trace / 'analysis.json')['checkpoint_sha256']
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    allowed = set(read(reference / 'object_lexicon.json')['token_ids'])
    canonical = parser.token_classes(tokenizer, allowed)
    with np.load(trace / 'predictions.npz', allow_pickle=False) as values:
        cache = {k: values[k].copy() for k in values.files}
    records, native, proposals, inputs = [], {}, {}, []
    for shard in (0, 1):
        feature_path = trace / f'features.shard{shard}.jsonl'
        records.extend(map(json.loads, feature_path.read_text().splitlines()))
        source = Path(plan['source']) / f'coco100/sadi.shard{shard}.jsonl'
        deployed = reference / f'coco100/sadi.shard{shard}.jsonl'
        a, b = source_groups(source), source_groups(deployed)
        assert group_keys(a) == group_keys(b)
        for x, y in zip([r for g in a for r in g], [r for g in b for r in g], strict=True):
            assert x['image_id'] == y['image_id'] and x['response_token_ids'] == y['response_token_ids']
            assert x['generated_text'] == y['generated_text']
            native[x['image_id']] = x
        reference_name, = previous['candidates']
        path = reference / f'coco100/{reference_name}.shard{shard}.jsonl'
        proposals.update({r['image_id']: r for g in source_groups(path) for r in g})
        inputs += [feature_path, source, deployed, path]
    assert len(native) == 100 and len(records) == len(cache['p_real']) == 1121
    np.testing.assert_array_equal(np.array([r['features'] for r in records], np.float32), cache['features'])
    np.testing.assert_array_equal([r['image_id'] for r in records], cache['image_id'])
    np.testing.assert_array_equal([r['step'] for r in records], cache['step'])
    np.testing.assert_array_equal([r['target_token_id'] for r in records], cache['token_id'])
    assert all(len(r['mentions']) == 1 and r['native_argmax_matches'] for r in records)
    inputs += [trace / n for n in ('protocol.json', 'analysis.json', 'predictions.npz')]
    inputs += [reference / n for n in ('protocol.json', 'object_lexicon.json', 'first_class_audit.json')]
    inputs += [Path(__file__), REPO / 'features/coco_object_history.py', checkpoint,
        Path(plan['probe']) / 'input_scaler.npz']
    stages = {key: [] for key in ('all_mentions', 'training_vocabulary', 'single_token_lexicon',
        'unseen_token_id', 'absent_class', 'cached_probe_below_half')}
    labelled, misses = [], []
    for i, record in enumerate(records):
        image, step, token = record['image_id'], record['step'], record['target_token_id']
        tokens = native[image]['response_token_ids']
        assert tokens[step] == token
        mention = record['mentions'][0]
        assert bool(cache['hallucinated'][i]) == mention['hallucinated']
        lexical = token in allowed
        unseen = lexical and token not in tokens[:step] and step > 0
        prefix = tokenizer.decode(tokens[:step], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prior = parser.classes(prefix) if lexical else set()
        absent = unseen and canonical[token] not in prior
        real = float(cache['p_real'][i])
        eligible = absent and real < .5
        masks = dict(all_mentions=True, training_vocabulary=record['in_train_vocabulary'],
            single_token_lexicon=lexical, unseen_token_id=unseen, absent_class=absent,
            cached_probe_below_half=eligible)
        assert not lexical or record['in_train_vocabulary']
        for key, value in masks.items():
            if value:
                stages[key].append(i)
        item = dict(image_id=image, step=step, token_id=token, token_text=tokenizer.decode([token]),
            canonical=mention['canonical'], hallucinated=mention['hallucinated'], repeat=mention['repeat'],
            p_real=real, filters=masks, history_classes=sorted(prior),
            context=tokenizer.decode(tokens[max(0, step-16):step+14], skip_special_tokens=True,
                clean_up_tokenization_spaces=False))
        labelled.append(item)
        if mention['hallucinated']:
            misses.append(item)
    report = dict(passed=True, scope='1121 automatically labelled native-SADI object positions only. '
        'The final probability gate is reconstructed from cached CPU probe values; altered rollouts and '
        'unlabelled online candidates are outside this coverage estimate.', images=100,
        labels_are_automatic_CHAIR=True, no_model_forward_or_fit=True, allowed_tokens=len(allowed), stages={})
    for name, indices in stages.items():
        subset = [labelled[i] for i in indices]
        false = [r for r in subset if r['hallucinated']]
        report['stages'][name] = dict(positions=len(subset), false_positions=len(false),
            real_positions=len(subset)-len(false), first_false_positions=sum(not r['repeat'] for r in false),
            false_image_classes=len({(r['image_id'], r['canonical']) for r in false}),
            false_images=len({r['image_id'] for r in false}))
    lookup = {(r['image_id'], r['step'], r['token_id']): r for r in labelled}
    agreement, uncovered = [], []
    for image, row in proposals.items():
        tokens = row['response_token_ids']
        events = row['sar_ems_statistics']['recycle_events']
        for event in events:
            step = event['step']
            if any(e['accepted'] and e['step'] < step for e in events):
                continue
            assert tokens[:step] == native[image]['response_token_ids'][:step]
            assert event['token'] == native[image]['response_token_ids'][step]
            key = (image, step, event['token'])
            item = dict(image_id=image, step=step, token=event['token'], online_p_real=event['probe_p_real'])
            if key in lookup:
                matched = lookup[key]
                item.update(cached_p_real=matched['p_real'], hallucinated=matched['hallucinated'],
                    canonical=matched['canonical'], probability_difference=abs(event['probe_p_real']-matched['p_real']))
                assert matched['filters']['absent_class']
                agreement.append(item)
            else:
                item['token_text'] = tokenizer.decode([event['token']])
                uncovered.append(item)
    report['native_prefix_online_check'] = dict(annotated_proposals=len(agreement),
        automatic_false=sum(r['hallucinated'] for r in agreement), unlabelled_proposals=len(uncovered),
        max_cached_online_probability_difference=max((r['probability_difference'] for r in agreement), default=0),
        annotated=agreement, unlabelled=uncovered)
    first_false = [r for r in misses if not r['repeat']]
    report['first_false_rejection_reasons'] = dict(Counter(
        'lexicon' if not r['filters']['single_token_lexicon'] else
        'seen_token' if not r['filters']['unseen_token_id'] else
        'history_class' if not r['filters']['absent_class'] else
        'probe' if not r['filters']['cached_probe_below_half'] else 'eligible'
        for r in first_false))
    output.mkdir()
    (output / 'protocol.json').write_text(json.dumps(dict(trace=str(trace), reference=str(reference),
        source_sha256={str(p): sha(p) for p in inputs}), indent=2)+'\n')
    (output / 'coverage.json').write_text(json.dumps(report, indent=2)+'\n')
    (output / 'positions.json').write_text(json.dumps(labelled, ensure_ascii=False, indent=2)+'\n')
    (output / 'false_positions.json').write_text(json.dumps(misses, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'native_prefix_online_check'}, ensure_ascii=False))
    print('ONLINE_CHECK', json.dumps({k: v for k, v in report['native_prefix_online_check'].items() if k not in ('annotated','unlabelled')}))
    print('FIRST_FALSE', json.dumps([dict(image_id=r['image_id'],step=r['step'],token=r['token_text'],
        canonical=r['canonical'],p_real=r['p_real'],eligible=r['filters']['cached_probe_below_half']) for r in first_false]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.trace, args.reference, args.output)
