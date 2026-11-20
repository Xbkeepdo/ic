"""Read108 features from cached, originally SADI-generated source captions.

The cached batch geometry is preserved. Labels use sentence CHAIR and actual
response-token offsets. Only annotated candidates eligible for the deployed
lexical/first-class gate are measured. Current argmax agreement is saved per
target; mismatched targets cannot enter the subsequent supervised fit.
"""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from scripts.verify_qwen3_teacher_sadi_capture import read, write, sha


def prepare(root):
    import nltk
    import numpy as np
    from transformers import AutoTokenizer
    from features.coco_object_history import ObjectClassHistory
    from scripts.run_qwen3_signed_feedback import source_groups
    from scripts.evaluate_generation_time_evidence_drift_llava import load_chair
    from scripts.evaluate_qwen3_next200_sentence_tokenized import load_original_chair_module, make_sentence_chair
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL_ROOT
    from utils.token_alignment import build_response_token_offsets, token_indices_for_char_span

    assert not root.exists()
    source = REPO / 'outputs/qwen3_causal_visual_tilt_validation500_20261119'
    assert read(source / 'final_provenance.json')['passed']
    lexicon = REPO / 'outputs/qwen3_mass_matched_visual_pilot_20261119/object_lexicon.json'
    current = REPO / 'outputs/qwen3_mass_matched_visual_pilot_20261119/cohort.json'
    reserved = REPO / 'outputs/qwen3_mass_matched_visual_reserved500_20261119/cohort.json'
    allowed = read(lexicon)['token_ids']
    groups, records = [], {}
    for shard in [0, 1]:
        for source_index, batch in enumerate(source_groups(source / f'coco500/sadi.shard{shard}.jsonl')):
            assert len(batch) <= 16 and all(record['variant'] == 'sadi' for record in batch)
            ids = [record['image_id'] for record in batch]
            assert not set(ids) & set(records)
            records.update({record['image_id']: record for record in batch})
            groups.append(dict(ids=ids, visual_tokens=batch[0]['visual_token_count'],
                               source_shard=shard, source_group=source_index))
    images = sorted(records)
    assert len(images) == 500
    assert not set(images) & set(read(current)['coco100'])
    assert not set(images) & set(read(reserved)['coco500'])
    files = [Path(__file__), source / 'protocol.json', source / 'final_provenance.json', lexicon, current, reserved,
        *[source / f'coco500/sadi.shard{s}.jsonl' for s in [0, 1]],
        *[REPO / name for name in [
            'scripts/run_qwen3_signed_feedback.py', 'features/coco_object_history.py',
            'scripts/evaluate_generation_time_evidence_drift_llava.py',
            'scripts/evaluate_qwen3_next200_sentence_tokenized.py', 'utils/token_alignment.py',
            'features/qwen3_source_native_capture.py', 'features/qwen3_native_risk_i_el.py',
            'features/qwen3_online_risk_f.py', 'features/dgst_t.py',
            'features/qwen3_sar_ems_attention_batch.py', 'features/llava_sadi.py',
            'features/llava_decoding_evidence.py', 'models/qwen3_vl_wrapper.py',
            'models/dgst_capture.py', 'models/base_wrapper.py', 'utils/config_utils.py',
            'scripts/run_qwen3_semantic_final_pilot.py',
            'scripts/generation_time_evidence_drift_qwen3_batched.py',
            'scripts/generation_time_evidence_drift_qwen3.py', 'coco-labeling/coco_chair.py',
            'configs/model_configs_coco4000_512_endac_baselines_v2.yaml']]]
    cached = read(source / 'protocol.json')['source_sha256']
    matched = []
    for path in files:
        if path.is_relative_to(REPO):
            expected = cached.get(str(path), cached.get(str(path.relative_to(REPO))))
            if expected is not None:
                assert sha(path) == expected, path
                matched.append(str(path))
    assert str(REPO / 'features/qwen3_sar_ems_attention_batch.py') in matched
    root.mkdir(); (root / 'preflight').mkdir(); (root / 'groups').mkdir()
    (root / 'evaluations').mkdir(); (root / 'source').mkdir()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, allowed)
    original = load_original_chair_module()
    sentence = make_sentence_chair(original.CocoChairEvaluator, nltk.data.load('tokenizers/punkt/english.pickle'))
    with patch.object(original, 'CocoChairEvaluator', sentence):
        chair, cache = load_chair(COCO_ROOT, root / 'evaluations', images, None)
    # The evaluator class is constructed above with the sentence-aware tokenizer.
    files.append(cache)
    train_images = set(map(int, np.random.default_rng(43).permutation(images)[:400]))
    entries, unaligned = [], []
    for group_index, group in enumerate(groups):
        for batch, image in enumerate(group['ids']):
            record = records[image]; ids = record['response_token_ids']; text = record['generated_text']
            offsets = build_response_token_offsets(tokenizer, ids, text)
            mentions = defaultdict(list)
            for mention in chair.compute_chair_token(image, text)['object_mentions']:
                indices = token_indices_for_char_span(offsets, mention['char_start'], mention['char_end'])
                assert indices, (image, mention)
                mentions[indices[0]].append(mention)
            for step, token in enumerate(ids):
                if step == 0 or token not in classes or token in ids[:step]:
                    continue
                prefix = tokenizer.decode(ids[:step], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if classes[token] in parser.classes(prefix):
                    continue
                item = dict(image_id=image, step=step, target_token_id=token,
                    runtime_canonical=classes[token], group=group_index, batch=batch,
                    train=image in train_images)
                targets = mentions.get(step, [])
                labels = {not bool(mention['hallucinated']) for mention in targets}
                if len(labels) != 1:
                    unaligned.append(dict(**item, reason='No unique aligned CHAIR label'))
                    continue
                entries.append(dict(**item, row=len(entries), label_real=int(labels.pop()),
                    mentions=[dict(canonical=m['canonical_name'], hallucinated=bool(m['hallucinated']),
                                   surface=m['surface']) for m in targets]))
    preflight_groups = [next(i for i, g in enumerate(groups) if g['source_shard'] == shard) for shard in [0, 1]]
    assert entries and all(any(item['group'] == index for item in entries) for index in preflight_groups)
    columns = {key: np.asarray([item[key] for item in entries]) for key in
               ['row', 'image_id', 'step', 'target_token_id', 'label_real', 'group', 'batch', 'train']}
    np.savez_compressed(root / 'index.npz', **columns)
    write(root / 'targets.json', entries); write(root / 'unaligned.json', unaligned)
    files += [root / 'index.npz', root / 'targets.json', root / 'unaligned.json']
    hashes = {str(path): sha(path) for path in files}
    for path in files:
        if path.is_relative_to(REPO) and path.suffix in ['.py', '.yaml']:
            target = root / 'source' / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    plan = dict(fixed_at=time.time(), source=str(source), source_sha256=hashes,
        images=images, groups=groups, shards=2, max_batch=16, preflight_groups=preflight_groups,
        complete_sadi_layers=list(range(5, 19)), feature_order=['risk0:36', 'I0:36', 'E_l0:36'],
        features='Original exact risk K32/tau.03 and unit cost/endpoint penalties; actual visual-write I; final-norm hpre El.',
        policy='Reuse actual cached SADI captions and exact original ordered batches. Real one-query KV; no new captions.',
        target_scope='Same309 lexical tokens, unseen token ID and class absent from the generated prefix; sentence-CHAIR aligned candidates.',
        labels='Automatic source training labels from complete source captions; no target development labels or reserved-image annotations read.',
        split='400/100 source images, default_rng43 over sorted500; fixed across future training seeds. Calibrate on real source-test targets only.',
        matched_native='Read current native argmax at every collected target. Later fitting excludes mismatches; original labels never assigned to substitute tokens.',
        preflight='First original group in each source shard. Require >=95% overall target argmax agreement before full collection; no target-label-based selection.',
        compute='Target-only output-head calls. Native visual probabilities built once per image/layer; preflight batches reused byte-for-byte by full workers.',
        original_generation_dependencies_checked=matched, no_current_or_reserved_training=True)
    write(root / 'protocol.json', plan)
    print(dict(prepared=True, images=500, groups=len(groups), targets=len(entries),
        unaligned=len(unaligned), labels=dict(Counter(item['label_real'] for item in entries)),
        preflight_groups=preflight_groups), flush=True)


def worker(root, shard, preflight):
    import numpy as np
    import torch
    from features.qwen3_source_native_capture import SourceNativeCapture
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    plan = read(root / 'protocol.json'); assert shard in [0, 1]
    for filename, expected in plan['source_sha256'].items(): assert sha(Path(filename)) == expected, filename
    if not preflight: assert read(root / 'preflight/verification.json')['passed']
    selected = plan['preflight_groups'] if preflight else list(range(len(plan['groups'])))
    records = {}
    for s in [0, 1]:
        for line in (Path(plan['source']) / f'coco500/sadi.shard{s}.jsonl').read_text().splitlines():
            import json
            row = json.loads(line); records[row['image_id']] = row
    indexed = read(root / 'targets.json')
    wrapper = build_wrapper(); model = wrapper.model; core = model.model
    encoder_calls = [0]
    hook = model.model.visual.register_forward_hook(lambda *_: encoder_calls.__setitem__(0, encoder_calls[0]+1))
    begun = time.time(); count = 0
    for group_index in selected:
        group = plan['groups'][group_index]
        if group['source_shard'] != shard: continue
        destination = root / ('preflight' if preflight else 'groups') / f'{group_index:04d}'
        assert not destination.with_suffix('.npz').exists()
        if not preflight and group_index in plan['preflight_groups']:
            import shutil
            original = root / 'preflight' / f'{group_index:04d}'
            for extension in ['.npz', '.json']:
                shutil.copyfile(original.with_suffix(extension), destination.with_suffix(extension))
            assert sha(destination.with_suffix('.npz')) == read(destination.with_suffix('.json'))['feature_sha256']
            continue
        wanted = defaultdict(list)
        for item in indexed:
            if item['group'] == group_index: wanted[item['step']].append(item)
        assert wanted, group_index
        (span, items), = same_span_batches(wrapper, group['ids'], COCO_ROOT, len(group['ids']))
        assert [i for i, _ in items] == group['ids'] and span[1]-span[0] == group['visual_tokens']
        inputs = {key: torch.cat([item[key] for _, item in items], dim=0).to(wrapper.device) for key in items[0][1]}
        responses = [records[image]['response_token_ids'] for image in group['ids']]
        collector = SourceNativeCapture(model, span, sorted({item['target_token_id'] for values in wanted.values() for item in values}))
        core.rope_deltas = None; past = None; prompt = inputs['input_ids'].shape[1]
        values, matches, identities, predicted = [], [], [], []
        before = encoder_calls[0]; started = time.time(); head_calls = 0
        with torch.inference_mode(), Qwen3BatchLateAttention(model, span, frozenset(), 'sadi').capture(), collector.capture():
            for step in range(max(wanted)+1):
                if step == 0:
                    current = dict(inputs, cache_position=torch.arange(prompt, device=wrapper.device))
                else:
                    previous = [r[step-1] if step <= len(r) else wrapper.tokenizer.eos_token_id for r in responses]
                    current = dict(input_ids=torch.tensor(previous, device=wrapper.device)[:, None],
                        attention_mask=torch.ones((len(responses), prompt+step), device=wrapper.device, dtype=inputs['attention_mask'].dtype),
                        past_key_values=past, cache_position=torch.tensor([prompt+step-1], device=wrapper.device))
                output = core(**current, use_cache=True, return_dict=True); past = output.past_key_values
                if step not in wanted: continue
                targets = wanted[step]
                native = model.get_output_embeddings()(output.last_hidden_state[:, -1:]).argmax(-1)[:, 0].tolist()
                head_calls += 1
                matrix = collector.features([t['batch'] for t in targets], [t['target_token_id'] for t in targets]).cpu().numpy()
                for target, vector in zip(targets, matrix, strict=True):
                    identities.append(target['row']); values.append(vector)
                    predicted.append(native[target['batch']]); matches.append(native[target['batch']] == target['target_token_id'])
        assert encoder_calls[0]-before == 1
        assert collector.feature_calls == len(identities) and collector.risk_solver_calls == 36*len(identities)
        np.savez_compressed(destination.with_suffix('.npz'), row=np.asarray(identities), features=np.asarray(values),
            native_match=np.asarray(matches), native_argmax=np.asarray(predicted))
        information = dict(group=group_index, shard=shard, images=group['ids'], targets=len(identities),
            native_matches=sum(matches), seconds=time.time()-started, encoder_calls=1,
            feature_calls=collector.feature_calls, risk_solver_calls=collector.risk_solver_calls,
            target_head_calls=head_calls, semantic_head_calls=collector.head_calls,
            feature_sha256=sha(destination.with_suffix('.npz')))
        write(destination.with_suffix('.json'), information); print(information, flush=True)
        count += 1
        del collector, inputs, output, past, items, values
    hook.remove()
    for filename, expected in plan['source_sha256'].items(): assert sha(Path(filename)) == expected, filename
    where = root / 'preflight' if preflight else root
    write(where / f'worker{shard}.json', dict(passed=True, shard=shard, seconds=time.time()-begun,
        freshly_computed_groups=count, encoder_calls=encoder_calls[0], peak_cuda_bytes=torch.cuda.max_memory_allocated()))


def audit(root, preflight):
    import numpy as np
    plan = read(root / 'protocol.json'); folder = root / 'preflight' if preflight else root / 'groups'
    for shard in [0, 1]: assert read((root / 'preflight' if preflight else root) / f'worker{shard}.json')['passed']
    selected = plan['preflight_groups'] if preflight else list(range(len(plan['groups'])))
    rows, matrices, matches, native = [], [], [], []
    for group_index in selected:
        path = folder / f'{group_index:04d}'
        info = read(path.with_suffix('.json')); assert sha(path.with_suffix('.npz')) == info['feature_sha256']
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as batch:
            assert batch['features'].shape == (len(batch['row']), 108) and np.isfinite(batch['features']).all()
            rows.extend(batch['row']); matrices.extend(batch['features']); matches.extend(batch['native_match']); native.extend(batch['native_argmax'])
    order = np.argsort(rows); rows = np.asarray(rows)[order]; matrices = np.asarray(matrices)[order]
    matches = np.asarray(matches)[order]; native = np.asarray(native)[order]
    assert len(rows) == len(set(map(int, rows)))
    index = np.load(root / 'index.npz', allow_pickle=False)
    expected = np.flatnonzero(np.isin(index['group'], selected)); np.testing.assert_array_equal(rows, expected)
    ratio = float(matches.mean())
    if preflight:
        output = dict(passed=ratio >= .95, protocol_sha256=sha(root / 'protocol.json'),
            groups=selected, targets=len(rows), native_matches=int(matches.sum()), native_match_fraction=ratio,
            no_new_captions=True, full_run_permitted=ratio >= .95)
        write(root / 'preflight/verification.json', output)
    else:
        np.testing.assert_array_equal(rows, np.arange(len(index['row'])))
        np.savez_compressed(root / 'features.npz', row=rows, features=matrices, native_match=matches, native_argmax=native)
        retained = {name: int((matches & (index['train'] == flag) & (index['label_real'] == label)).sum())
                    for flag, split in [(True, 'train'), (False, 'calibration')] for label, truth in [(1, 'real'), (0, 'hall')]
                    for name in [f'{split}_{truth}']}
        output = dict(passed=True, protocol_sha256=sha(root / 'protocol.json'), feature_sha256=sha(root / 'features.npz'),
            images=500, groups=len(selected), targets=len(rows), native_matches=int(matches.sum()),
            native_match_fraction=ratio, retained_for_fit=retained, no_new_captions=True,
            no_training=True, no_development_or_reserved_labels=True)
        write(root / 'verification.json', output)
    for filename, expected_sha in plan['source_sha256'].items(): assert sha(Path(filename)) == expected_sha, filename
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=int, choices=[0, 1])
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        elif args.worker is not None: worker(args.root, args.worker, args.preflight)
        elif args.audit: audit(args.root, args.preflight)
        else: parser.error('Choose prepare, worker or audit.')
    except Exception:
        if args.root.exists():
            path = args.root / (f'failure.worker{args.worker}.json' if args.worker is not None else 'failure.json')
            if not path.exists(): write(path, dict(failed_at=time.time(), traceback=traceback.format_exc()))
        raise
