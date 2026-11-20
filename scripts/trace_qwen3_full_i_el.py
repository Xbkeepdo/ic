"""Check original full I/E_l probe transfer on saved, exactly replayed SADI captions."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared

RUN = shared.OUT / 'qwen3_sadi_full_i_el_trace'
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-vocabulary-pilot')
TRAIN = REPO.parent / 'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'


def prepare():
    import numpy as np
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    if RUN.exists():
        raise FileExistsError(RUN)
    files = (*shared.generation_sources(), 'features/qwen3_online_i_el.py',
             'features/attention_writes.py', 'models/dgst_capture.py',
             'scripts/trace_qwen3_full_i_el.py', 'scripts/analyze_qwen3_full_i_el.py',
             'scripts/evaluate_qwen3_next200_sentence_tokenized.py', 'coco-labeling/coco_chair.py',
             'utils/token_alignment.py')
    hashes = {str(REPO / name): shared.sha(REPO / name) for name in files}
    cached = shared.read(SOURCE / 'protocol.json')['source_sha256']
    for name in shared.generation_sources():
        expected = cached.get(str(REPO / name), cached.get(name))
        assert expected is not None and hashes[str(REPO / name)] == expected, name
    for path in (SOURCE / 'protocol.json', TRAIN / 'compact_features.npz',
                 TRAIN / 'I+E_l/seed43/model.pt', TRAIN / 'I+E_l/seed43/input_scaler.npz'):
        hashes[str(path)] = shared.sha(path)
    with np.load(TRAIN / 'compact_features.npz', allow_pickle=False) as data:
        vocabulary = sorted(set(map(int, data['target_token_id'][data['train_mask']])))
        old_ids = set(map(int, data['image_id']))
    ids = shared.read(SOURCE / 'cohort.json')['coco100']
    assert len(ids) == 100 and not old_ids.intersection(ids)
    groups = []
    for shard in (0, 1):
        path = SOURCE / f'coco100/sadi.shard{shard}.jsonl'
        hashes[str(path)] = shared.sha(path)
        groups.append([dict(ids=[r['image_id'] for r in group],
                            visual_token_count=group[0]['visual_token_count'])
                       for group in shared.source_groups(path)])
    RUN.mkdir()
    for directory in ('logs', 'semantics', 'server_sources'):
        (RUN / directory).mkdir()
    for name in files:
        target = RUN / 'server_sources' / name
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(REPO / name, target)
    shared.write(RUN / 'cohort.json', dict(trace=ids))
    shared.write(RUN / 'training_vocabulary.json', dict(token_ids=vocabulary))
    shared.write(RUN / 'protocol.json', dict(
        source=str(SOURCE), images=100, shards=2, groups=groups, max_new_tokens=512,
        source_sha256=hashes, probe=str(TRAIN / 'I+E_l/seed43'),
        purpose='Diagnostic transfer of the frozen original full I+E_l probe; no mitigation or parameter fitting.',
        label_protocol='sentence CHAIR object mentions, first aligned token per mention, repeated mentions included.',
        cache_policy='Original training token IDs plus measured diagnostic targets; semantic sums only.',
        gates_to_report=['p_real < 0.5', 'p_real < 0.1', 'old-training real 5th percentile'],
        sample_scope='Previously observed 100-image development set; disjoint old COCO4000 detector data.'))


def preflight(wrapper, nouns):
    import torch
    from features.qwen3_online_i_el import OnlineIElCapture
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.generation_time_evidence_drift_qwen3 import MODEL, COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16))
    assert len(group) == 2
    inputs = {k: torch.cat([item[k] for _, item in group]).to(wrapper.device) for k in group[0][1]}
    target = [wrapper.tokenizer.encode(' person', add_special_tokens=False)[0],
              wrapper.tokenizer.encode(' dog', add_special_tokens=False)[0]]
    options = dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=16)

    def run(capture):
        logits, report = [], {}

        def callback(reader, _output):
            if reader.step == 0:
                report.update(reader.reference_check([0, 1], target))

        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(wrapper.model, span, nouns, 'sadi').capture())
            if capture:
                stack.enter_context(OnlineIElCapture(wrapper.model, span, target,
                                                     callback=callback, validate=True).capture())
            handle = wrapper.model.register_forward_hook(lambda _m, _a, out: logits.append(out.logits[:, -1].cpu()))
            stack.callback(handle.remove)
            tokens = wrapper.model.generate(**inputs, **options).cpu()
        return tokens, logits, report

    with torch.inference_mode():
        expected, logits, _ = run(False)
        tokens, current, report = run(True)
    assert torch.equal(expected, tokens)
    assert len(logits) == len(current) and all(torch.equal(a, b) for a, b in zip(logits, current, strict=True))
    assert report['I_max_relative'] < .01 and report['E_l_max_relative'] < 1e-4, report
    report.update(passed=True, exact_baseline_logits_and_tokens=True, steps=len(logits),
                  image_ids=[i for i, _ in group])
    return report


def worker(shard):
    import nltk
    import numpy as np
    import torch
    from transformers import LogitsProcessor, StoppingCriteria
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_online_i_el import OnlineIElCapture
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from models import build_model
    from scripts.evaluate_generation_time_evidence_drift_llava import load_chair
    from scripts.evaluate_qwen3_next200_sentence_tokenized import load_original_chair_module, make_sentence_chair
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    from utils.config_utils import get_extraction_model_cfg, load_config
    from utils.token_alignment import build_response_token_offsets, token_indices_for_char_span

    plan = shared.read(RUN / 'protocol.json')
    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL), hf_name=str(MODEL_ROOT))
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    if shard == 0:
        shared.write(RUN / 'feature_preflight.json', preflight(wrapper, nouns))
    original = load_original_chair_module()
    sentence = make_sentence_chair(original.CocoChairEvaluator, nltk.data.load('tokenizers/punkt/english.pickle'))
    with patch.object(original, 'CocoChairEvaluator', sentence):
        chair, cache = load_chair(COCO_ROOT, SOURCE / 'evaluations/sentence_v1',
                                  shared.read(SOURCE / 'cohort.json')['coco100'], None)
    assert shared.read(cache)['evaluator_provenance']['evaluator'] == 'sentence'
    allowed = frozenset(shared.read(RUN / 'training_vocabulary.json')['token_ids'])
    eos = wrapper.model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos

    class Replay(LogitsProcessor):
        def __init__(self, records, prompt):
            self.records, self.prompt = records, prompt
            self.checked = self.mismatches = 0

        def __call__(self, ids, scores):
            step = ids.shape[1] - self.prompt
            predicted = scores.argmax(-1).tolist()
            chosen = []
            for row, record in enumerate(self.records):
                active = step < len(record['response_token_ids'])
                token = record['response_token_ids'][step] if active else eos
                self.checked += active
                self.mismatches += active and predicted[row] != token
                chosen.append(token)
            return torch.full_like(scores, -float('inf')).scatter_(1, scores.new_tensor(chosen, dtype=torch.long)[:, None], 0)

    class Stop(StoppingCriteria):
        def __init__(self, length):
            self.length = length

        def __call__(self, ids, scores, **_):
            return ids.new_full((len(ids),), ids.shape[1] >= self.length, dtype=torch.bool)

    path = RUN / f'features.shard{shard}.jsonl'
    checks = []
    with torch.inference_mode(), path.open('x', encoding='utf-8') as stream:
        groups = shared.source_groups(SOURCE / f'coco100/sadi.shard{shard}.jsonl')
        for index, (group, expected) in enumerate(zip(groups, plan['groups'][shard], strict=True)):
            ids = [r['image_id'] for r in group]
            assert ids == expected['ids']
            wanted = {}
            for row, record in enumerate(group):
                text = record['generated_text']
                offsets = build_response_token_offsets(wrapper.tokenizer, record['response_token_ids'], text)
                seen = set()
                for mention in chair.compute_chair_token(record['image_id'], text)['object_mentions']:
                    repeat = mention['canonical_name'] in seen
                    seen.add(mention['canonical_name'])
                    positions = token_indices_for_char_span(offsets, mention['char_start'], mention['char_end'])
                    assert positions, (record['image_id'], mention)
                    step = positions[0]
                    event = wanted.setdefault(step, {}).setdefault(row, dict(
                        image_id=record['image_id'], step=step, shard=shard, batch=index,
                        target_token_id=record['response_token_ids'][step], mentions=[]))
                    event['mentions'].append(dict(canonical=mention['canonical_name'],
                        hallucinated=bool(mention['hallucinated']), repeat=repeat))
            if not wanted:
                continue
            tokens = sorted(allowed | {e['target_token_id'] for events in wanted.values() for e in events.values()})
            (span, items), = same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [i for i, _ in items] == ids and span[1] - span[0] == expected['visual_token_count']
            inputs = {k: torch.cat([data[k] for _, data in items]).to(wrapper.device) for k in items[0][1]}
            prompt = inputs['input_ids'].shape[1]
            processor = Replay(group, prompt)

            def callback(reader, output):
                events = wanted.get(reader.step, {})
                if not events:
                    return
                rows = list(events)
                targets = [events[row]['target_token_id'] for row in rows]
                features = reader.features(rows, targets).cpu().tolist()
                support = reader.visual_support31(rows, targets).cpu().tolist()
                predicted = output.logits[:, -1].argmax(-1).tolist()
                for row, feature, v in zip(rows, features, support, strict=True):
                    events[row].update(features=feature, visual_support31=v,
                        native_argmax_matches=predicted[row] == events[row]['target_token_id'],
                        in_train_vocabulary=events[row]['target_token_id'] in allowed,
                        token_is_noun=events[row]['target_token_id'] in nouns)

            reader = OnlineIElCapture(wrapper.model, span, tokens, callback=callback)
            options = dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=512,
                           logits_processor=[processor], stopping_criteria=[Stop(prompt + max(wanted) + 1)])
            started = time.time()
            with Qwen3BatchLateAttention(wrapper.model, span, nouns, 'sadi').capture(), reader.capture():
                wrapper.model.generate(**inputs, **options)
            elapsed = time.time() - started
            assert all('features' in e for events in wanted.values() for e in events.values())
            for step in sorted(wanted):
                for event in wanted[step].values():
                    stream.write(shared.json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            np.savez_compressed(RUN / f'semantics/shard{shard}.batch{index}.npz',
                image_ids=np.array(ids), token_ids=np.array(tokens), span=np.array(span),
                E_l=torch.stack([reader.semantics[i] for i in range(36)]).cpu().numpy())
            checks.append(dict(batch=index, ids=ids, positions=sum(map(len, wanted.values())),
                checked_tokens=processor.checked, native_argmax_mismatches=processor.mismatches,
                seconds=elapsed, semantic_seconds=reader.semantic_seconds,
                feature_seconds=reader.feature_seconds))
            shared.write(RUN / f'batch_checks.shard{shard}.json', checks)
            print('BATCH', shard, index, 'positions', checks[-1]['positions'], 'seconds', round(elapsed, 2),
                  'mismatches', processor.mismatches, flush=True)


def run():
    prepare()
    started = time.time()
    children = []
    try:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        assert len(used.splitlines()) == 2 and max(map(int, used.splitlines())) < 100
        for shard in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', NLTK_DATA=str(shared.OUT / 'generation_time_evidence_drift_llava/nltk_data'))
            with (RUN / f'logs/gpu{shard}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', str(shard)],
                                                  cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        shared.write(RUN / 'status.json', dict(status='running', driver=os.getpid(),
                                              workers=[p.pid for p in children], started=started))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Full I/E_l trace worker failed')
            time.sleep(3)
        assert all(p.returncode == 0 for p in children)
        shared.write(RUN / 'status.json', dict(status='analyzing', driver=os.getpid(), started=started))
        subprocess.run([sys.executable, str(REPO / 'scripts/analyze_qwen3_full_i_el.py'), str(RUN)],
                       cwd=REPO, check=True)
        shared.write(RUN / 'status.json', dict(status='complete', started=started, finished=time.time()))
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        shared.write(RUN / 'status.json', dict(status='failed', error=str(exc), started=started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
