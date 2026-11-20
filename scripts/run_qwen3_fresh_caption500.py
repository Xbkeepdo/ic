"""Evaluate a frozen attention intervention on 500 previously unused COCO images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
COCO = REPO.parent / 'dataset/coco'
OUT = REPO / 'outputs'
RUN = OUT / 'generation_time_evidence_drift_qwen3_fresh_alternative500'
SADI, NEXT, CANDIDATE, SVAR = ('sadi', 'sadi_sar_ems_next_19_20', 'sadi_sar_ems_alternative_25_26', 'svar')
VARIANTS = (SADI, NEXT, CANDIDATE, SVAR)
COUNT, CAP, BATCH, SEED = 500, 256, 16, 20261117
SOURCES = (
    'features/qwen3_sar_ems_alternative.py', 'features/qwen3_sar_ems_signed.py',
    'features/qwen3_sar_ems_layer_feedback.py', 'features/qwen3_sar_ems_attention_batch.py',
    'features/qwen3_sar_ems_attention.py', 'features/llava_sar_ems_attention.py',
    'features/llava_sadi.py', 'features/llava_decoding_evidence.py',
    'scripts/generation_time_qwen3_layer_feedback.py',
    'scripts/generation_time_evidence_drift_qwen3_batched.py',
    'scripts/generation_time_evidence_drift_qwen3.py',
    'scripts/generation_time_sar_ems_seed43.py',
    'models/qwen3_vl_wrapper.py', 'configs/model_configs_coco4000_512_endac_baselines_v2.yaml',
    'scripts/evaluate_caption_chair.py', 'scripts/evaluate_generation_time_evidence_drift_llava.py',
    'scripts/evaluate_qwen3_next200_sentence_tokenized.py', 'coco-labeling/coco_chair.py',
    'scripts/experiment_snapshots/qwen3_new200_b16/prepare.py',
    'scripts/run_qwen3_fresh_caption500.py',
)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ids_sha(ids):
    return sha(json.dumps(sorted(ids), separators=(',', ':')).encode())


def prepare():
    from scripts.experiment_snapshots.qwen3_new200_b16.prepare import cohort_ids, record_ids

    if RUN.exists():
        raise FileExistsError(RUN)
    all_ids = {int(path.stem.rsplit('_', 1)[-1]) for path in (COCO / 'val2014').glob('COCO_val2014_*.jpg')}
    if len(all_ids) != 40504:
        raise ValueError(f'Incomplete COCO val2014: {len(all_ids)} files')
    excludes, sources = set(), []

    def add(path, parser):
        contents = path.read_bytes()
        if not contents.strip():
            return
        if path.suffix == '.jsonl':
            ids = set()
            for line in contents.splitlines():
                if line.strip():
                    ids.update(parser(json.loads(line)))
        else:
            ids = parser(json.loads(contents))
        matched = ids & all_ids
        excludes.update(matched)
        sources.append({'path': str(path), 'file_sha256': sha(contents),
                        'in_val2014': len(matched), 'matched_ids_sha256': ids_sha(matched)})

    def caption_keys(value):
        if not isinstance(value, dict) or not all(str(k).isdigit() for k in value):
            raise ValueError('Expected COCO captions keyed by image ID')
        return set(map(int, value))

    for root in (REPO.parent / 'ic/outputs', OUT):
        for path in sorted(root.glob('*/COCO*/generations.json')):
            add(path, caption_keys)
    for path in sorted((REPO.parent / 'dataset/pope').glob('coco_pope_*.json')):
        # The POPE files are JSONL despite their .json suffix.
        body = path.read_bytes()
        ids = set().union(*(record_ids(json.loads(line)) for line in body.splitlines() if line.strip())) & all_ids
        excludes.update(ids)
        sources.append({'path': str(path), 'file_sha256': sha(body), 'in_val2014': len(ids),
                        'matched_ids_sha256': ids_sha(ids)})
    add(REPO.parent / 'dataset/qa_sam_svar_base/pope/questions.jsonl', record_ids)
    for folder in sorted(OUT.glob('generation_time_*')):
        manifest = folder / 'cohort.json'
        if manifest.is_file():
            add(manifest, cohort_ids)
        for path in sorted(folder.glob('*/*.shard*.jsonl')):
            if not path.is_symlink():
                add(path, record_ids)
    # The historical GPU preflights used these two photos outside some manifests.
    excludes.update({485532, 558955} & all_ids)
    matched = sorted(all_ids - excludes)
    images = sorted(np.random.default_rng(SEED).choice(matched, COUNT, replace=False).tolist())
    assert len(set(images)) == COUNT and not set(images) & excludes

    hashes = {p: sha((REPO / p).read_bytes()) for p in SOURCES}
    previous = OUT / 'generation_time_evidence_drift_qwen3_signed_feedback_alternative-500/protocol.json'
    previous_protocol = read(previous)
    for path in SOURCES:
        expected = previous_protocol['source_sha256'].get(str(REPO / path))
        if expected is not None and path != 'scripts/run_qwen3_signed_feedback.py':
            assert expected == hashes[path], ('Previously used source changed', path)
    assert previous_protocol['source_sha256'][str(REPO / 'features/qwen3_sar_ems_alternative.py')] == hashes['features/qwen3_sar_ems_alternative.py']

    RUN.mkdir()
    (RUN / 'logs').mkdir()
    (RUN / 'coco500').mkdir()
    write(RUN / 'cohort.json', {'coco500': images})
    selection = {'seed': SEED, 'rng': 'NumPy PCG64', 'pool_size': len(matched), 'excluded_ids': sorted(excludes),
                 'sources': sources, 'sample_sha256': ids_sha(images), 'available_images': len(all_ids),
                 'scope': 'COCO IDs absent from enumerated caption outputs, prior generation cohorts, POPE and diagnostic IDs.'}
    write(RUN / 'selection.json', selection)
    write(RUN / 'protocol.json', {
        'model': 'Qwen3-VL-8B-Instruct', 'variants': VARIANTS, 'prompt': 'Describe this image.',
        'seed': SEED, 'image_count': COUNT, 'max_new_tokens': CAP, 'max_batch': BATCH,
        'shards': 2, 'generation': 'BF16 eager greedy; sorted ID stride-2 shards; common realized batches',
        'candidate': 'Frozen alternative entity evidence recipe at layers 24->25 / 25->26 (zero-based)',
        'prior_protocol': str(previous), 'prior_protocol_sha256': sha(previous.read_bytes()),
        'sample_sha256': ids_sha(images), 'source_sha256': hashes,
        'primary': 'Sentence CHAIR; candidate vs SADI. Publish both parsers, all methods, paired intervals and mention counts.',
    })
    print('FROZEN', COUNT, 'excluded', len(excludes), 'sources', len(sources), 'pool', len(matched), ids_sha(images), flush=True)


def worker(shard):
    from unittest.mock import patch

    import torch

    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_sar_ems_alternative import Qwen3BatchSupportedAlternative
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as shared
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, MODEL, MODEL_ROOT
    from utils.config_utils import load_config, get_extraction_model_cfg

    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config['hf_name'] = str(MODEL_ROOT)
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    assert nouns
    base_generate = shared.generate_group

    def candidate_factory(model, span, nouns, mode, strength):
        assert mode == 'next'
        return Qwen3BatchSupportedAlternative(model, span, nouns, strength=strength)

    def generate(wrapper, span, group, nouns, method, cap):
        if method in (SADI, SVAR):
            return base_generate(wrapper, span, group, nouns, method, cap)
        if method == NEXT:
            return original.generate_group(wrapper, span, group, nouns, NEXT, cap)
        assert method == CANDIDATE
        with patch.object(original, 'Qwen3BatchLayerFeedback', candidate_factory):
            return original.generate_group(wrapper, span, group, nouns, NEXT, cap)

    ids = read(RUN / 'cohort.json')['coco500'][shard::2]
    files = {name: RUN / 'coco500' / f'{name}.shard{shard}.jsonl' for name in VARIANTS}
    if any(path.exists() for path in files.values()):
        raise FileExistsError('Existing output; inspect pairing before any retry')
    existing = {name: set() for name in VARIANTS}
    with torch.inference_mode(), patch.object(shared, 'generate_group', generate):
        for span, group in shared.same_span_batches(wrapper, ids, COCO, BATCH):
            shared.write_group(wrapper, span, group, nouns, existing, files, CAP, shard)
    assert all(rows == set(ids) for rows in existing.values())


def audit_batches():
    ids = read(RUN / 'cohort.json')['coco500']
    protocol = read(RUN / 'protocol.json')
    assert len(set(ids)) == COUNT and ids_sha(ids) == protocol['sample_sha256']
    assert set(ids).isdisjoint(read(RUN / 'selection.json')['excluded_ids'])
    assert all(sha((REPO / p).read_bytes()) == digest for p, digest in protocol['source_sha256'].items())
    caption_hashes, plans = {}, {}
    for shard in range(2):
        shard_plans = []
        for name in VARIANTS:
            path = RUN / 'coco500' / f'{name}.shard{shard}.jsonl'
            data = path.read_bytes()
            caption_hashes[path.name] = sha(data)
            rows = [json.loads(line) for line in data.splitlines()]
            assert len(rows) == COUNT // 2 and {r['image_id'] for r in rows} == set(ids[shard::2])
            cursor, groups = 0, []
            while cursor < len(rows):
                first = rows[cursor]
                n = first['batch_size']
                group = rows[cursor:cursor + n]
                assert 0 < n <= BATCH and len(group) == n
                assert all(row['variant'] == name and 0 < len(row['response_token_ids']) <= CAP
                           and row['batch_size'] == n and row['visual_token_count'] == first['visual_token_count']
                           and row['batch_wall_seconds'] == first['batch_wall_seconds'] for row in group)
                groups.append({'ids': [row['image_id'] for row in group], 'vision': first['visual_token_count']})
                cursor += n
            shard_plans.append(groups)
        assert all(groups == shard_plans[0] for groups in shard_plans[1:]), f'Unpaired shard {shard}'
        plans[str(shard)] = shard_plans[0]
    write(RUN / 'batch_audit.json', {'images': COUNT, 'caption_sha256': caption_hashes, 'paired_batches': plans})


def evaluate():
    audit_batches()
    for parser in ('sentence', 'legacy'):
        subprocess.run([sys.executable, str(REPO / 'scripts/evaluate_caption_chair.py'),
                        '--evaluator', parser, '--generation-root', str(RUN), '--cohort', 'coco500',
                        '--limit', str(COUNT), '--bootstrap-draws', '5000', '--variants', *VARIANTS],
                       cwd=REPO, check=True)


def run():
    prepare()
    children = []
    started = time.time()
    try:
        for shard in range(2):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                       NLTK_DATA=str(OUT / 'generation_time_evidence_drift_llava/nltk_data'), PYTHONUNBUFFERED='1')
            with (RUN / f'logs/gpu{shard}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', str(shard)], cwd=REPO,
                                                env=env, stdout=log, stderr=subprocess.STDOUT))
        write(RUN / 'run_status.json', {'status': 'generating', 'pid': os.getpid(),
                                       'workers': [p.pid for p in children], 'started': started})
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Generation worker failed')
            time.sleep(3)
        if any(p.returncode for p in children):
            raise RuntimeError('Generation worker failed')
        generated = time.time()
        write(RUN / 'run_status.json', {'status': 'evaluating', 'pid': os.getpid(), 'started': started, 'generated': generated})
        evaluate()
        write(RUN / 'run_status.json', {'status': 'complete', 'started': started, 'generated': generated, 'finished': time.time()})
    except BaseException as exc:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        for proc in children:
            proc.wait()
        write(RUN / 'run_status.json', {'status': 'failed', 'error': str(exc), 'started': started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
    elif args.evaluate:
        evaluate()
    else:
        run()
