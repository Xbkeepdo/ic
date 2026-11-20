"""Test frozen per-head feedback on object tokens seen in the old training set."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-vocabulary-pilot')
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-pilot')
TRAIN = REPO.parent / 'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b/compact_features.npz'
NAME = 'sadi_head_training_vocabulary_31'
BROAD = 'sadi_head_final_gradient_31'


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    cached = shared.read(SOURCE / 'protocol.json')
    new_files = ('features/qwen3_sar_ems_head_gradient.py', 'features/qwen3_sar_ems_final_target.py',
                 'features/qwen3_sar_ems_removal_gate.py', 'features/qwen3_visual_write_contrast.py',
                 'scripts/run_qwen3_object_vocabulary_pilot.py', 'scripts/run_qwen3_signed_feedback.py',
                 'scripts/evaluate_caption_chair.py', 'scripts/evaluate_generation_time_evidence_drift_llava.py',
                 'scripts/evaluate_qwen3_next200_sentence_tokenized.py', 'coco-labeling/coco_chair.py')
    hashes = {str(REPO / name): shared.sha(REPO / name)
              for name in (*shared.generation_sources(), *new_files)}
    for name, digest in hashes.items():
        prior = cached['source_sha256'].get(name)
        if prior is not None and name != str(REPO / 'scripts/run_qwen3_signed_feedback.py'):
            assert digest == prior, name
    data = np.load(TRAIN, allow_pickle=False)
    vocabulary = sorted(set(data['target_token_id'][data['train_mask']].tolist()))
    assert len(vocabulary) > 0
    hashes[str(TRAIN)] = shared.sha(TRAIN)
    ids = shared.read(SOURCE / 'cohort.json')['coco100']
    assert len(set(ids)) == 100 and not set(ids).intersection(data['image_id'].tolist())
    OUT.mkdir()
    (OUT / 'coco100').mkdir()
    (OUT / 'logs').mkdir()
    for shard in (0, 1):
        controls = [SOURCE / 'coco100' / f'{name}.shard{shard}.jsonl' for name in ('sadi', BROAD)]
        assert shared.group_keys(shared.source_groups(controls[0])) == shared.group_keys(shared.source_groups(controls[1]))
        for path in controls:
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT / 'coco100' / path.name)
    shared.write(OUT / 'training_vocabulary.json', dict(token_ids=vocabulary, source=str(TRAIN),
                                                       source_sha256=hashes[str(TRAIN)], split='train_mask only'))
    hashes[str(OUT / 'training_vocabulary.json')] = shared.sha(OUT / 'training_vocabulary.json')
    shared.write(OUT / 'cohort.json', {'coco100': ids})
    fresh = shared.OUT / 'generation_time_evidence_drift_qwen3_fresh_head500'
    fresh_status = shared.read(fresh / 'run_status.json')
    recipe = dict(cached['recipe'], candidate_scope='WordNet physical nouns intersect old training object token IDs.')
    recipe['gradient_scope'] = {NAME: cached['recipe']['gradient_scope'][BROAD]}
    protocol = dict(source=str(SOURCE), stage='head-gradient-vocabulary-pilot', cohort='coco100',
                    images=100, shards=2, max_new_tokens=512, model='Qwen3-VL-8B-Instruct',
                    variants=['sadi', BROAD, NAME], candidates=[NAME], cached_controls=['sadi'],
                    copied_controls=['sadi', BROAD], source_sha256=hashes,
                    batch_policy='Replay the original cached batches and GPU assignment.',
                    recipe=recipe,
                    selection='Fixed vocabulary-only variant. Old training targets supply token identities, including both labels.',
                    defined_during_fresh_validation=fresh_status['status'],
                    fresh_validation_sample_sha256=shared.read(fresh / 'protocol.json')['sample_sha256'])
    shared.write(OUT / 'protocol.json', protocol)
    print('FROZEN_OBJECT_VOCABULARY', len(vocabulary), flush=True)


def worker(gpu):
    from features import qwen3_sar_ems_head_gradient as head
    original = head.Qwen3BatchHeadGradient
    allowed = frozenset(shared.read(OUT / 'training_vocabulary.json')['token_ids'])

    def make_controller(model, span, nouns, **kwargs):
        return original(model, span, nouns & allowed, **kwargs)

    with patch.object(shared, 'HEAD_FINAL', NAME), patch.object(head, 'Qwen3BatchHeadGradient', make_controller):
        shared.worker(OUT, gpu)


def run():
    prepare()
    queued = started = time.time()
    children = []
    try:
        shared.write(OUT / 'run_status.json', dict(status='waiting_for_gpus', pid=os.getpid(), queued=queued))
        while True:
            used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if len(used.splitlines()) == 2 and max(map(int, used.splitlines())) < 100:
                break
            time.sleep(10)
        started = time.time()
        for gpu in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', NLTK_DATA=str(shared.OUT / 'generation_time_evidence_drift_llava/nltk_data'))
            with (OUT / f'logs/gpu{gpu}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', str(gpu)], cwd=REPO,
                                                  env=env, stdout=log, stderr=subprocess.STDOUT))
        shared.write(OUT / 'run_status.json', dict(status='generating', pid=os.getpid(),
                                                  pids=[p.pid for p in children], queued=queued, started=started))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Object-vocabulary worker failed')
            time.sleep(3)
        if any(p.returncode for p in children):
            raise RuntimeError('Object-vocabulary worker failed')
        shared.write(OUT / 'run_status.json', dict(status='evaluating', pid=os.getpid(), started=started))
        protocol = shared.read(OUT / 'protocol.json')
        for shard in (0, 1):
            for name in protocol['copied_controls']:
                filename = f'{name}.shard{shard}.jsonl'
                assert (OUT / 'coco100' / filename).read_bytes() == (SOURCE / 'coco100' / filename).read_bytes()
        shared.evaluate(OUT, protocol)
        shared.write(OUT / 'run_status.json', dict(status='complete', queued=queued, started=started, finished=time.time()))
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        shared.write(OUT / 'run_status.json', dict(status='failed', error=str(exc), started=started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
