"""One-change repeat-eligibility pilot on exactly cached caption batches."""

import argparse
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

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-repeat-pilot')
SOURCE = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-vocabulary-pilot')
PREVIOUS = 'sadi_head_training_vocabulary_31'
NAME = 'sadi_head_training_vocabulary_repeated_31'


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    cached = shared.read(SOURCE / 'protocol.json')
    # Preserve every dependency of the cached reference and the two new files.
    hashes = dict(cached['source_sha256'])
    for filename, expected in hashes.items():
        assert shared.sha(Path(filename)) == expected, filename
    for filename in ('features/qwen3_sar_ems_repeat_guidance.py', 'scripts/run_qwen3_repeat_guidance.py'):
        hashes[str(REPO / filename)] = shared.sha(REPO / filename)
    ids = shared.read(SOURCE / 'cohort.json')['coco100']
    assert len(ids) == len(set(ids)) == 100
    OUT.mkdir()
    (OUT / 'coco100').mkdir()
    (OUT / 'logs').mkdir()
    controls = ('sadi', PREVIOUS)
    for shard in (0, 1):
        paths = [SOURCE / 'coco100' / f'{name}.shard{shard}.jsonl' for name in controls]
        assert shared.group_keys(shared.source_groups(paths[0])) == shared.group_keys(shared.source_groups(paths[1]))
        for path in paths:
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT / 'coco100' / path.name)
    shutil.copyfile(SOURCE / 'training_vocabulary.json', OUT / 'training_vocabulary.json')
    hashes[str(OUT / 'training_vocabulary.json')] = shared.sha(OUT / 'training_vocabulary.json')
    shared.write(OUT / 'cohort.json', {'coco100': ids})
    recipe = dict(cached['recipe'], exclude_prior_token_id=False)
    recipe['gradient_scope'] = {NAME: cached['recipe']['gradient_scope'][PREVIOUS]}
    plan = dict(source=str(SOURCE), stage='head-gradient-repeat-pilot', cohort='coco100',
                images=100, shards=2, max_new_tokens=512, model='Qwen3-VL-8B-Instruct',
                variants=[*controls, NAME], candidates=[NAME], cached_controls=['sadi'],
                copied_controls=list(controls), source_sha256=hashes, recipe=recipe,
                batch_policy='Replay the original cached batches and GPU assignment.',
                selection='Change only repeat eligibility; keep vocabulary, thresholds, strengths and acceptance frozen.',
                source_protocol_sha256=shared.sha(SOURCE / 'protocol.json'))
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    from features import qwen3_sar_ems_head_gradient as head
    from features.qwen3_sar_ems_repeat_guidance import Qwen3BatchRepeatedObjectGuidance
    allowed = frozenset(shared.read(OUT / 'training_vocabulary.json')['token_ids'])

    def factory(model, span, nouns, **kwargs):
        return Qwen3BatchRepeatedObjectGuidance(model, span, nouns & allowed, **kwargs)

    with patch.object(shared, 'HEAD_FINAL', NAME), patch.object(head, 'Qwen3BatchHeadGradient', factory):
        shared.worker(OUT, gpu)


def run():
    plan = prepare()
    started = time.time()
    children = []
    try:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        assert len(used.splitlines()) == 2 and max(map(int, used.splitlines())) < 100, 'GPUs occupied'
        for gpu in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', NLTK_DATA=str(shared.OUT / 'generation_time_evidence_drift_llava/nltk_data'))
            with (OUT / f'logs/gpu{gpu}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', str(gpu)],
                                                  cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        shared.write(OUT / 'run_status.json', dict(status='generating', pid=os.getpid(),
                                                  pids=[p.pid for p in children], started=started))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Repeat-guidance worker failed')
            time.sleep(3)
        assert all(p.returncode == 0 for p in children)
        shared.write(OUT / 'run_status.json', dict(status='evaluating', pid=os.getpid(), started=started))
        for shard in (0, 1):
            for name in plan['copied_controls']:
                filename = f'{name}.shard{shard}.jsonl'
                assert (OUT / 'coco100' / filename).read_bytes() == (SOURCE / 'coco100' / filename).read_bytes()
        shared.evaluate(OUT, plan)
        shared.write(OUT / 'run_status.json', dict(status='complete', started=started, finished=time.time()))
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
