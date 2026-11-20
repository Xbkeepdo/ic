"""Compare fixed and EMS-guided history routes on cached 100-image batches."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-history-route-pilot')
FIXED = 'sadi_history_route_fixed_31'
GUIDED = 'sadi_history_route_ems_31'


def prepare():
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'NAME', GUIDED):
        plan = repeat.prepare()
    for name in ('features/qwen3_sar_ems_history_route.py', 'scripts/run_qwen3_history_route_pilot.py'):
        plan['source_sha256'][str(REPO / name)] = shared.sha(REPO / name)
    plan.update(stage='head-gradient-history-route-pilot', candidates=[FIXED, GUIDED],
                variants=[*plan['copied_controls'], FIXED, GUIDED],
                selection='Prospective matched comparison of fixed history-route suppression and FFN-drift-guided suppression.',
                references=['https://arxiv.org/abs/2605.24024', 'https://arxiv.org/abs/2503.08342'])
    plan['recipe'] = dict(
        source_layer=31, intervention_layer=31, target_read_layer=35, base_layers=[5, 18],
        gradient_scope={name: 'source FFN and tail layers 32--35' for name in (FIXED, GUIDED)},
        objective='log probability of the original final candidate', support_threshold=.35,
        candidate_scope='WordNet physical nouns intersect old training object token IDs, repeats included.',
        history_scope='Generated output keys only; original visual, prompt and system route components remain fixed.',
        head_selection='At most four heads with negative visual effect, positive history effect, smallest visual reliance.',
        fixed_suppression=.5, guided_suppression='.5 + .5 * HellingerSquared(Q_negative, P_ffn)',
        tau=.03, acceptance='Final candidate probability strictly decreases after the proposal.',
        cache_policy='Frozen cached-prefix gradient probe; current tail KV replacement and rollback on rejection.',
        method_scope='Single-layer adaptation of route competition and history attention; not full CRG or AttnReal.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    from features import qwen3_sar_ems_head_gradient as head
    from features.qwen3_sar_ems_history_route import Qwen3BatchHistoryRoute
    from models import build_model
    import models

    allowed = frozenset(shared.read(OUT / 'training_vocabulary.json')['token_ids'])

    def controller(model, span, nouns, *, final_gradient):
        return Qwen3BatchHistoryRoute(model, span, nouns & allowed, use_ems=final_gradient)

    def checked_build(*args, **kwargs):
        wrapper = build_model(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        if gpu == 0:
            from features.llava_decoding_evidence import lexical_nouns
            from scripts import verify_qwen3_final_target_replay as check
            nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
            with patch.object(check, 'Qwen3BatchFinalTargetReplay', Qwen3BatchHistoryRoute):
                report = check.verify(wrapper, nouns)
            shared.write(OUT / 'route_preflight.json', report)
        return wrapper

    with patch.object(shared, 'HEAD_FINAL', GUIDED), patch.object(shared, 'HEAD_SOURCE', FIXED), \
            patch.object(head, 'Qwen3BatchHeadGradient', controller), patch.object(models, 'build_model', checked_build):
        shared.worker(OUT, gpu)


def run():
    plan = prepare()
    queued = started = time.time()
    children = []
    try:
        shared.write(OUT / 'run_status.json', dict(status='waiting_for_gpus', pid=os.getpid(), queued=queued))
        while True:
            used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            if len(used.splitlines()) == 2 and max(map(int, used.splitlines())) < 100:
                break
            time.sleep(10)
        started = time.time()
        for gpu in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', NLTK_DATA=str(shared.OUT / 'generation_time_evidence_drift_llava/nltk_data'))
            with (OUT / f'logs/gpu{gpu}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable, __file__, '--worker', str(gpu)],
                                                  cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        shared.write(OUT / 'run_status.json', dict(status='generating', pid=os.getpid(),
                                                  pids=[p.pid for p in children], queued=queued, started=started))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('History-route worker failed')
            time.sleep(3)
        assert all(p.returncode == 0 for p in children)
        shared.write(OUT / 'run_status.json', dict(status='evaluating', pid=os.getpid(), started=started))
        shared.evaluate(OUT, plan)
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
