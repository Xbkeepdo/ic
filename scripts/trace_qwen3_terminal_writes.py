"""Observe final-layer visual writes on the frozen SADI object prefixes."""

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import trace_qwen3_visual_removal as trace

RUN = trace.OUT / 'qwen3_sadi_terminal_write_trace'


def worker(shard):
    from features import qwen3_sar_ems_layer_feedback as feedback

    class TerminalObserver(feedback.Qwen3BatchLayerFeedback):
        @contextmanager
        def capture(self):
            # The observer returns no budget. Use the existing same-layer
            # capture path so the final decoder layer has no successor.
            self.mode = 'same'
            self.targets = self.sources
            with super().capture():
                yield

    trace.RUN = RUN
    trace.LAYERS = (35,)
    with patch.object(feedback, 'Qwen3BatchLayerFeedback', TerminalObserver):
        trace.worker(shard)


def run():
    assert trace.load(trace.LABELS / 'status.json')['status'] == 'complete'
    if RUN.exists():
        raise FileExistsError(RUN)
    plan = trace.load(trace.LABELS / 'protocol.json')
    sources = [trace.LABELS / name for name in
               ('protocol.json', 'trace.shard0.jsonl', 'trace.shard1.jsonl')]
    sources += [Path(__file__), Path(trace.__file__),
                REPO / 'features/qwen3_sar_ems_layer_feedback.py',
                REPO / 'features/qwen3_visual_write_contrast.py']
    plan.update(contrast_layers=[35],
                contrast_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources},
                interpretation='Observe final-layer visual-write removal and FFN gain on cached prefixes.')
    RUN.mkdir()
    trace.write(RUN / 'protocol.json', plan)
    trace.write(RUN / 'cohort.json', trace.load(trace.LABELS / 'cohort.json'))
    children = []
    try:
        for shard in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard),
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       NLTK_DATA=str(trace.OUT / 'generation_time_evidence_drift_llava/nltk_data'))
            with (RUN / f'gpu{shard}.log').open('w') as log:
                children.append(subprocess.Popen(
                    [sys.executable, __file__, '--worker', str(shard)],
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        trace.write(RUN / 'status.json', dict(status='running', driver=os.getpid(),
                                             workers=[p.pid for p in children]))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Terminal trace worker failed')
            time.sleep(2)
        if any(p.returncode for p in children):
            raise RuntimeError('Terminal trace worker failed')
        for path, digest in plan['contrast_sources'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        trace.write(RUN / 'status.json', {'status': 'complete'})
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        trace.write(RUN / 'status.json', dict(status='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
