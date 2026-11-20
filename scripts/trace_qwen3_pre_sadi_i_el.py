"""Reuse E_l caches to check a pre-SADI local-attention I readout."""

import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import trace_qwen3_full_i_el as original
from scripts import run_qwen3_signed_feedback as shared

RUN = shared.OUT / 'qwen3_sadi_pre_attention_i_el_trace'
SOURCE = original.RUN


def prepare():
    assert not RUN.exists()
    assert shared.read(SOURCE / 'status.json')['status'] == 'complete'
    previous = shared.read(SOURCE / 'protocol.json')
    for name, expected in previous['source_sha256'].items():
        assert shared.sha(Path(name)) == expected, name
    plan = dict(previous, purpose='Frozen original probe on local pre-SADI attention I with actual SADI states; reuse E_l.',
                source_trace=str(SOURCE), readout='Replace I only on SADI layers 5--18 using unedited local attention.')
    plan['source_sha256'] = dict(previous['source_sha256'])
    for name in ('features/qwen3_pre_sadi_i_el.py', 'scripts/trace_qwen3_pre_sadi_i_el.py'):
        plan['source_sha256'][str(REPO / name)] = shared.sha(REPO / name)
    plan['source_sha256'][str(SOURCE / 'protocol.json')] = shared.sha(SOURCE / 'protocol.json')
    for path in (SOURCE / 'semantics').glob('*.npz'):
        plan['source_sha256'][str(path)] = shared.sha(path)
    RUN.mkdir()
    for directory in ('logs', 'server_sources'):
        (RUN / directory).mkdir()
    # The parent worker serializes the sums again. Keep its destination
    # separate so the frozen input caches remain untouched.
    shutil.copytree(SOURCE / 'semantics', RUN / 'semantics')
    for name in ('cohort.json', 'training_vocabulary.json'):
        shutil.copyfile(SOURCE / name, RUN / name)
    for filename in plan['source_sha256']:
        path = Path(filename)
        if path.is_relative_to(REPO) and path.suffix in ('.py', '.yaml', '.yml'):
            target = RUN / 'server_sources' / path.relative_to(REPO)
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(path, target)
    shared.write(RUN / 'protocol.json', plan)


def worker(shard):
    import numpy as np
    import torch
    from features import qwen3_online_i_el as online
    from features.qwen3_pre_sadi_i_el import PreSadiIElCapture
    plan = shared.read(RUN / 'protocol.json')
    index = 0

    def factory(model, span, token_ids, **kwargs):
        nonlocal index
        if kwargs.get('validate'):
            return PreSadiIElCapture(model, span, token_ids, **kwargs)
        path = SOURCE / f'semantics/shard{shard}.batch{index}.npz'
        with np.load(path, allow_pickle=False) as cache:
            assert cache['image_ids'].tolist() == plan['groups'][shard][index]['ids']
            assert cache['span'].tolist() == list(span)
            assert cache['token_ids'].tolist() == sorted(set(token_ids))
            values = torch.from_numpy(cache['E_l'].copy())
        index += 1
        return PreSadiIElCapture(model, span, token_ids, cached_semantics=values, **kwargs)

    with patch.object(original, 'RUN', RUN), patch.object(online, 'OnlineIElCapture', factory):
        original.worker(shard)
    assert index == len(plan['groups'][shard])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(original, 'RUN', RUN), patch.object(original, 'prepare', prepare), \
                patch.object(original, '__file__', __file__):
            original.run()
    else:
        worker(args.worker)
