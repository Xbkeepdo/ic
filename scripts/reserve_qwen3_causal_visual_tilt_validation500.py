"""Reserve a different500 before any causal-tilt pilot metrics are produced."""

import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from scripts import run_qwen3_block_acg_fresh500 as selector
from scripts.run_qwen3_signed_feedback import read, write, sha


def reserve(output, seed):
    pilot = REPO/'outputs/qwen3_causal_visual_tilt_pilot_20261119'
    preflight = REPO/'outputs/qwen3_causal_visual_tilt_preflight_20261119'
    previous = REPO/'outputs/qwen3_bounded_visual_tilt_validation500_20261119'
    assert not output.exists(), 'Do not redraw an existing reservation'
    assert not pilot.exists(), 'Reserve before the new pilot starts'
    assert read(preflight/'verification.json')['passed']
    assert read(previous/'final_provenance.json')['passed']
    for name, value in read(preflight/'request.json')['source_sha256'].items():
        assert sha(Path(name)) == value, name
    with patch.object(selector, 'SEED', seed):
        ids, selection = selector.select_images()
    assert not set(ids) & set(read(previous/'cohort.json')['coco500'])
    assert not set(ids) & set(read(REPO/'outputs/qwen3_block_acg_pilot_20261118/cohort.json')['coco100'])
    for source in selection['sources']:
        assert sha(Path(source['path'])) == source['file_sha256']
    files = ['scripts/reserve_qwen3_causal_visual_tilt_validation500.py',
             'scripts/run_qwen3_block_acg_fresh500.py',
             'scripts/experiment_snapshots/qwen3_new200_b16/prepare.py']
    output.mkdir()
    write(output/'cohort.json', dict(coco500=ids))
    write(output/'selection.json', selection)
    write(output/'reservation.json', dict(reserved_at=time.time(), reserve_only=True,
        neural_generation_started=False, before_causal_pilot_metrics=True,
        seed=seed, numpy_version=np.__version__, sample_sha256=selector.ids_sha(ids),
        planned_pilot=str(pilot),
        preflight_request_sha256=sha(preflight/'request.json'),
        preflight_verification_sha256=sha(preflight/'verification.json'),
        prior_validation_provenance_sha256=sha(previous/'final_provenance.json'),
        source_sha256={str(REPO/name): sha(REPO/name) for name in files},
        rule='Use only if a candidate merits validation. Keep this cohort unchanged; '
             'never redraw or choose parameters from its outputs. Consuming the reservation '
             'requires a separate consumed.json bound to the actual validation protocol.'))
    print(__import__('json').dumps(dict(reserved=True, seed=seed,
        sample_sha256=selector.ids_sha(ids), pool_size=selection['pool_size'],
        excluded=len(selection['excluded_ids']), sources=len(selection['sources']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=REPO/'outputs/qwen3_causal_visual_tilt_reserved500_20261119')
    parser.add_argument('--seed', type=int, default=2026111902)
    args = parser.parse_args()
    reserve(args.output, args.seed)
