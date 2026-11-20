"""Reserve another unseen500 before any mass-matched development metrics."""

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
    pilot = REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
    preflight = REPO/'outputs/qwen3_mass_matched_visual_preflight_20261119'
    previous = REPO/'outputs/qwen3_causal_visual_tilt_validation500_20261119'
    assert not output.exists(), 'Do not redraw an existing reservation'
    assert not pilot.exists(), 'Reserve before new pilot metrics'
    assert read(preflight/'verification.json')['passed']
    assert read(previous/'final_provenance.json')['passed']
    assert read(previous/'human_review.json')['no_more_validation_for_unchanged_fixed_candidate']
    for name, value in read(preflight/'request.json')['source_sha256'].items():
        assert sha(Path(name)) == value, name
    with patch.object(selector, 'SEED', seed):
        ids, selection = selector.select_images()
    assert not set(ids) & set(read(previous/'cohort.json')['coco500'])
    assert not set(ids) & set(read(REPO/'outputs/qwen3_causal_visual_tilt_pilot_20261119/cohort.json')['coco100'])
    for source in selection['sources']:
        assert sha(Path(source['path'])) == source['file_sha256']
    files = ['scripts/reserve_qwen3_mass_matched_visual_validation500.py',
        'scripts/run_qwen3_block_acg_fresh500.py',
        'scripts/experiment_snapshots/qwen3_new200_b16/prepare.py',
        'scripts/verify_qwen3_global_batch_planner.py']
    planner = REPO/'outputs/qwen3_global_batch_planner_cpu_20261119/verification.json'
    assert read(planner)['passed'] and read(planner)['old_batch_boundaries_and_visual_counts_exact']
    output.mkdir()
    write(output/'cohort.json', dict(coco500=ids))
    write(output/'selection.json', selection)
    write(output/'reservation.json', dict(reserved_at=time.time(), reserve_only=True,
        neural_generation_started=False, before_mass_matched_pilot_metrics=True,
        seed=seed, numpy_version=np.__version__, sample_sha256=selector.ids_sha(ids),
        planned_pilot=str(pilot), preflight_request_sha256=sha(preflight/'request.json'),
        preflight_verification_sha256=sha(preflight/'verification.json'),
        prior_validation_provenance_sha256=sha(previous/'final_provenance.json'),
        batch_policy='For prospective validation only: global same-span/length batches, max16, balanced number of groups; all methods generated fresh with one frozen manifest. Old100 retains its cached original groups.',
        batch_planner_verification_sha256=sha(planner),
        source_sha256={str(REPO/name):sha(REPO/name) for name in files},
        rule='Use only if development yields useful evidence. Keep cohort unchanged. A separate consumed.json must bind the actual validation protocol; no redraw or validation tuning.'))
    print(__import__('json').dumps(dict(reserved=True, seed=seed, sample_sha256=selector.ids_sha(ids),
        pool_size=selection['pool_size'], excluded=len(selection['excluded_ids']), sources=len(selection['sources']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO/'outputs/qwen3_mass_matched_visual_reserved500_20261119')
    parser.add_argument('--seed', type=int, default=2026111903)
    args = parser.parse_args(); reserve(args.output, args.seed)
