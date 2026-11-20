"""Reserve one unused COCO500 cohort before reading bounded-tilt pilot scores."""

import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from scripts import run_qwen3_block_acg_fresh500 as selector
from scripts.run_qwen3_signed_feedback import read,write,sha


def reserve(output,seed):
    pilot=REPO/'outputs/qwen3_bounded_visual_tilt_pilot_20261119'
    assert not output.exists(),'Do not redraw an existing reserved cohort'
    assert not (pilot/'evaluations').exists(),'Reserve before pilot metric inspection'
    plan=read(pilot/'protocol.json');assert plan['images']==100
    with patch.object(selector,'SEED',seed):ids,selection=selector.select_images()
    prior=REPO/'outputs/qwen3_block_acg_fresh500_20261118'
    assert not set(ids)&set(read(prior/'cohort.json')['coco500'])
    assert not set(ids)&set(read(pilot/'cohort.json')['coco100'])
    for info in selection['sources']:assert sha(Path(info['path']))==info['file_sha256']
    sources=['scripts/reserve_qwen3_mitigation_validation500.py','scripts/run_qwen3_block_acg_fresh500.py',
        'scripts/experiment_snapshots/qwen3_new200_b16/prepare.py']
    output.mkdir()
    write(output/'cohort.json',dict(coco500=ids));write(output/'selection.json',selection)
    write(output/'reservation.json',dict(reserved_at=time.time(),reserve_only=True,
        neural_generation_started=False,before_pilot_metrics=True,seed=seed,numpy_version=np.__version__,
        sample_sha256=selector.ids_sha(ids),pilot_protocol_sha256=sha(pilot/'protocol.json'),
        source_sha256={str(REPO/n):sha(REPO/n) for n in sources},
        rule='Use only if a candidate merits validation. Reuse this exact cohort without redrawing; no caption or metric has been generated for these500 images. Once exposed, it is no longer an unseen cohort.'))
    print(__import__('json').dumps(dict(reserved=True,sample_sha256=selector.ids_sha(ids),
        pool_size=selection['pool_size'],excluded=len(selection['excluded_ids']),sources=len(selection['sources']))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO/'outputs/qwen3_mitigation_reserved500_20261119')
    p.add_argument('--seed',type=int,default=2026111901)
    args=p.parse_args();reserve(args.output,args.seed)
