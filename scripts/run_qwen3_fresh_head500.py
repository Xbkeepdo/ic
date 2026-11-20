"""Validate the frozen per-head feedback on 500 unused COCO images."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_fresh_caption500 as fresh

CANDIDATE = 'sadi_head_final_gradient_31'
RUN = fresh.OUT / 'generation_time_evidence_drift_qwen3_fresh_head500'
PREVIOUS = fresh.OUT / 'generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-500/protocol.json'


def configure():
    fresh.RUN = RUN
    fresh.CANDIDATE = CANDIDATE
    fresh.VARIANTS = ('sadi', CANDIDATE, 'svar')
    fresh.SEED = 20261118
    fresh.SOURCES += (
        'features/qwen3_sar_ems_head_gradient.py', 'features/qwen3_sar_ems_final_target.py',
        'features/qwen3_sar_ems_removal_gate.py', 'features/qwen3_visual_write_contrast.py',
        'scripts/run_qwen3_fresh_head500.py', 'scripts/audit_qwen3_fresh_caption500.py',
        'scripts/audit_qwen3_head_feedback.py', 'scripts/run_qwen3_signed_feedback.py',
    )
    # The shared driver starts workers through its __file__ entry point.
    fresh.__file__ = str(Path(__file__).resolve())
    prepare_base = fresh.prepare

    def prepare():
        prior = fresh.read(PREVIOUS)
        assert CANDIDATE in prior['candidates']
        for name in fresh.SOURCES:
            recorded = prior['source_sha256'].get(str(REPO / name))
            if recorded is not None and name != 'scripts/run_qwen3_signed_feedback.py':
                assert fresh.sha((REPO / name).read_bytes()) == recorded, name
        prepare_base()
        # Diagnostic cohorts are also registered outside generation_time_*.
        from scripts.experiment_snapshots.qwen3_new200_b16.prepare import cohort_ids
        selection = fresh.read(RUN / 'selection.json')
        excluded = set(selection['excluded_ids'])
        available = {int(p.stem.rsplit('_', 1)[-1]) for p in
                     (fresh.COCO / 'val2014').glob('COCO_val2014_*.jpg')}
        for path in sorted(fresh.OUT.glob('qwen3_*/cohort.json')):
            data = path.read_bytes()
            ids = cohort_ids(fresh.read(path)) & available
            excluded.update(ids)
            selection['sources'].append(dict(path=str(path), file_sha256=fresh.sha(data),
                                              kind='diagnostic_cohort', in_val2014=len(ids),
                                              matched_ids_sha256=fresh.ids_sha(ids)))
        pool = sorted(available - excluded)
        ids = sorted(np.random.default_rng(fresh.SEED).choice(pool, fresh.COUNT, replace=False).tolist())
        assert len(set(ids)) == 500 and not set(ids).intersection(excluded)
        digest = fresh.ids_sha(ids)
        selection.update(excluded_ids=sorted(excluded), pool_size=len(pool), sample_sha256=digest)
        fresh.write(RUN / 'selection.json', selection)
        fresh.write(RUN / 'cohort.json', {'coco500': ids})
        protocol = fresh.read(RUN / 'protocol.json')
        protocol.update(candidate_method=CANDIDATE, candidates=[CANDIDATE], cohort='coco500', images=500,
                        candidate='Frozen layer-31 per-head final-target gradient with FFN spatial drift and probability-decrease acceptance.',
                        recipe=prior['recipe'], prior_protocol=str(PREVIOUS),
                        prior_protocol_sha256=fresh.sha(PREVIOUS.read_bytes()), sample_sha256=digest)
        fresh.write(RUN / 'protocol.json', protocol)
        print('FROZEN_HEAD_COHORT', digest, 'pool', len(pool), 'excluded', len(excluded), flush=True)

    fresh.prepare = prepare


def worker(shard):
    from features import qwen3_sar_ems_alternative as alternative
    from features.qwen3_sar_ems_head_gradient import Qwen3BatchHeadGradient

    def make_controller(model, span, nouns, strength):
        # The shared worker has the older candidate factory signature.
        assert strength == 5.0
        return Qwen3BatchHeadGradient(model, span, nouns, final_gradient=True)

    with patch.object(alternative, 'Qwen3BatchSupportedAlternative', make_controller):
        fresh.worker(shard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    configure()
    if args.worker is not None:
        worker(args.worker)
    elif args.evaluate:
        fresh.evaluate()
    else:
        fresh.run()
