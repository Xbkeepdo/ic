"""Replicate the unchanged first-class candidate on another 500 unused photos."""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_fresh_first_class500 as original

fresh = original.fresh
REFERENCE = fresh.OUT / 'generation_time_evidence_drift_qwen3_fresh_first_class500'
RUN = fresh.OUT / 'generation_time_evidence_drift_qwen3_replication_first_class500'
TRAIN = REPO.parent / 'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b/compact_features.npz'


def configure():
    original.RUN = RUN
    original.configure()
    fresh.SEED = 20261120
    fresh.__file__ = __file__
    fresh.SOURCES += ('scripts/run_qwen3_replication_first_class500.py',)
    prepare_original = fresh.prepare

    def prepare():
        reference = fresh.read(REFERENCE / 'protocol.json')
        assert fresh.read(REFERENCE / 'run_status.json')['status'] == 'complete'
        assert fresh.read(REFERENCE / 'first_class_runtime_audit.json')['passed']
        assert fresh.read(REFERENCE / 'independent_summary.json')['status'] == 'passed'
        # No model, attention, probe, vocabulary, or decoding source may change.
        for relative, expected in reference['source_sha256'].items():
            assert fresh.sha((REPO / relative).read_bytes()) == expected, relative
        prepare_original()
        plan = fresh.read(RUN / 'protocol.json')
        assert plan['variants'] == reference['variants']
        for key in ('recipe', 'max_new_tokens', 'max_batch', 'model', 'prompt'):
            assert plan[key] == reference[key], key
        ids = set(fresh.read(RUN / 'cohort.json')['coco500'])
        previous_ids = set(fresh.read(REFERENCE / 'cohort.json')['coco500'])
        development = set(fresh.read(original.PILOT / 'cohort.json')['coco100'])
        with np.load(TRAIN, allow_pickle=False) as data:
            training_ids = set(map(int, data['image_id']))
        assert ids.isdisjoint(previous_ids | development | training_ids)
        plan.update(
            replication_of=str(REFERENCE),
            replication_protocol_sha256=fresh.sha((REFERENCE / 'protocol.json').read_bytes()),
            prospective_rule='Unchanged candidate, no sign gate, no parameter search; both CHAIR metrics vs SADI are primary. Report all500 and both evaluators regardless of outcome.',
            reason='First independent500 improved both CHAIR point estimates; CHAIRi interval nearly touches zero. Additional sign gate lost most benefit. Replicate the original candidate without retuning.',
            advancement='Frozen replication of the original first-class candidate on a second disjoint500; not development or threshold selection.',
        )
        for path in (REFERENCE / 'protocol.json', REFERENCE / 'cohort.json', TRAIN):
            plan['source_sha256'][str(path)] = fresh.sha(path.read_bytes())
        fresh.write(RUN / 'protocol.json', plan)
        fresh.write(RUN / 'cohort_independence.json', dict(
            passed=True, images=len(ids), old_probe_images=len(training_ids),
            overlap_old_probe_images=len(ids & training_ids),
            overlap_development100=len(ids & development),
            overlap_first_independent500=len(ids & previous_ids),
            excluded_images=len(fresh.read(RUN / 'selection.json')['excluded_ids']),
            sample_sha256=plan['sample_sha256'], source_count=len(plan['source_sha256']),
        ))
        print('REPLICATION_PREPARED', plan['sample_sha256'], flush=True)

    fresh.prepare = prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    configure()
    if args.worker is not None:
        original.worker(args.worker)
    elif args.evaluate:
        fresh.evaluate()
    else:
        fresh.run()
