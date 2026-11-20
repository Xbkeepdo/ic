"""Validate the frozen SVAR transfer on another500 previously unused images."""

import argparse
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_fresh_first_class500 as original

fresh = original.fresh
NAME = 'svar_probe_first_class_recycle_19'
RUN = fresh.OUT / 'generation_time_evidence_drift_qwen3_fresh_svar_first_class500'
PILOT = fresh.OUT / 'generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-svar-first-class-pilot'


def configure():
    original.RUN, original.PILOT, original.NAME = RUN, PILOT, NAME
    original.configure()
    fresh.SEED = 20261121
    fresh.__file__ = __file__
    fresh.SOURCES += ('features/qwen3_probe_svar_first_class.py',
        'scripts/run_qwen3_fresh_svar_first_class500.py', 'scripts/audit_qwen3_first_class500_with_baseline.py',
        'scripts/audit_qwen3_svar_first_class_v2.py',
        str(PILOT / 'screening.json'), str(PILOT / 'first_class_audit.json'),
        str(PILOT / 'svar_recycle_event_audit.json'), str(PILOT / 'svar_audit_compatibility.json'),
        str(PILOT / 'svar_semantic_reuse_preflight.json'))
    prepare_original, semantic_preflight = fresh.prepare, original.preflight

    def prepare():
        assert fresh.read(PILOT / 'screening.json')['advance_to_500'] == [NAME]
        assert fresh.read(PILOT / 'svar_recycle_event_audit.json')['passed']
        assert fresh.read(PILOT / 'svar_audit_compatibility.json')['passed']
        prepare_original()
        plan = fresh.read(RUN / 'protocol.json')
        for filename in ('object_lexicon.json', 'training_vocabulary.json'):
            path = RUN / filename
            shutil.copyfile(PILOT / filename, path)
            plan['source_sha256'][str(path)] = fresh.sha(path.read_bytes())
        plan.update(source=str(PILOT), source_protocol_sha256=fresh.sha((PILOT / 'protocol.json').read_bytes()),
            copied_controls=[], probe=fresh.read(PILOT / 'protocol.json')['probe'],
            primary_baseline='svar', primary='Sentence CHAIR: frozen candidate vs native SVAR. SADI is an additional reference. Publish both evaluators and all500 regardless of outcome.',
            candidate='Unchanged first-class I/E_l-gated layer19 attention action on the native SVAR5--18 base.',
            pilot='Old100 native SVAR Cs45/Ci9.571/F176.722 -> candidate42/8.784/76.963; false87->78. All fixed checks pass, including leave-one-out.',
            advancement='Frozen independent validation after the prospective SVAR development screen; no retuning on the previous two500 cohorts.',
            actuation_preflight=str(RUN / 'svar_adapter_preflight.json'),
            semantic_preflight=str(PILOT / 'svar_semantic_reuse_preflight.json'),
            semantic_cache='Cache image-only E_l during SADI, reuse for the exact same candidate batch. Image-prefix equality across SADI/SVAR was verified on16 development photos.',
            analysis_primary='Use the SVAR primary mention audit for outlier checks; the generic fresh report also contains supplementary SADI-relative sensitivity.',
            timing='SADI includes shared E_l setup; not a pure baseline speed comparison.')
        ids = set(fresh.read(RUN / 'cohort.json')['coco500'])
        with np.load(Path(plan['probe']).parents[1] / 'compact_features.npz', allow_pickle=False) as data:
            old_ids = set(map(int, data['image_id']))
        development = set(fresh.read(PILOT / 'cohort.json')['coco100'])
        prior = [set(fresh.read(fresh.OUT / name / 'cohort.json')['coco500']) for name in (
            'generation_time_evidence_drift_qwen3_fresh_first_class500',
            'generation_time_evidence_drift_qwen3_replication_first_class500')]
        assert ids.isdisjoint(old_ids | development | set.union(*prior))
        fresh.write(RUN / 'protocol.json', plan)
        fresh.write(RUN / 'cohort_independence.json', dict(passed=True, images=500,
            old_probe_images=len(old_ids), overlap_old_probe_images=len(ids & old_ids),
            overlap_development100=len(ids & development), overlap_previous500=[len(ids & group) for group in prior],
            excluded_images=len(fresh.read(RUN / 'selection.json')['excluded_ids']),
            sample_sha256=plan['sample_sha256'], source_count=len(plan['source_sha256'])))
        print('SVAR_FROZEN_FRESH500', plan['sample_sha256'], flush=True)

    def preflight(wrapper, nouns, semantic_tokens):
        from features.coco_object_history import ObjectClassHistory
        from features.qwen3_online_i_el import FullIElProbe
        from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass
        from scripts import verify_qwen3_early_recycle as check
        report = semantic_preflight(wrapper, nouns, semantic_tokens)
        probe = FullIElProbe(fresh.read(PILOT / 'protocol.json')['probe']).to(wrapper.device).eval()
        allowed = frozenset(fresh.read(PILOT / 'object_lexicon.json')['token_ids'])
        parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
        native = check.Qwen3BatchLateAttention

        def svar(model, span, targets, _variant):
            return native(model, span, targets, 'svar')

        def factory(model, span, targets, **kwargs):
            return Qwen3BatchSvarFirstClass(model, span, targets, tokenizer=wrapper.tokenizer,
                                           history_parser=parser, **kwargs)

        with patch.object(check, 'Qwen3BatchLateAttention', svar), \
                patch.object(check, 'Qwen3BatchProbeEarlyRecycle', factory):
            checked = check.verify(wrapper, probe, allowed)
        checked['baseline_variant'] = 'svar'
        fresh.write(RUN / 'svar_adapter_preflight.json', checked)
        return report

    fresh.prepare, original.preflight = prepare, preflight


def worker(shard):
    from features import qwen3_probe_first_class as first
    from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass
    with patch.object(first, 'Qwen3BatchProbeFirstClass', Qwen3BatchSvarFirstClass):
        original.worker(shard)


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
