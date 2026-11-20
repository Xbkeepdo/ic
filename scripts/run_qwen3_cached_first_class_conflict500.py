"""Add one frozen sign-gated arm to the cached independent500 comparison.

The sign rule was declared while the first500 generation was still running.
This runner regenerates only the new arm and reuses identical ordered batches,
native controls, and original E_l caches. It must not tune on test metrics.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_signed_feedback as shared

SOURCE = shared.OUT / 'generation_time_evidence_drift_qwen3_fresh_first_class500'
PILOT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-first-class-conflict-pilot')
OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-first-class-conflict-500')
NAME = 'sadi_probe_first_class_conflict_19'


def prepare():
    assert not OUT.exists(), 'Inspect existing output before any retry'
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    assert shared.read(PILOT / 'run_status.json')['status'] == 'complete'
    assert shared.read(PILOT / 'first_class_conflict_audit.json')['passed']
    pilot = shared.read(PILOT / 'protocol.json')
    assert pilot['independent500_status_at_design'] == 'generating'
    summary = shared.read(PILOT / 'independent_summary.json')['eval']['sentence']
    base, candidate = summary['sadi'], summary[NAME]
    # Prospective development-only filter. The independent500 metric files
    # are not consulted to choose parameters or decide whether to run this arm.
    assert candidate['chair_s'] <= base['chair_s']
    assert candidate['chair_i'] <= base['chair_i']
    assert candidate['object_f1'] >= base['object_f1'] - .015
    assert candidate['audit_false_mentions'] < base['audit_false_mentions']
    source = shared.read(SOURCE / 'protocol.json')
    assert source['max_new_tokens'] == 512 and source['max_batch'] == 16
    hashes = {str(REPO / path): expected for path, expected in source['source_sha256'].items()}
    for path, expected in hashes.items():
        assert shared.sha(Path(path)) == expected, path
    for filename in ('features/qwen3_probe_first_class_conflict.py',
                     'scripts/run_qwen3_cached_first_class_conflict500.py'):
        hashes[str(REPO / filename)] = shared.sha(REPO / filename)
    for path in (SOURCE / 'protocol.json', PILOT / 'protocol.json', PILOT / 'independent_summary.json',
                 PILOT / 'object_lexicon.json', PILOT / 'training_vocabulary.json'):
        hashes[str(path)] = shared.sha(path)
    for path in (SOURCE / 'semantics').glob('*.npz'):
        hashes[str(path)] = shared.sha(path)
    OUT.mkdir()
    (OUT / 'coco500').mkdir()
    (OUT / 'logs').mkdir()
    controls = list(source['variants'])
    for shard in (0, 1):
        reference = None
        for method in controls:
            path = SOURCE / f'coco500/{method}.shard{shard}.jsonl'
            groups = shared.group_keys(shared.source_groups(path))
            if reference is None:
                reference = groups
            assert groups == reference
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT / 'coco500' / path.name)
    for name in ('cohort.json', 'cohort_independence.json'):
        shutil.copyfile(SOURCE / name, OUT / name)
    for name in ('object_lexicon.json', 'training_vocabulary.json'):
        shutil.copyfile(PILOT / name, OUT / name)
        hashes[str(OUT / name)] = shared.sha(OUT / name)
    plan = dict(source=str(SOURCE), source_protocol_sha256=shared.sha(SOURCE / 'protocol.json'),
        source_trace=pilot['source_trace'], stage='head-gradient-first-class-conflict-500',
        cohort='coco500', images=500, shards=2, max_new_tokens=512, max_batch=16,
        model='Qwen3-VL-8B-Instruct', variants=[*controls, NAME], candidates=[NAME],
        cached_controls=['sadi'], copied_controls=controls, source_sha256=hashes,
        probe=pilot['probe'], recipe=pilot['recipe'], actuation_preflight=pilot['actuation_preflight'],
        cache_source=str(SOURCE), independent500_status_at_design='generating',
        pilot_protocol_sha256=shared.sha(PILOT / 'protocol.json'),
        selection='Run only if the old100 development Cs/Ci do not worsen, F1 loss<=1.5pp, and false mentions decrease. Do not consult test500 metrics.',
        validation_scope='Same fixed independent500 cohort; report both predeclared variants and all controls, not a new independent sample.',
        batch_policy='Reuse exact source500 ordered groups and GPU shards; no automatic re-batching or partial resume.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    import models
    from features import qwen3_sar_ems_head_gradient as head
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_first_class_conflict import Qwen3BatchFirstClassConflict
    plan = shared.read(OUT / 'protocol.json')
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    build = models.build_model
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    groups = shared.source_groups(SOURCE / f'coco500/sadi.shard{gpu}.jsonl')
    probe, tokenizer, index = None, None, 0

    def checked_build(*args, **kwargs):
        nonlocal probe, tokenizer
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
        tokenizer = wrapper.tokenizer
        return wrapper

    def factory(model, span, nouns, *, final_gradient):
        nonlocal index
        assert final_gradient
        ids = [r['image_id'] for r in groups[index]]
        # Match the source cache's exact key; no feature recomputation.
        key = hashlib.sha256(str((ids, tuple(span))).encode()).hexdigest()[:20]
        path = SOURCE / f'semantics/shard{gpu}.{key}.npz'
        assert shared.sha(path) == plan['source_sha256'][str(path)]
        with np.load(path, allow_pickle=False) as saved:
            assert saved['image_ids'].tolist() == ids and saved['span'].tolist() == list(span)
            semantic_tokens = saved['token_ids'].tolist()
            cached = torch.from_numpy(saved['E_l'].copy())
        index += 1
        return Qwen3BatchFirstClassConflict(model, span, allowed, probe=probe,
            semantic_tokens=semantic_tokens, cached_semantics=cached, guided=True,
            tokenizer=tokenizer, history_parser=parser)

    with patch.object(shared, 'HEAD_FINAL', NAME), patch.object(head, 'Qwen3BatchHeadGradient', factory), \
            patch.object(models, 'build_model', checked_build):
        shared.worker(OUT, gpu)
    assert index == len(groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare):
            driver.run()
    else:
        worker(args.worker)
