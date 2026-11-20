"""Version2: preserve all checks, use retained-action history for native-state parity."""

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import audit_qwen3_probe_first_class as classes
from scripts import audit_qwen3_svar_first_class as reference
from scripts import audit_qwen3_svar_attention_midpoint as midpoint
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, write
from scripts.audit_qwen3_retained_prefix_state import verify_native_state_prefix, physical_reference_audit


def audit(root):
    plan = read(root / 'protocol.json')
    assert read(root / 'run_status.json')['status'] == 'complete'
    assert read(root / 'cohort_independence.json')['passed']
    assert read(root / 'svar_cache_preflight.json')['exact_native_svar_logits_and_tokens']
    assert plan['variants'] == ['svar', 'svar_probe_first_class_recycle_19',
                              'svar_first_class_half_19', 'svar_first_class_geometric_19']
    midpoint.audit(root)
    # The legacy full class/physics verifier accepts one method. Supply an
    # in-memory view for the unchanged reference, preserving protocol bytes.
    original = copy.deepcopy(plan)
    original.update(candidates=[plan['reference_method']], actuation_preflight=plan['reference_actuation_preflight'],
                    semantic_preflight=plan['reference_semantic_preflight'])
    original_read = read

    def reference_read(path):
        return original if Path(path).resolve() == (root / 'protocol.json').resolve() else original_read(path)

    def reference_write(path, value):
        return write(path.with_name('reference_' + path.name), value)

    with patch.object(reference, 'read', reference_read), patch.object(classes, 'read', reference_read), \
            patch.object(reference, 'write', reference_write), patch.object(classes, 'write', reference_write), \
            patch.object(reference, 'physical_audit', physical_reference_audit):
        reference.audit(root)
    tokens = sorted(read(root / 'training_vocabulary.json')['token_ids'])
    report = dict(passed=True, images=0, paired_batches=0, cache_files={}, native_prefix_events={n: 0 for n in plan['candidates']})
    for shard in (0, 1):
        prefix = read(root / f'cache_prefix.gpu{shard}.json')
        assert prefix['passed'] and prefix['exact_native_svar_prefix_with_cache'] and prefix['steps'] == 64
        caches = {}
        for path in (root / 'semantics').glob(f'shard{shard}.*.npz'):
            with np.load(path, allow_pickle=False) as saved:
                ids, span = saved['image_ids'].tolist(), tuple(saved['span'].tolist())
                key = sha_bytes(str((ids, span)).encode())[:20]
                assert path.name == f'shard{shard}.{key}.npz'
                assert saved['token_ids'].tolist() == tokens
                values = saved['E_l']
                assert values.shape == (36, len(ids), len(tokens)) and values.dtype == np.float32
                assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= span[1] - span[0] + 1e-4
                assert tuple(ids) not in caches
                caches[tuple(ids)] = (path, span, sha(path))
        base = source_groups(root / f'coco500/svar.shard{shard}.jsonl')
        lookups = {n: {r['image_id']: r for group in source_groups(root / f'coco500/{n}.shard{shard}.jsonl') for r in group}
                   for n in plan['variants']}
        for group in base:
            ids = [r['image_id'] for r in group]
            path, span, digest = caches[tuple(ids)]
            assert span[1] - span[0] == group[0]['visual_token_count']
            report['cache_files'][str(path)] = digest
            report['paired_batches'] += 1
            for row in group:
                report['images'] += 1
                old = row['response_token_ids']
                for name in plan['candidates']:
                    candidate = lookups[name][row['image_id']]
                    current = candidate['response_token_ids']
                    for event in candidate['sar_ems_statistics']['recycle_events']:
                        step = event['step']
                        if verify_native_state_prefix(old, current, candidate['sar_ems_statistics']['recycle_events'], event):
                            assert old[step] == event['token']
                            report['native_prefix_events'][name] += 1
    assert report['images'] == 500
    write(root / 'midpoint500_runtime_audit.json', report)
    print('MIDPOINT500_RUNTIME_AUDIT_PASS', report['paired_batches'], 'caches')


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
