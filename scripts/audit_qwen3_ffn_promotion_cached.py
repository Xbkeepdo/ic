"""Verify the full gate and exact provenance of unchanged cached batches."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_qwen3_ffn_promotion import audit as gate_audit
from scripts.run_qwen3_ffn_promotion_cached_pilot import promoted_cached_row, NAME
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, write


def audit(root):
    gate_audit(root)
    plan = read(root / 'protocol.json')
    for path, expected in plan['source_sha256'].items():
        assert sha(Path(path)) == expected, path
    output = {shard: source_groups(root / 'coco100' / f'{NAME}.shard{shard}.jsonl') for shard in (0, 1)}
    report = dict(passed=True, cached_images=0, fresh_images=0, checked_risk_positive_decisions=0,
                  cache_groups=0, fresh_groups=0)
    fresh_ids = []
    for batch in plan['cache_replay_batches']:
        source = Path(batch['source'])
        old = source_groups(source)[batch['index']]
        new = output[batch['shard']][batch['index']]
        assert [r['image_id'] for r in old] == [r['image_id'] for r in new] == batch['ids']
        differences = [dict(image_id=row['image_id'], **d) for row in old
                       for d in row['sar_ems_statistics']['conflict_decisions']
                       if d['permitted'] != (d['gain'] > 0)]
        assert differences == batch['differing_decisions'] and batch['reuse'] == (not differences)
        if batch['reuse']:
            assert new == [promoted_cached_row(row, source) for row in old]
            report['cached_images'] += len(new)
            report['cache_groups'] += 1
        else:
            assert all(r['generation_provenance']['fresh_generation'] for r in new)
            report['fresh_images'] += len(new)
            report['fresh_groups'] += 1
            fresh_ids.append(batch['ids'])
        report['checked_risk_positive_decisions'] += sum(
            len(row['sar_ems_statistics']['conflict_decisions']) for row in old)
    replay = read(root / 'sparse_native_replay.json')
    assert replay['passed'] and all(b['native_sadi_prefix_exact'] for b in replay['batches'])
    assert [b['ids'] for b in replay['batches']] == fresh_ids
    assert report['cached_images'] == plan['reused_images'] and report['fresh_images'] == plan['fresh_images']
    assert report['cached_images'] + report['fresh_images'] == plan['images']
    write(root / 'ffn_promotion_cache_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
