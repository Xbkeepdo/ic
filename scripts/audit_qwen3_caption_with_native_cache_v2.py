"""Extend the original numeric audit with explicit certified native-cache rows."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit_qwen3_fresh_caption500 import boot, calculate


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def validate_cached_group(group):
    if group[0]['decoder'] == 'native_qwen3_generate_same_span_batch':
        assert all(r['decoder'] == group[0]['decoder'] and not r.get('generation_reused', False) for r in group)
        return
    assert all(r['decoder'] == 'cached_native_no_positive_gate' for r in group)
    info = group[0]['cache_origin']
    certificate_path = Path(info['certificate'])
    assert sha(certificate_path) == info['certificate_sha256']
    certificate = read(certificate_path)
    assert certificate['passed'] and certificate['joint17_agreement_verified']
    entry, = [e for e in certificate['eligible_groups'] if e['shard'] == info['shard'] and e['group'] == info['group']]
    assert [r['image_id'] for r in group] == entry['image_ids']
    origin = Path(info['source'])
    matches=[v for n,v in certificate['source_sha256'].items()
             if (REPO/Path(n)).resolve()==origin.resolve()]
    assert matches and all(v==info['source_sha256'] for v in matches)
    assert sha(origin)==info['source_sha256']
    original = {r['image_id']: r for r in map(json.loads, origin.read_bytes().splitlines())}
    for row in group:
        assert row['cache_origin'] == info and row['generation_reused']
        ref = original[row['image_id']]
        assert row['response_token_ids'] == ref['response_token_ids'] and row['generated_text'] == ref['generated_text']
        assert not ref['sar_ems_statistics']['recycle_events']
        assert not any(g['positive'] for g in ref['sar_ems_statistics']['gate_events'])
        assert row['sar_ems_statistics']['full_gate_events'] == ref['sar_ems_statistics']['gate_events']
        assert row['seconds'] == row['batch_wall_seconds'] == row['actual_model_calls_in_batch'] == 0
        assert row['sar_ems_statistics']['total_accepted'] == 0


def audit(root):
    plan = read(root / 'protocol.json')
    status = read(root / 'run_status.json')
    assert status['status'] == 'complete'
    ids = read(root / 'cohort.json')[plan['cohort']]
    n = len(ids)
    assert n == len(set(ids)) == plan['images']
    for filename, expected in plan['source_sha256'].items():
        assert sha(Path(filename)) == expected, filename
    if 'source_protocol_sha256' in plan:
        assert sha(Path(plan['source']) / 'protocol.json') == plan['source_protocol_sha256']
    output = dict(passed=True, protocol_sha256=sha(root / 'protocol.json'),
                  images=n, eval={}, generation={}, caption_sha256={})
    raw, grouped = {}, {}
    for name in plan['variants']:
        raw[name], grouped[name] = {}, {}
        for shard in range(plan['shards']):
            path = root / plan['cohort'] / f'{name}.shard{shard}.jsonl'
            output['caption_sha256'][str(path)] = sha(path)
            if name in plan['copied_controls']:
                assert path.read_bytes() == (Path(plan['source']) / plan['cohort'] / path.name).read_bytes()
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            groups, offset = [], 0
            while offset < len(rows):
                first = rows[offset]
                size = first['batch_size']
                group = rows[offset:offset + size]
                assert 1 <= size <= 32 and len(group) == size
                validate_cached_group(group)
                for row in group:
                    assert row['image_id'] not in raw[name] and row['variant'] == name
                    assert row['batch_size'] == size and row['batch_wall_seconds'] == first['batch_wall_seconds']
                    assert row['visual_token_count'] == first['visual_token_count']
                    assert row['decoder'] in ('native_qwen3_generate_same_span_batch', 'cached_native_no_positive_gate')
                    assert 1 <= len(row['response_token_ids']) <= plan['max_new_tokens']
                    assert abs(row['seconds'] * size - first['batch_wall_seconds']) < 1e-8
                    raw[name][row['image_id']] = row
                groups.append((tuple(row['image_id'] for row in group), first['visual_token_count']))
                offset += size
            grouped[name][shard] = groups
        assert set(raw[name]) == set(ids)
        assert grouped[name] == grouped[plan['variants'][0]]
        records = list(raw[name].values())
        stats = [r['sar_ems_statistics'] for r in records]
        assert all(not s.get('diagnostic_force_guidance') and not s.get('diagnostic_reject_all')
                   and not s.get('diagnostic_zero_replay') for s in stats)
        output['generation'][name] = dict(
            mean_tokens=statistics.mean(len(r['response_token_ids']) for r in records),
            caps=sum(len(r['response_token_ids']) == plan['max_new_tokens'] for r in records),
            gpu_seconds=sum(r['seconds'] for r in records),
            reused_images=sum(r.get('generation_reused', False) for r in records),
            proposals=sum(s.get('guided_attention_calls', 0) for s in stats),
            accepted=sum(s.get('accepted_guidance_calls', 0) for s in stats),
            repeated_proposals=sum(s.get('repeated_object_proposals', 0) for s in stats),
            repeated_accepted=sum(s.get('repeated_object_accepted', 0) for s in stats),
            repeated_token_changes=sum(s.get('repeated_object_token_changes', 0) for s in stats))
    draws = np.random.default_rng(43).integers(0, n, size=(5000, n))
    for mode in ('sentence', 'legacy'):
        folder = root / f'evaluations/{mode}_v1/{plan["cohort"]}'
        metric_path, = folder.glob('metrics.*.json')
        metrics = read(metric_path)
        assert metrics['evaluation_inputs']['caption_sha256'] == output['caption_sha256']
        per_path, = folder.glob('chair_per_image*.jsonl')
        rows = [json.loads(line) for line in per_path.read_bytes().splitlines()]
        assert [r['image_id'] for r in rows] == ids
        scores = {name: [r['methods'][name] for r in rows] for name in plan['variants']}
        summaries = {name: calculate(values) for name, values in scores.items()}
        samples = {name: boot(values, draws) for name, values in scores.items()}
        intervals = 0
        for name, summary in summaries.items():
            m = metrics['methods'][name]
            for key in ('chair_s', 'chair_i', 'object_recall', 'object_f1'):
                np.testing.assert_allclose(m[key], summary[key], rtol=0, atol=1e-12)
            for reference, pair in m['paired_against'].items():
                for key in ('chair_s', 'chair_i', 'object_recall', 'object_f1'):
                    np.testing.assert_allclose(pair['delta'][key], summary[key] - summaries[reference][key], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(pair['delta_ci95'][key],
                                               np.quantile(samples[name][key] - samples[reference][key], [.025, .975]),
                                               rtol=0, atol=1e-12)
                    intervals += 1
                assert pair['fixed_hallucinated_images'] == sum(a['chair_s'] > b['chair_s'] for a, b in zip(scores[reference], scores[name], strict=True))
                assert pair['introduced_hallucinated_images'] == sum(a['chair_s'] < b['chair_s'] for a, b in zip(scores[reference], scores[name], strict=True))
                assert pair['changed_token_sequences'] == sum(raw[name][i]['response_token_ids'] != raw[reference][i]['response_token_ids'] for i in ids)
            m.update(audit_mentions=summary['object_mentions'], audit_false_mentions=summary['hallucinated_mentions'],
                     audit_false_image_classes=summary['hallucinated_image_classes'])
        output['eval'][mode] = metrics['methods']
        output[mode + '_intervals_verified'] = intervals
    (root / 'independent_summary.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('AUDIT_PASS', n * len(plan['variants']), 'captions')
    for name, m in output['eval']['sentence'].items():
        print(name, *(round(m[k] * 100, 3) for k in ('chair_s', 'chair_i', 'object_recall', 'object_f1')),
              m['audit_false_mentions'], m['audit_mentions'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
