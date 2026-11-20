"""Independent metric reconstruction, denominator checks and review cases."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from scripts.audit_qwen3_fresh_caption500 import calculate, boot
from scripts.audit_qwen3_visual_feedback_mentions import compare
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, group_keys, write


def analyze(root):
    plan = read(root/'protocol.json')
    assert read(root/'run_status.json')['status'] == 'complete'
    assert read(root/'mechanism_audit.json')['passed']
    assert not (root/'independent_summary.json').exists()
    ids = read(root/'cohort.json')['coco100']
    assert len(ids) == len(set(ids)) == plan['images'] == 100
    for name, expected in plan['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    raw, caption_sha, activity = {}, {}, {}
    native_groups = {s: source_groups(root/f'coco100/sadi.shard{s}.jsonl') for s in (0, 1)}
    for name in plan['variants']:
        raw[name] = {}
        batches = []
        for shard in (0, 1):
            p = root/f'coco100/{name}.shard{shard}.jsonl'
            caption_sha[str(p)] = sha(p)
            grouped = source_groups(p)
            assert group_keys(grouped) == group_keys(native_groups[shard])
            origin = plan['source'] if name in plan['copied_controls'] else plan['additional_copied_controls'].get(name)
            if origin:
                assert p.read_bytes() == (Path(origin)/'coco100'/p.name).read_bytes()
            for group in grouped:
                for row in group:
                    assert row['image_id'] not in raw[name]
                    assert row['variant'] == name and row['batch_size'] == len(group) <= 16
                    assert row['visual_token_count'] == group[0]['visual_token_count']
                    assert row['batch_wall_seconds'] == group[0]['batch_wall_seconds']
                    assert abs(row['seconds']*len(group)-row['batch_wall_seconds']) < 1e-8
                    assert row['decoder'] == 'native_qwen3_generate_same_span_batch'
                    assert 1 <= len(row['response_token_ids']) <= 512
                    raw[name][row['image_id']] = row
                batches.append(group[0]['sar_ems_statistics'])
        assert set(raw[name]) == set(ids)
        events = [e for row in raw[name].values() for e in row['sar_ems_statistics'].get('recycle_events', [])]
        activity[name] = dict(images=100, gpu_seconds=sum(r['seconds'] for r in raw[name].values()),
            mean_tokens=sum(len(r['response_token_ids']) for r in raw[name].values())/100,
            capped_images=sum(len(r['response_token_ids']) == 512 for r in raw[name].values()),
            proposals=len(events), kept=sum(e['accepted'] for e in events),
            shared_static_hits=sum(s.get('shared_static_hits', 0) for s in batches),
            semantic_head_calls=sum(s.get('static_head_calls', 0) for s in batches),
            exact_small_OT_calls=sum(s.get('native_risk_solver_calls', 0) for s in batches),
            feature_seconds=sum(s.get('native_feature_seconds', 0) for s in batches),
            static_seconds=sum(s.get('static_seconds', 0) for s in batches))
    report, cases = dict(passed=True, protocol_sha256=sha(root/'protocol.json'),
        analysis_source_sha256=sha(Path(__file__)), images=100, caption_sha256=caption_sha,
        activity=activity, eval={}), {}
    draws = np.random.default_rng(43).integers(0, 100, size=(5000, 100))
    for mode in ('sentence', 'legacy'):
        folder = root/f'evaluations/{mode}_v1/coco100'
        mp, = folder.glob('metrics.*.json'); metrics = read(mp)
        assert metrics['evaluation_inputs']['caption_sha256'] == caption_sha
        pp, = folder.glob('chair_per_image*.jsonl')
        rows = [json.loads(s) for s in pp.read_text().splitlines()]
        assert [r['image_id'] for r in rows] == ids
        scores = {name: [r['methods'][name] for r in rows] for name in plan['variants']}
        summary = {name: calculate(s) for name, s in scores.items()}
        samples = {name: boot(s, draws) for name, s in scores.items()}
        for name, value in summary.items():
            measured = metrics['methods'][name]
            for key in ('chair_s', 'chair_i', 'object_recall', 'object_f1'):
                np.testing.assert_allclose(value[key], measured[key], atol=1e-12, rtol=0)
            for reference, paired in measured['paired_against'].items():
                for key in ('chair_s', 'chair_i', 'object_recall', 'object_f1'):
                    np.testing.assert_allclose(paired['delta'][key], value[key]-summary[reference][key], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(paired['delta_ci95'][key],
                        np.quantile(samples[name][key]-samples[reference][key], [.025, .975]), atol=1e-12, rtol=0)
                assert paired['changed_token_sequences'] == sum(raw[name][i]['response_token_ids'] != raw[reference][i]['response_token_ids'] for i in ids)
        pairs = {name: {ref: compare(scores[ref], scores[name], ids, draws)
                       for ref in plan['variants'] if ref != name} for name in plan['candidates']}
        report['eval'][mode] = dict(metrics=summary, pairs=pairs)
        if mode == 'sentence':
            by_id = {r['image_id']: r['methods'] for r in rows}
            for name in plan['candidates']:
                cases[name] = {}
                for i in ids:
                    a, b = by_id[i]['sadi'], by_id[i][name]
                    if raw[name][i]['generated_text'] != raw['sadi'][i]['generated_text']:
                        cases[name][str(i)] = dict(
                            false_mentions_changed=Counter(map(tuple, a['hallucinated_objects'])) != Counter(map(tuple, b['hallucinated_objects'])),
                            methods={m: dict(text=raw[m][i]['generated_text'], scores=by_id[i][m],
                                events=raw[m][i]['sar_ems_statistics'].get('recycle_events', [])) for m in plan['variants']})
    report['numerical_screen'] = read(root/'screening.json')
    report['human_review'] = 'pending'
    write(root/'changed_text_cases.json', cases)
    write(root/'independent_summary.json', report)
    print(json.dumps(dict(metrics=report['eval']['sentence']['metrics'], activity=activity,
                         screen=report['numerical_screen']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    analyze(p.parse_args().root)
