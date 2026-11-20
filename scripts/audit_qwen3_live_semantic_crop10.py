"""Independently aggregate scores and verify generation/metadata provenance."""

import argparse
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_bytes())
def rows(p): return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root):
    plan = read(root/'protocol.json'); checks = 0
    assert read(root/'generation_complete.json')['returncodes'] == [0, 0]
    assert read(root/'evaluation_recovery_v3_complete.json')['passed']
    assert read(root/'preflight.json')['native_zero_reject_all36_kv_logits_tokens_exact']
    assert read(root/'preflight.json')['positive_commit_tested']; checks += 4
    raw = {v: {r['image_id']: r for s in [0, 1] for r in rows(root/f'coco10/{v}.shard{s}.jsonl')}
        for v in plan['variants']}
    for receipt in plan['copied_control_receipts']:
        assert sha(root/'coco10'/Path(receipt['target']).name) == receipt['sha256']; checks += 1
    changed = {}; accepted = {}; totals = {}; coverage = []
    for variant in plan['candidates']:
        changed[variant] = [i for i in raw[variant] if raw[variant][i]['response_token_ids'] != raw['sadi'][i]['response_token_ids']]
        accepted[variant] = 0
        for image, item in raw[variant].items():
            events = item['sar_ems_statistics']['full_events']
            for event in events:
                expected = event['proposed_token'] if event['accepted'] else event['token']
                assert item['response_token_ids'][event['step']] == expected; checks += 1
                if event['accepted']:
                    assert event['positive_commit_readback_exact'] and event['current_kv_difference'] > 0; checks += 1
                    accepted[variant] += 1
            derived = next(r for s in [0, 1] for r in rows(root/f'evaluation_inputs_v3/coco10/{variant}.shard{s}.jsonl') if r['image_id'] == image)
            restored = dict(derived); del restored['seconds']
            restored['sar_ems_statistics'] = restored.pop('semantic_crop_statistics')
            assert restored == item and derived['seconds'] == item['batch_wall_seconds']/item['batch_size']; checks += 1
    assert changed == dict(semantic_crop=[353411], semantic_crop_route=[], shifted_crop=[]); checks += 1
    assert accepted == dict(semantic_crop=1, semantic_crop_route=0, shifted_crop=0); checks += 1
    for version in ['sentence', 'legacy']:
        folder = root/f'evaluations/recovery_v3_{version}_v1/coco10'
        metrics_path, = folder.glob('metrics.*.json'); per_path, = folder.glob('chair_per_image.*.jsonl')
        official = read(metrics_path)['methods']; per = rows(per_path); totals[version] = {}
        assert len(per) == 10 and {r['image_id'] for r in per} == set(raw['sadi']); checks += 1
        for variant in plan['variants']:
            scored = [r['methods'][variant] for r in per]
            false = sum(r['hallucinated_mentions'] for r in scored)
            mentions = sum(r['object_mentions'] for r in scored)
            result = dict(chair_s=sum(r['chair_s'] for r in scored)/10,
                chair_i=false/max(1, mentions), object_recall=sum(r['object_recall'] for r in scored)/10,
                object_f1=sum(r['object_f1'] for r in scored)/10,
                hallucinated_mentions=false, object_mentions=mentions)
            for k, value in result.items(): assert abs(value-official[variant][k]) < 1e-12; checks += 1
            totals[version][variant] = result
            if variant in plan['candidates']:
                for row in per:
                    baseline, current = row['methods']['sadi'], row['methods'][variant]
                    for key in ['chair_s', 'chair_i', 'hallucinated_mentions', 'object_mentions', 'object_recall', 'object_f1', 'hallucinated_objects']:
                        assert baseline[key] == current[key]; checks += 1
        if version == 'sentence':
            for row in per:
                image = row['image_id']; false_classes = sorted({v[1] for v in row['methods']['sadi']['hallucinated_objects']})
                if not false_classes: continue
                gates = raw['semantic_crop_route'][image]['sar_ems_statistics']['gates']
                for name in false_classes:
                    matched = [g for g in gates if g['target'] == name]
                    coverage.append(dict(image_id=image, false_class=name, positive_gain_prefix_novel_logged=bool(matched),
                        semantic_eligible=any(g['eligible'] for g in matched)))
    result = dict(passed=True, checks=checks, new_model_calls=0, changed_images=changed, accepted=accepted,
        all_candidate_per_image_object_scores_equal_sadi=True, metrics=totals,
        coverage=dict(image_class_cases=len(coverage), positive_gain_logged=sum(r['positive_gain_prefix_novel_logged'] for r in coverage),
            semantic_eligible=sum(r['semantic_eligible'] for r in coverage), rows=coverage),
        decision='Close this candidate: nap rewrite retains fork in the full caption; no100 or500 advancement.')
    (root/'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'metrics'}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); args = p.parse_args(); audit(args.root)
