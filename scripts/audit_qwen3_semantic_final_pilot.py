"""Independent target/history/no-action and semantic mechanism bookkeeping."""

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.coco_object_history import ObjectClassHistory
from scripts.run_qwen3_signed_feedback import read, sha, source_groups, group_keys, write


def audit(root):
    from transformers import AutoTokenizer
    plan = read(root / 'protocol.json')
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root / 'object_lexicon.json')['token_ids'])
    report = dict(passed=True, variants={})
    for name, source in plan['additional_copied_controls'].items():
        for shard in (0, 1):
            filename = f'coco100/{name}.shard{shard}.jsonl'
            assert (root / filename).read_bytes() == (Path(source) / filename).read_bytes()
    for name in plan['candidates']:
        totals = dict(images=0, events=0, accepted=0, decisions=0, blocked=0,
                      no_action_exact_images=0, probability_decreases=0, changed_sequences=0)
        for shard in (0, 1):
            native = source_groups(root / f'coco100/sadi.shard{shard}.jsonl')
            current = source_groups(root / f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native) == group_keys(current)
            for a, b in zip([r for g in native for r in g], [r for g in current for r in g], strict=True):
                assert a['image_id'] == b['image_id']
                tokens, stats = b['response_token_ids'], b['sar_ems_statistics']
                assert stats['gradient_scope'] == 'none' and stats['gradient_probe_calls'] == 0
                assert not any(stats[k] for k in ('diagnostic_zero_replay', 'diagnostic_reject_all', 'permuted_spatial_control'))
                events = stats['semantic_events']
                assert len(events) == stats['guided_attention_calls'] == stats['local_recompute_events'] == stats['accepted_guidance_calls']
                assert len({e['step'] for e in events}) == len(events)
                decisions = {d['step']: d for d in stats['class_decisions']}
                assert len(decisions) == len(stats['class_decisions'])
                assert sum(d['blocked'] for d in decisions.values()) == stats['blocked_by_class_history']
                for step, d in decisions.items():
                    prefix = tokenizer.decode(tokens[:step], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    prior = parser.classes(prefix)
                    assert d['token'] in classes and d['token'] not in tokens[:step]
                    assert d['canonical'] == classes[d['token']] and d['prior_classes'] == sorted(prior)
                    assert d['blocked'] == (d['canonical'] in prior)
                    if d['blocked']:
                        assert tokens[step] == d['token']
                for e in events:
                    d = decisions[e['step']]
                    assert e['source_layer'] == e['target_layer'] == 31
                    assert e['token'] == d['token'] and not d['blocked'] and e['accepted']
                    assert 0 < e['budget_max'] <= 5.000001 and e['gain'] > 0 and 0 < e['drift'] <= 1
                    assert abs(e['target_sum']-1) < 1e-5 and abs(e['preference_sum']-1) < 1e-5
                    assert math.isfinite(e['actual_log_probability_change'])
                if not events:
                    assert tokens == a['response_token_ids'] and b['generated_text'] == a['generated_text']
                    totals['no_action_exact_images'] += 1
                else:
                    first = min(e['step'] for e in events)
                    assert tokens[:first] == a['response_token_ids'][:first]
                    assert events[0]['token'] == a['response_token_ids'][first]
                totals['images'] += 1
                totals['events'] += len(events)
                totals['accepted'] += sum(e['accepted'] for e in events)
                totals['probability_decreases'] += sum(e['actual_log_probability_change'] < 0 for e in events)
                totals['decisions'] += len(decisions)
                totals['blocked'] += sum(d['blocked'] for d in decisions.values())
                totals['changed_sequences'] += tokens != a['response_token_ids']
        assert totals['images'] == plan['images']
        report['variants'][name] = totals
    report['protocol_sha256'] = sha(root / 'protocol.json')
    write(root / 'semantic_mechanism_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
