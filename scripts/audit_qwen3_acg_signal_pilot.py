"""Reconstruct native SAM-V2 decisions and signed ACG head scales."""

import argparse
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.coco_object_history import ObjectClassHistory
from scripts.run_qwen3_signed_feedback import read, write, sha, source_groups, group_keys


def audit(root):
    from transformers import AutoTokenizer
    plan = read(root/'protocol.json')
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root/'object_lexicon.json')['token_ids'])
    result = dict(passed=True, protocol_sha256=sha(root/'protocol.json'), variants={})
    for name in plan['candidates']:
        total = dict(images=0, proposals=0, kept=0, changed_images=0, no_kept_exact_images=0)
        for shard in (0, 1):
            native = source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
            current = source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native) == group_keys(current)
            for a, b in zip([r for g in native for r in g], [r for g in current for r in g], strict=True):
                tokens, stats = b['response_token_ids'], b['sar_ems_statistics']
                assert not stats['diagnostic_force_probe_gate'] and not stats['diagnostic_reject_all'] and not stats['diagnostic_zero_replay']
                assert stats['maximum_probe_hidden_difference'] == 0
                events = stats['recycle_events']
                decisions = {d['step']: d for d in stats['class_decisions']}
                gates = {d['step']: d for d in stats['gate_events']}
                assert len(events) == stats['guided_attention_calls']
                assert len({e['step'] for e in events}) == len(events)
                kept = [e for e in events if e['accepted']]
                first = min((e['step'] for e in kept), default=len(tokens))
                for d in decisions.values():
                    prefix = tokenizer.decode(tokens[:d['step']], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    prior = parser.classes(prefix)
                    assert d['token'] in classes and d['token'] not in tokens[:d['step']]
                    assert d['canonical'] == classes[d['token']] and d['prior_classes'] == sorted(prior)
                    assert d['blocked'] == (d['canonical'] in prior)
                for g in gates.values():
                    assert not decisions[g['step']]['blocked'] and g['token'] == decisions[g['step']]['token']
                    assert g['threshold'] == plan['real_threshold']
                    assert g['positive'] == (g['p_real'] < plan['real_threshold'])
                for e in events:
                    g = gates[e['step']]
                    assert g['positive'] and g['p_real'] == e['probe_p_real'] and g['token'] == e['token']
                    assert e['source_layer'] == e['target_layer'] == 31
                    scales = e['head_scales']
                    assert len(scales) == 32 and all(0 <= s <= 1.400001 for s in scales)
                    assert sum(s > 0 for s in scales) == e['nonzero_heads']
                    if e['signal_guided']:
                        effects = e['head_effects']; neg = [max(0, -x) for x in effects]; peak = max(neg)
                        assert peak > 0 and 0 <= e['drift'] <= 1
                        for s, n in zip(scales, neg, strict=True):
                            assert abs(s-1.4*e['drift']*n/peak) < 1e-6
                        assert abs(sum(s*x for s, x in zip(scales, effects, strict=True))-e['predicted_log_probability_change']) < 1e-5
                        assert e['predicted_log_probability_change'] <= 1e-6
                    else:
                        assert all(abs(s-1.4) < 1e-6 for s in scales)
                    assert math.isfinite(e['actual_log_probability_change'])
                    assert e['accepted'] == (e['actual_log_probability_change'] < 0)
                    if e['step'] <= first:
                        assert tokens[:e['step']] == a['response_token_ids'][:e['step']]
                        assert e['token'] == a['response_token_ids'][e['step']]
                if not kept:
                    assert tokens == a['response_token_ids'] and b['generated_text'] == a['generated_text']
                    total['no_kept_exact_images'] += 1
                total['images'] += 1; total['proposals'] += len(events); total['kept'] += len(kept)
                total['changed_images'] += tokens != a['response_token_ids']
        assert total['images'] == 100
        result['variants'][name] = total
    write(root/'mechanism_audit.json', result)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path)
    audit(p.parse_args().root)
