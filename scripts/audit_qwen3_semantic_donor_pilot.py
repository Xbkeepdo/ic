"""Check direct SAM-V2 gates, generated class history and accepted state paths."""

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
        total = dict(images=0, proposals=0, kept=0, changed_images=0, no_kept_exact_images=0, visual_donor_events=0)
        for shard in (0, 1):
            native = source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
            current = source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native) == group_keys(current)
            for a, b in zip([r for g in native for r in g], [r for g in current for r in g], strict=True):
                tokens, stats = b['response_token_ids'], b['sar_ems_statistics']
                assert not stats['diagnostic_force_probe_gate'] and not stats['diagnostic_reject_all'] and not stats['diagnostic_zero_replay']
                assert stats['maximum_probe_hidden_difference'] == 0
                events, decisions, gates = stats['recycle_events'], stats['class_decisions'], stats['gate_events']
                assert len(events) == stats['guided_attention_calls']
                assert len({e['step'] for e in events}) == len(events)
                by_decision, by_gate = {d['step']: d for d in decisions}, {d['step']: d for d in gates}
                kept = [e for e in events if e['accepted']]
                first_kept = min((e['step'] for e in kept), default=len(tokens))
                for d in decisions:
                    prefix = tokenizer.decode(tokens[:d['step']], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    prior = parser.classes(prefix)
                    assert d['token'] in classes and d['token'] not in tokens[:d['step']]
                    assert d['canonical'] == classes[d['token']] and d['prior_classes'] == sorted(prior)
                    assert d['blocked'] == (d['canonical'] in prior)
                for g in gates:
                    assert not by_decision[g['step']]['blocked'] and g['token'] == by_decision[g['step']]['token']
                    assert g['threshold'] == plan['real_threshold']
                    assert g['positive'] == (g['p_real'] < plan['real_threshold'])
                for e in events:
                    g = by_gate[e['step']]
                    assert g['positive'] and e['probe_p_real'] == g['p_real']
                    assert e['source_layer'] == e['target_layer'] == 19 and e['token'] == g['token']
                    assert 0 < e['budget_max'] <= .500001 and 0 <= e['unsupported_fraction'] <= 1
                    assert e['predicted_log_probability_change'] <= 1e-6
                    assert math.isfinite(e['actual_log_probability_change'])
                    assert e['accepted'] == (e['actual_log_probability_change'] < 0)
                    if e['transport_mode'] == 'history':
                        assert e['moved_visual_mass'] == 0
                    if e['step'] <= first_kept:
                        assert tokens[:e['step']] == a['response_token_ids'][:e['step']]
                        assert e['token'] == a['response_token_ids'][e['step']]
                    total['visual_donor_events'] += e['moved_visual_mass'] > 0
                if not kept:
                    assert tokens == a['response_token_ids'] and b['generated_text'] == a['generated_text']
                    total['no_kept_exact_images'] += 1
                assert len(kept) == stats['accepted_guidance_calls']
                total['images'] += 1; total['proposals'] += len(events); total['kept'] += len(kept)
                total['changed_images'] += tokens != a['response_token_ids']
        assert total['images'] == 100
        result['variants'][name] = total
    write(root/'mechanism_audit.json', result)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
