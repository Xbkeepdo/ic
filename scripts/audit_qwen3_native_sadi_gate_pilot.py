"""Check unchanged uniform attention with the frozen risk/E_l input projection."""

import argparse
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.coco_object_history import ObjectClassHistory
from scripts.run_qwen3_signed_feedback import read, write, sha, group_keys
from scripts.qwen3_caption_groups_with_cache import source_groups

TARGETS = list(range(20, 36))


def _operator_audit(root):
    from transformers import AutoTokenizer
    plan = read(root/'protocol.json'); ids = read(root/'cohort.json')['coco100']
    assert len(set(ids)) == 100 and read(root/'run_status.json')['status'] == 'complete'
    for name, value in plan['source_sha256'].items():
        assert sha(REPO/name) == value, name
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    eos = plan['eos_token_ids']; eos = set(eos if isinstance(eos, list) else [eos])
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root/'object_lexicon.json')['token_ids'])
    report = dict(passed=True, protocol_sha256=sha(root/'protocol.json'), variants={})
    for name in plan['candidates']:
        mode = plan['mode_by_candidate'][name]; refresh = plan['refresh_by_candidate'][name]
        total = dict(cached_images=0, images=0, proposals=0, kept=0, changed_images=0,
            no_kept_exact_images=0, kept_token_changes=0, gradient_calls=0,
            terminal_eos_accepts=0, actual_eos_accepts=0, per_layer_coefficients_checked=0,
            causally_refreshed_layers_with_changed_budget=0, layer_calls=0, refreshed_norm_calls=0)
        for shard in (0, 1):
            native = source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
            current = source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native) == group_keys(current)
            assert {r['image_id'] for group in current for r in group} == set(ids[shard::2])
            for original_group, group in zip(native, current, strict=True):
                event_steps = {e['step'] for row in group for e in row['sar_ems_statistics']['full_joint_events']}
                event_count = sum(len(row['sar_ems_statistics']['full_joint_events']) for row in group)
                layer_calls = 16*len(event_steps)
                norm_calls = 15*event_count if refresh and mode == 'evidence' else 0
                assert {row['sar_ems_statistics']['causal_layer_calls'] for row in group} == {layer_calls}
                assert {row['sar_ems_statistics']['refreshed_norm_calls'] for row in group} == {norm_calls}
                assert {row['sar_ems_statistics']['gradient_probe_calls'] for row in group} == {0}
                assert len({row['actual_model_calls_in_batch'] for row in group}) == 1
                total['layer_calls'] += layer_calls; total['refreshed_norm_calls'] += norm_calls
                for a, b in zip(original_group, group, strict=True):
                    tokens, stats = b['response_token_ids'], b['sar_ems_statistics']
                    assert not stats['signal_guided'] and stats['bounded_visual_tilt'] and stats['causal_visual_tilt']
                    assert stats['tilt_mode'] == mode and stats['tilt_scale'] == 1.
                    assert stats['gate_feature_components'] == ['risk','I','E_l']
                    assert stats['gate_feature_indices'] == list(range(108))
                    assert stats['feature_reader_output_width'] == 108 and stats['probe_input_width'] == 108
                    assert stats['refresh_source'] == refresh and not stats['common_native_point']
                    assert stats['budgets_common_native_point'] == (not refresh)
                    assert stats['attention_scores_from_current_replay'] and stats['same_final_target_throughout_replay']
                    assert stats['gradient_sources'] == [] and not stats['one_backward_for_all_layers']
                    assert stats['maximum_probe_hidden_difference'] is None and not stats['gradient_hidden_probe_performed']
                    assert not any(stats[key] for key in ('diagnostic_zero_tilt', 'diagnostic_force_probe_gate',
                        'diagnostic_reject_all', 'diagnostic_zero_replay'))
                    assert stats['zero_replay_checks'] == 0 and stats['maximum_zero_replay_hidden_difference'] == 0
                    assert stats['intervention_layers'] == stats['replayed_tail_layers'] == TARGETS
                    assert stats['budget_source_layers'] == list(range(19, 35))
                    total['cached_images'] += b['generation_reused']
                    if b['generation_reused']:
                        assert b['actual_model_calls_in_batch'] == 0
                    events = stats['full_joint_events']; decisions = {d['step']: d for d in stats['full_class_decisions']}
                    gates = {g['step']: g for g in stats['full_gate_events']}
                    assert stats['full_event_log_includes_terminal_eos']
                    assert len({e['step'] for e in events}) == len(events) == stats['total_proposals']
                    assert stats['recycle_events'] == [dict({key: value for key, value in event.items()
                        if key not in ('emitted_token', 'terminal_eos', 'emitted_is_eos')}, generated_token=event['emitted_token'])
                        for event in events if not event['terminal_eos']]
                    kept = [e for e in events if e['accepted']]
                    assert len(kept) == stats['total_accepted']
                    first = min((e['step'] for e in kept), default=len(tokens))
                    for decision in decisions.values():
                        step = decision['step']; assert step <= len(tokens)
                        prior = parser.classes(tokenizer.decode(tokens[:step], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False))
                        token = decision['token']
                        assert token in classes and token not in tokens[:step]
                        assert decision['canonical'] == classes[token] and decision['prior_classes'] == sorted(prior)
                        assert decision['blocked'] == (decision['canonical'] in prior)
                    for gate in gates.values():
                        assert not decisions[gate['step']]['blocked']
                        assert decisions[gate['step']]['token'] == gate['token']
                        assert gate['threshold'] == plan['real_threshold']
                        assert gate['positive'] == (gate['p_real'] < plan['real_threshold'])
                    for event in events:
                        gate = gates[event['step']]
                        assert gate['positive'] and event['probe_p_real'] == gate['p_real'] and event['token'] == gate['token']
                        assert not event['signal_guided'] and event['source_layer'] == 19 and event['target_layer'] == 20
                        assert event['bounded_visual_tilt'] and event['causal_visual_tilt']
                        assert event['tilt_mode'] == mode and event['refresh_source'] == refresh
                        assert event['same_final_target_throughout_replay']
                        assert event['probe_real_threshold'] == plan['real_threshold']
                        assert event['budget_units'] == 'visual_attention_logit' and event['budget_max'] > 0
                        assert event['joint_layers'] == TARGETS and set(event['layer_details']) == set(map(str, TARGETS))
                        for target in TARGETS:
                            layer = event['layer_details'][str(target)]
                            assert layer['source_layer'] == target-1 and layer['target_layer'] == target
                            assert layer['source_refreshed'] == (refresh and target != 20)
                            lo, hi, mean = (layer[key] for key in ('coefficient_min', 'coefficient_max', 'coefficient_mean'))
                            drift = layer['drift']
                            assert 0 <= lo <= hi <= 1 and lo-1e-6 <= mean <= hi+1e-6 and 0 <= drift <= 1
                            assert 0 <= layer['visual_logit_shift_min'] <= layer['visual_logit_shift_max']
                            assert math.isfinite(layer['visual_logit_shift_max']) and layer['bounded_positive_visual_tilt']
                            assert layer['budget_max'] == layer['visual_logit_shift_max']
                            delta = layer['coefficient_native_max_difference']; assert 0 <= delta <= 1
                            assert layer['source_attention_max_difference'] >= 0 and layer['source_ffn_max_difference'] >= 0
                            if target == 20:
                                assert delta == layer['source_attention_max_difference'] == layer['source_ffn_max_difference'] == 0
                            if not refresh or mode == 'fixed':
                                assert delta == 0
                            if mode == 'fixed':
                                assert drift == 1 and lo == hi == .5 and abs(mean-.5) <= 2**-24
                            else:
                                assert hi <= drift+1e-6
                            total['causally_refreshed_layers_with_changed_budget'] += layer['source_refreshed'] and delta > 0
                            total['per_layer_coefficients_checked'] += 1
                        assert max(layer['budget_max'] for layer in event['layer_details'].values()) == event['budget_max']
                        assert math.isfinite(event['actual_log_probability_change'])
                        assert event['accepted'] == (event['actual_log_probability_change'] < 0)
                        assert event['native_top2_gap'] >= 0 and event['original_target_log_probability'] <= 0
                        assert event['immediate_token_changed'] == (event['proposed_token'] != event['token'])
                        emitted = event['proposed_token'] if event['accepted'] else event['token']
                        assert event['emitted_token'] == emitted and event['terminal_eos'] == (event['step'] == len(tokens))
                        assert event['emitted_is_eos'] == (emitted in eos)
                        if event['terminal_eos']:
                            assert emitted in eos
                        else:
                            assert tokens[event['step']] == emitted
                        if event['step'] <= first:
                            assert tokens[:event['step']] == a['response_token_ids'][:event['step']]
                            assert event['token'] == a['response_token_ids'][event['step']]
                    if not kept:
                        assert tokens == a['response_token_ids'] and b['generated_text'] == a['generated_text']
                        total['no_kept_exact_images'] += 1
                    assert stats['terminal_eos_accepts'] == sum(e['terminal_eos'] for e in kept)
                    assert stats['actual_eos_accepts'] == sum(e['emitted_is_eos'] for e in kept)
                    total['images'] += 1; total['proposals'] += len(events); total['kept'] += len(kept)
                    total['changed_images'] += tokens != a['response_token_ids']
                    total['kept_token_changes'] += sum(e['immediate_token_changed'] for e in kept)
                    total['terminal_eos_accepts'] += stats['terminal_eos_accepts']
                    total['actual_eos_accepts'] += stats['actual_eos_accepts']
        assert total['images'] == 100 and total['cached_images'] == 0
        report['variants'][name] = total
    write(root/'operator_mechanism_audit.json', report)
    print(__import__('json').dumps(report, indent=2))




def audit(root):
    # All operator, KV, class-history, event and unchanged-caption checks above
    # are retained from the existing audit. Only the declared gate width changes.
    _operator_audit(root)
    plan=read(root/'protocol.json')
    diagnostic=read(Path(plan['gate_diagnostic'])/'predictions.json')
    lookup={(r['image_id'],r['step'],r['token_id']):r for r in diagnostic}
    checked=0;maximum=0.;unmatched=0
    for shard in (0,1):
        rows=source_groups(root/f'coco100/{plan["candidates"][0]}.shard{shard}.jsonl')
        for group in rows:
            for row in group:
                stats=row['sar_ems_statistics']
                first=min((e['step'] for e in stats['full_joint_events'] if e['accepted']),
                          default=len(row['response_token_ids']))
                for gate in stats['full_gate_events']:
                    if gate['step']>first:continue
                    key=row['image_id'],gate['step'],gate['token']
                    if key not in lookup:
                        unmatched+=1;continue
                    reference=lookup[key]['scores'][plan['gate_diagnostic_score_key']]
                    error=abs(gate['p_real']-reference)
                    assert error<=plan['common_native_probability_limit'],(key,error)
                    checked+=1;maximum=max(maximum,error)
    assert checked>0
    report=read(root/'operator_mechanism_audit.json')
    report['source_adapted_gate']=dict(
        probe_input_width=108,gate_variant=plan['gate_variant'],
        diagnostic_sha256=sha(Path(plan['gate_diagnostic'])/'predictions.json'),
        common_native_slots_checked=checked,unmatched_common_slots=unmatched,
        common_native_probability_max_error=maximum,
        probability_limit=plan['common_native_probability_limit'])
    write(root/'mechanism_audit.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
