"""Audit bounded visual-logit tilts, original gates and native-cache outputs."""

import argparse
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from features.coco_object_history import ObjectClassHistory
from features.qwen3_block_acg_signal import POST
from scripts.run_qwen3_signed_feedback import read,write,sha,group_keys
from scripts.qwen3_caption_groups_with_cache import source_groups


def audit(root):
    from transformers import AutoTokenizer
    plan=read(root/'protocol.json');ids=read(root/'cohort.json')['coco100']
    assert len(set(ids))==100 and read(root/'run_status.json')['status']=='complete'
    for n,v in plan['source_sha256'].items():assert sha(Path(n))==v,n
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    eos=read(Path('/root/rivermind-data/models/qwen3_vl_8b/generation_config.json'))['eos_token_id']
    eos=set(eos if isinstance(eos,list) else [eos])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(tokenizer,read(root/'object_lexicon.json')['token_ids'])
    report=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),variants={})
    for name in plan['candidates']:
        guided=False;mode=plan['mode_by_candidate'][name]
        total=dict(cached_images=0,images=0,proposals=0,kept=0,changed_images=0,no_kept_exact_images=0,
            kept_token_changes=0,gradient_calls=0,terminal_eos_accepts=0,per_layer_coefficients_checked=0)
        for shard in (0,1):
            native=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
            current=source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native)==group_keys(current)
            assert {r['image_id'] for group in current for r in group}==set(ids[shard::2])
            for original_group,group in zip(native,current,strict=True):
                positives={g['step'] for r in group for g in r['sar_ems_statistics']['full_gate_events'] if g['positive']}
                expected_calls=len(positives) if guided else 0
                assert {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}=={expected_calls}
                assert len({r['actual_model_calls_in_batch'] for r in group})==1
                total['gradient_calls']+=expected_calls
                for a,b in zip(original_group,group,strict=True):
                    tokens,stats=b['response_token_ids'],b['sar_ems_statistics']
                    assert stats['signal_guided']==False and stats['bounded_visual_tilt']
                    assert stats['tilt_mode']==mode and stats['tilt_scale']==1.
                    assert stats['gradient_sources']==[] and not stats['one_backward_for_all_layers']
                    assert not stats['diagnostic_zero_tilt']
                    total['cached_images']+=b['generation_reused']
                    if b['generation_reused']:assert b['actual_model_calls_in_batch']==0
                    assert stats['full_event_log_includes_terminal_eos']
                    assert stats['maximum_probe_hidden_difference']==0 and stats['common_native_point']
                    assert stats['intervention_layers']==list(POST)
                    assert not any(stats[k] for k in ('diagnostic_force_probe_gate','diagnostic_reject_all','diagnostic_zero_replay'))
                    events=stats['full_joint_events'];decisions={d['step']:d for d in stats['full_class_decisions']}
                    gates={g['step']:g for g in stats['full_gate_events']}
                    assert len({e['step'] for e in events})==len(events)==stats['total_proposals']
                    assert stats['recycle_events']==[dict({k:v for k,v in e.items() if k not in ('emitted_token','terminal_eos')},generated_token=e['emitted_token']) for e in events if not e['terminal_eos']]
                    kept=[e for e in events if e['accepted']]
                    assert len(kept)==stats['total_accepted']
                    first=min((e['step'] for e in kept),default=len(tokens))
                    for d in decisions.values():
                        assert d['step']<=len(tokens)
                        prefix=tokenizer.decode(tokens[:d['step']],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                        prior=parser.classes(prefix)
                        assert d['token'] in classes and d['token'] not in tokens[:d['step']]
                        assert d['canonical']==classes[d['token']] and d['prior_classes']==sorted(prior)
                        assert d['blocked']==(d['canonical'] in prior)
                    for g in gates.values():
                        assert not decisions[g['step']]['blocked'] and decisions[g['step']]['token']==g['token']
                        assert g['threshold']==plan['real_threshold'] and g['positive']==(g['p_real']<plan['real_threshold'])
                    for e in events:
                        g=gates[e['step']]
                        assert g['positive'] and e['probe_p_real']==g['p_real'] and e['token']==g['token']
                        assert e['signal_guided']==False and e['source_layer']==e['target_layer']==19
                        assert e['bounded_visual_tilt'] and e['tilt_mode']==mode
                        assert e['probe_real_threshold']==plan['real_threshold'] and e['budget_max']>0
                        assert e['joint_layers']==list(POST) and set(e['layer_details'])==set(map(str,POST))
                        for layer_index in POST:
                            layer=e['layer_details'][str(layer_index)]
                            lo,hi,mean=layer['coefficient_min'],layer['coefficient_max'],layer['coefficient_mean']
                            drift=layer['drift']
                            assert 0<=lo<=hi<=1 and lo-1e-6<=mean<=hi+1e-6 and 0<=drift<=1
                            assert 0<=layer['visual_logit_shift_min']<=layer['visual_logit_shift_max']
                            assert math.isfinite(layer['visual_logit_shift_max']) and layer['bounded_positive_visual_tilt']
                            assert layer['budget_max']>=0 and math.isfinite(layer['budget_max'])
                            assert 'head_scales' not in layer and 'head_effects' not in layer
                            if mode=='fixed':assert drift==1 and lo==hi==.5 and mean==.5
                            elif mode=='drift':assert abs(lo-.5*drift)<1e-6 and abs(hi-.5*drift)<1e-6
                            else:assert hi<=drift+1e-6
                            total['per_layer_coefficients_checked']+=1
                        assert max(v['budget_max'] for v in e['layer_details'].values())==e['budget_max']
                        assert math.isfinite(e['actual_log_probability_change'])
                        assert e['accepted']==(e['actual_log_probability_change']<0)
                        assert e['native_top2_gap']>=0 and e['original_target_log_probability']<=0
                        assert e['immediate_token_changed']==(e['proposed_token']!=e['token'])
                        emitted=e['proposed_token'] if e['accepted'] else e['token']
                        assert e['emitted_token']==emitted and e['terminal_eos']==(e['step']==len(tokens))
                        if e['terminal_eos']:assert emitted in eos
                        else:assert tokens[e['step']]==emitted
                        if e['step']<=first:
                            assert tokens[:e['step']]==a['response_token_ids'][:e['step']]
                            assert e['token']==a['response_token_ids'][e['step']]
                    if not kept:
                        assert tokens==a['response_token_ids'] and b['generated_text']==a['generated_text']
                        total['no_kept_exact_images']+=1
                    total['images']+=1;total['proposals']+=len(events);total['kept']+=len(kept)
                    total['changed_images']+=tokens!=a['response_token_ids']
                    total['kept_token_changes']+=sum(e['immediate_token_changed'] for e in kept)
                    total['terminal_eos_accepts']+=stats['terminal_eos_accepts']
        assert total['images']==100 and total['cached_images']==9
        report['variants'][name]=total
    write(root/'mechanism_audit.json',report);print(__import__('json').dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);audit(p.parse_args().root)
