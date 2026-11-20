"""Keep full route audit; independently check strong and half-attenuated schedules."""

import argparse
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha,group_keys
from scripts.qwen3_caption_groups_with_cache import source_groups
from features.coco_object_history import ObjectClassHistory


def audit(root):
    from transformers import AutoTokenizer
    plan=read(root/'protocol.json');ids=read(root/'cohort.json')['coco100']
    assert len(set(ids))==100 and read(root/'run_status.json')['status']=='complete'
    assert plan['no_learned_gate'] and plan['route_k']==8 and plan['attenuation_scale']==.5
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    eos=plan['eos_token_ids'];eos=set(eos if isinstance(eos,list) else [eos])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(tokenizer,read(root/'object_lexicon.json')['token_ids'])
    report=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),variants={})
    for name in plan['candidates']:
        mode=plan['mode_by_candidate'][name]
        total=dict(images=0,proposals=0,kept=0,changed_images=0,no_kept_exact_images=0,
            kept_token_changes=0,gradients=0,layer_gate_checks=0,actual_route_calls=0,
            actual_eos_accepts=0,maximum_probe_hidden_difference=0.)
        for shard in (0,1):
            native=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
            current=source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native)==group_keys(current)
            assert {r['image_id'] for g in current for r in g}==set(ids[shard::2])
            for original_group,group in zip(native,current,strict=True):
                gate_steps={g['step'] for r in group for g in r['sar_ems_statistics']['full_gate_events']}
                event_steps={e['step'] for r in group for e in r['sar_ems_statistics']['full_joint_events']}
                assert {r['sar_ems_statistics']['gradient_probe_calls'] for r in group}=={len(gate_steps)}
                assert {r['sar_ems_statistics']['route_layer_calls'] for r in group}=={17*len(event_steps)}
                total['gradients']+=len(gate_steps);total['actual_route_calls']+=17*len(event_steps)
                for a,b in zip(original_group,group,strict=True):
                    s=b['sar_ems_statistics'];tokens=b['response_token_ids']
                    assert not b['generation_reused'] and b['decoder']=='native_qwen3_generate_same_span_batch'
                    assert s['probe'] is s['probe_real_threshold'] is None
                    assert s['no_learned_gate'] and s['route_mode']==mode and s['route_k']==8
                    assert s['mild_conflict_route'] and s['attenuation_scale']==.5
                    assert s['selected_text_gate_range']==[.5,.75]
                    assert s['probe_calls']==s['probe_positive_calls']==s['native_risk_solver_calls']==0
                    assert s['signal_guided'] and s['common_native_point'] and s['one_backward_for_all_layers']
                    assert s['maximum_probe_hidden_difference']==0 and not s['original108_features_computed']
                    assert s['visual_attention_weights_preserved_at_each_modified_attention']
                    assert s['intervention_layers']==s['gradient_sources']==list(range(19,36))
                    assert s['replayed_tail_layers']==list(range(20,36))
                    assert not any(s[k] for k in ['diagnostic_force_probe_gate','diagnostic_force_guidance',
                        'diagnostic_reject_all','diagnostic_zero_replay','diagnostic_zero_route','diagnostic_flat_signal'])
                    events=s['full_joint_events'];decisions={d['step']:d for d in s['full_class_decisions']}
                    gates={g['step']:g for g in s['full_gate_events']}
                    assert len({e['step'] for e in events})==len(events)==s['total_proposals']
                    kept=[e for e in events if e['accepted']];first=min((e['step'] for e in kept),default=len(tokens))
                    assert len(kept)==s['total_accepted']
                    for d in decisions.values():
                        step=d['step'];assert step<=len(tokens)
                        prior=parser.classes(tokenizer.decode(tokens[:step],skip_special_tokens=True,clean_up_tokenization_spaces=False))
                        assert d['token'] in classes and d['token'] not in tokens[:step]
                        assert d['canonical']==classes[d['token']] and d['prior_classes']==sorted(prior)
                        assert d['blocked']==(d['canonical'] in prior)
                    for step,g in gates.items():
                        assert g['gate_source']=='lexical_first_class_only' and g['positive']
                        assert g['p_real'] is g['threshold'] is None
                        assert not decisions[step]['blocked'] and decisions[step]['token']==g['token']
                    for e in events:
                        step=e['step'];assert step in gates and e['token']==gates[step]['token']
                        assert e['probe_p_real'] is e['probe_real_threshold'] is None
                        assert e['no_learned_gate'] and e['route_mode']==mode and e['signal_guided']
                        assert e['attenuation_scale']==.5
                        assert e['source_layer']==e['target_layer']==19 and e['joint_layers']==list(range(19,36))
                        assert set(e['layer_details'])==set(map(str,range(19,36)))
                        for index in range(19,36):
                            layer=e['layer_details'][str(index)]
                            assert layer['source_layer']==layer['target_layer']==index and layer['native_point']
                            arrays={key:np.asarray(layer[key],np.float32) for key in ['visual_effect','textual_effect','vri','raw_head_risk','applied_head_risk','text_gates']}
                            assert all(v.shape==(32,) and np.isfinite(v).all() for v in arrays.values())
                            vis,txt,g=arrays['visual_effect'],arrays['textual_effect'],arrays['text_gates']
                            conflict=(vis<0)&(txt>0);assert layer['native_conflict_mask']==conflict.tolist()
                            assert np.array_equal(g[~conflict],np.ones_like(g[~conflict]))
                            risk=arrays['raw_head_risk'];assert np.min(risk)>=-1e-6 and np.max(risk)<=1+1e-6
                            applied=np.roll(risk,16) if mode=='roll' else risk
                            np.testing.assert_array_equal(applied,arrays['applied_head_risk'])
                            vri=np.abs(vis)/(np.abs(vis)+np.abs(txt)+np.float32(1e-8))
                            np.testing.assert_allclose(vri,arrays['vri'],atol=1e-7,rtol=0)
                            heads=np.flatnonzero(conflict);heads=heads[np.argsort(vri[heads],kind='stable')]
                            if mode!='vri':heads=heads[np.argsort(-applied[heads],kind='stable')]
                            chosen=heads[:8];assert set(chosen.tolist())==set(layer['selected_heads'])
                            assert set(np.flatnonzero(g<1).tolist())==set(chosen.tolist())
                            schedule=np.linspace(0,.5,len(chosen),dtype=np.float32) if len(chosen)>1 else np.array([0.],np.float32) if len(chosen) else np.empty(0,np.float32)
                            strong=np.asarray(layer['strong_text_gates'],np.float32)
                            assert strong.shape==(32,) and np.isfinite(strong).all()
                            assert layer['attenuation_scale']==.5
                            assert np.array_equal(strong[~conflict],np.ones_like(strong[~conflict]))
                            np.testing.assert_allclose(strong[chosen],schedule,atol=2**-24,rtol=0)
                            expected=np.float32(1)-np.float32(.5)*(np.float32(1)-strong)
                            np.testing.assert_array_equal(g,expected)
                            assert np.min(g[chosen],initial=1)>=.5 and np.max(g[chosen],initial=0)<=.75
                            predicted=float(((g.astype(np.float64)-1)*txt.astype(np.float64)).sum())
                            tolerance=32*np.finfo(np.float32).eps*max(1.,abs(predicted))
                            assert abs(predicted-layer['predicted_log_probability_change'])<=tolerance
                            assert layer['predicted_log_probability_change']<=1e-6
                            total['layer_gate_checks']+=1
                        assert abs(e['predicted_log_probability_change']-sum(x['predicted_log_probability_change'] for x in e['layer_details'].values()))<1e-9
                        assert e['accepted']==(e['actual_log_probability_change']<0)
                        assert e['immediate_token_changed']==(e['proposed_token']!=e['token'])
                        emitted=e['proposed_token'] if e['accepted'] else e['token']
                        assert e['emitted_token']==emitted and e['emitted_is_eos']==(emitted in eos)
                        assert e['terminal_eos']==(step==len(tokens))
                        if step==len(tokens):assert emitted in eos
                        else:assert tokens[step]==emitted
                        if step<=first:
                            assert tokens[:step]==a['response_token_ids'][:step] and e['token']==a['response_token_ids'][step]
                    if not kept:
                        assert tokens==a['response_token_ids'] and b['generated_text']==a['generated_text'];total['no_kept_exact_images']+=1
                    assert s['actual_eos_accepts']==sum(e['emitted_is_eos'] for e in kept)
                    total['actual_eos_accepts']+=s['actual_eos_accepts'];total['images']+=1
                    total['proposals']+=len(events);total['kept']+=len(kept)
                    total['changed_images']+=tokens!=a['response_token_ids'];total['kept_token_changes']+=sum(e['immediate_token_changed'] for e in kept)
        assert total['images']==100;report['variants'][name]=total
    write(root/'mechanism_audit.json',report);print(__import__('json').dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);audit(p.parse_args().root)
