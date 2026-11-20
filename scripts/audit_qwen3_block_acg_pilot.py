"""Audit joint head scales, native margins and causal acceptance paths."""

import argparse
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from features.coco_object_history import ObjectClassHistory
from features.qwen3_block_acg_signal import POST
from scripts.run_qwen3_signed_feedback import read,write,sha,source_groups,group_keys


def audit(root):
    from transformers import AutoTokenizer
    plan=read(root/'protocol.json')
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(tokenizer,read(root/'object_lexicon.json')['token_ids'])
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),variants={})
    for name in plan['candidates']:
        total=dict(images=0,proposals=0,kept=0,changed_images=0,no_kept_exact_images=0,
            proposed_token_changes=0,kept_token_changes=0,gradient_calls=0,zero_margin_events=0)
        for shard in (0,1):
            native=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
            current=source_groups(root/f'coco100/{name}.shard{shard}.jsonl')
            assert group_keys(native)==group_keys(current)
            for original_group,group in zip(native,current,strict=True):
                positive_steps={g['step'] for r in group for g in r['sar_ems_statistics']['gate_events'] if g['positive']}
                actual_calls={r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
                assert len(actual_calls)==1
                expected_calls=len(positive_steps) if group[0]['sar_ems_statistics']['signal_guided'] else 0
                assert actual_calls=={expected_calls}
                total['gradient_calls']+=expected_calls
                for a,b in zip(original_group,group,strict=True):
                    tokens,stats=b['response_token_ids'],b['sar_ems_statistics']
                    assert not any(stats[k] for k in ('diagnostic_force_probe_gate','diagnostic_reject_all','diagnostic_zero_replay'))
                    assert stats['maximum_probe_hidden_difference']==0 and stats['common_native_point']
                    assert stats['intervention_layers']==list(POST)
                    events=stats['recycle_events'];decisions={d['step']:d for d in stats['class_decisions']}
                    gates={g['step']:g for g in stats['gate_events']}
                    assert len(events)==stats['guided_attention_calls']
                    assert len({e['step'] for e in events})==len(events)
                    kept=[e for e in events if e['accepted']]
                    first=min((e['step'] for e in kept),default=len(tokens))
                    for d in decisions.values():
                        prefix=tokenizer.decode(tokens[:d['step']],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                        prior=parser.classes(prefix)
                        assert d['token'] in classes and d['token'] not in tokens[:d['step']]
                        assert d['canonical']==classes[d['token']] and d['prior_classes']==sorted(prior)
                        assert d['blocked']==(d['canonical'] in prior)
                    for g in gates.values():
                        assert not decisions[g['step']]['blocked'] and decisions[g['step']]['token']==g['token']
                        assert g['threshold']==plan['real_threshold']
                        assert g['positive']==(g['p_real']<plan['real_threshold'])
                    for e in events:
                        g=gates[e['step']]
                        assert g['positive'] and e['probe_p_real']==g['p_real'] and e['token']==g['token']
                        assert e['source_layer']==e['target_layer']==19 and e['joint_layers']==list(POST)
                        assert set(e['layer_details'])==set(map(str,POST))
                        prediction=0.
                        for index in POST:
                            layer=e['layer_details'][str(index)];scales=layer['head_scales']
                            assert len(scales)==32 and all(0<=x<=1.400001 for x in scales)
                            assert sum(x>0 for x in scales)==layer['nonzero_heads']
                            assert 0<=layer['drift']<=1
                            if e['signal_guided']:
                                effects=layer['head_effects'];negative=[max(0,-x) for x in effects];peak=max(negative)
                                expected=[1.4*layer['drift']*n/peak if peak>0 else 0. for n in negative]
                                assert all(abs(x-y)<1e-6 for x,y in zip(scales,expected,strict=True))
                                value=sum(x*y for x,y in zip(scales,effects,strict=True))
                                assert abs(value-layer['predicted_log_probability_change'])<1e-5
                                assert value<=1e-6;prediction+=value
                            else:assert all(abs(x-1.4)<1e-6 for x in scales)
                        if e['signal_guided']:
                            assert abs(prediction-e['predicted_log_probability_change'])<1e-4
                            assert prediction<=1e-6
                        assert math.isfinite(e['actual_log_probability_change'])
                        assert e['accepted']==(e['actual_log_probability_change']<0)
                        assert e['native_top2_gap']>=0 and e['original_target_log_probability']<=0
                        assert e['immediate_token_changed']==(e['proposed_token']!=e['token'])
                        assert tokens[e['step']]==(e['proposed_token'] if e['accepted'] else e['token'])
                        if e['step']<=first:
                            assert tokens[:e['step']]==a['response_token_ids'][:e['step']]
                            assert e['token']==a['response_token_ids'][e['step']]
                        total['zero_margin_events']+=e['native_top2_gap']==0
                    if not kept:
                        assert tokens==a['response_token_ids'] and b['generated_text']==a['generated_text']
                        total['no_kept_exact_images']+=1
                    total['images']+=1;total['proposals']+=len(events);total['kept']+=len(kept)
                    total['proposed_token_changes']+=sum(e['immediate_token_changed'] for e in events)
                    total['kept_token_changes']+=sum(e['immediate_token_changed'] for e in kept)
                    total['changed_images']+=tokens!=a['response_token_ids']
        assert total['images']==100
        result['variants'][name]=total
    write(root/'mechanism_audit.json',result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);audit(p.parse_args().root)
