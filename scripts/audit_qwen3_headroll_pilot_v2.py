"""Audit the frozen head permutation and complete causal event logs."""

import argparse
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from features.coco_object_history import ObjectClassHistory
from features.qwen3_block_acg_signal import POST
from scripts.audit_qwen3_caption_with_native_cache_v2 import validate_cached_group
from scripts.run_qwen3_signed_feedback import read,write,sha,group_keys
from scripts.qwen3_caption_groups_with_cache import source_groups


def audit(root):
    from transformers import AutoTokenizer
    plan=read(root/'protocol.json')
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    eos=set(tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id,list) else [tokenizer.eos_token_id])
    config=read(Path('/root/rivermind-data/models/qwen3_vl_8b/generation_config.json'))
    value=config.get('eos_token_id',[]);eos.update(value if isinstance(value,list) else [value])
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(tokenizer,read(root/'object_lexicon.json')['token_ids'])
    total=dict(images=0,proposals=0,kept=0,changed_images=0,no_kept_exact_images=0,
        kept_token_changes=0,gradient_calls=0,reused_images=0,terminal_eos_accepts=0,
        positive_prediction_events=0,per_layer_scale_vectors_checked=0)
    for shard in (0,1):
        native=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')
        current=source_groups(root/f'coco100/headroll.shard{shard}.jsonl')
        assert group_keys(native)==group_keys(current)
        for original_group,group in zip(native,current,strict=True):
            validate_cached_group(group)
            positive_steps={g['step'] for r in group for g in r['sar_ems_statistics']['full_gate_events'] if g['positive']}
            actual_calls={r['sar_ems_statistics']['gradient_probe_calls'] for r in group}
            assert actual_calls=={len(positive_steps)}
            total['gradient_calls']+=len(positive_steps)
            for a,b in zip(original_group,group,strict=True):
                tokens,stats=b['response_token_ids'],b['sar_ems_statistics']
                assert stats['signal_guided'] and stats['permuted_head_control'] and stats['head_scale_roll_by']==10
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
                    assert g['threshold']==plan['real_threshold']
                    assert g['positive']==(g['p_real']<plan['real_threshold'])
                for e in events:
                    g=gates[e['step']]
                    assert g['positive'] and e['probe_p_real']==g['p_real'] and e['token']==g['token']
                    assert e['source_layer']==e['target_layer']==19 and e['joint_layers']==list(POST)
                    assert set(e['layer_details'])==set(map(str,POST))
                    prediction=0.
                    for index in POST:
                        layer=e['layer_details'][str(index)];scales=layer['head_scales'];effects=layer['head_effects']
                        assert len(scales)==len(effects)==32 and all(0<=x<=1.400001 for x in scales)
                        assert sum(x>0 for x in scales)==layer['nonzero_heads'] and 0<=layer['drift']<=1
                        negative=[max(0,-x) for x in effects];peak=max(negative)
                        expected=[1.4*layer['drift']*n/peak if peak>0 else 0. for n in negative]
                        expected=expected[-10:]+expected[:-10]
                        assert all(abs(x-y)<1e-6 for x,y in zip(scales,expected,strict=True))
                        value=sum(x*y for x,y in zip(scales,effects,strict=True))
                        assert abs(value-layer['predicted_log_probability_change'])<1e-5
                        prediction+=value;total['per_layer_scale_vectors_checked']+=1
                    assert abs(prediction-e['predicted_log_probability_change'])<1e-4
                    total['positive_prediction_events']+=prediction>0
                    assert math.isfinite(e['actual_log_probability_change'])
                    assert e['accepted']==(e['actual_log_probability_change']<0)
                    assert e['native_top2_gap']>=0 and e['original_target_log_probability']<=0
                    assert e['immediate_token_changed']==(e['proposed_token']!=e['token'])
                    emitted=e['proposed_token'] if e['accepted'] else e['token']
                    assert e['emitted_token']==emitted and e['terminal_eos']==(e['step']==len(tokens))
                    if e['terminal_eos']:
                        assert emitted in eos
                    else:assert tokens[e['step']]==emitted
                    if e['step']<=first:
                        assert tokens[:e['step']]==a['response_token_ids'][:e['step']]
                        assert e['token']==a['response_token_ids'][e['step']]
                if not kept:
                    assert tokens==a['response_token_ids'] and b['generated_text']==a['generated_text']
                    total['no_kept_exact_images']+=1
                total['images']+=1;total['proposals']+=len(events);total['kept']+=len(kept)
                total['kept_token_changes']+=sum(e['immediate_token_changed'] for e in kept)
                total['terminal_eos_accepts']+=stats['terminal_eos_accepts']
                total['changed_images']+=tokens!=a['response_token_ids']
                total['reused_images']+=b['generation_reused']
    assert total['images']==100 and total['reused_images']==9
    for shard,info in plan['recovery_prefixes'].items():
        source=Path(info['path']);assert sha(source)==info['sha256']
        assert (root/f'coco100/headroll.shard{shard}.jsonl').read_bytes()[:info['bytes']]==source.read_bytes()
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),variants={'headroll':total},
        all96_original_records_preserved_bytewise=True,scoped_generated_token_equals_full_emitted_token=True)
    write(root/'mechanism_audit.json',result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);audit(p.parse_args().root)
