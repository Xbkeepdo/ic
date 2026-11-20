"""Explain first-proposal direction signs using already recorded gradients only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha


def analyze(root):
    from transformers import AutoTokenizer
    assert read(root/'post_audit_status.json')['status']=='passed'
    assert read(root/'mechanism_audit.json')['passed']
    assert not (root/'direction_signs.json').exists()
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    inputs={};records={}
    for name in ('sadi','acg_fixed','acg_guided'):
        rows={}
        for shard in (0,1):
            p=root/f'coco500/{name}.shard{shard}.jsonl';inputs[str(p)]=sha(p)
            rows.update((r['image_id'],r) for r in map(json.loads,p.read_bytes().splitlines()))
        records[name]=rows
    counter=Counter();examples={};all_rows=[]
    for image,guided_row in records['acg_guided'].items():
        events=guided_row['sar_ems_statistics']['full_joint_events']
        if not events:continue
        e=events[0];step=e['step'];native=records['sadi'][image]['response_token_ids']
        assert guided_row['response_token_ids'][:step]==native[:step] and e['token']==native[step]
        plain=attenuated=0.
        for layer in e['layer_details'].values():
            effect=sum(layer['head_effects']);plain+=1.4*effect
            attenuated+=1.4*layer['drift']*effect
        value=dict(image_id=image,step=step,target=tokenizer.decode([e['token']]),
            probe_p_real=e['probe_p_real'],native_top2_gap=e['native_top2_gap'],
            all_head_first_order=plain,all_head_drift_first_order=attenuated,
            all_head_additive_drift_first_order=plain+attenuated,
            selected_negative_heads_first_order=e['predicted_log_probability_change'],
            guided_actual_log_probability_change=e['actual_log_probability_change'])
        counter['first_native_proposals']+=1
        counter['all_head_positive_guided_negative']+=plain>1e-6 and e['predicted_log_probability_change']< -1e-6
        counter['all_head_drift_positive']+=attenuated>1e-6
        counter['all_head_additive_positive']+=plain+attenuated>1e-6
        fixed_events=records['acg_fixed'][image]['sar_ems_statistics']['full_joint_events']
        if fixed_events:
            f=fixed_events[0]
            same=(f['step']==step and f['token']==e['token'] and
                f['original_target_log_probability']==e['original_target_log_probability'] and
                records['acg_fixed'][image]['response_token_ids'][:step]==native[:step])
            if same:
                counter['matched_original_native_point']+=1
                value['fixed_actual_log_probability_change']=f['actual_log_probability_change']
                value['fixed_accepted']=f['accepted']
                reversal=f['actual_log_probability_change']>0 and e['actual_log_probability_change']<0
                counter['fixed_actual_increases_guided_decreases']+=reversal
                if reversal:examples[str(image)]=value
        all_rows.append(value)
    output=dict(scope='Exploratory diagnosis after independent500, using existing first-native gradients. Linear predictions are not generation outcomes or hallucination labels.',
        passed=True,counts=dict(counter),matched_sign_reversal_examples=examples,first_proposals=all_rows,
        input_sha256=inputs,source_sha256=sha(Path(__file__)),new_neural_computation=False,
        interpretation='Selecting only negative target effects can reverse the collective visual direction. Determine target correctness from actual images; do not label these targets real from a positive derivative alone.')
    write(root/'direction_signs.json',output)
    print(json.dumps(dict(counts=dict(counter),example_ids=sorted(map(int,examples)),
        selected={i:examples[i] for i in ('103027','416240','62540','523212') if i in examples}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
