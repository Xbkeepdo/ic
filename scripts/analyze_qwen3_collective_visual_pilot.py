"""Use the existing paired analysis and add complete collective-gate records."""

import argparse
from collections import Counter
from pathlib import Path
import sys
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import analyze_qwen3_scalar_visual_pilot as core
from scripts.qwen3_caption_groups_with_cache import source_groups
from scripts.run_qwen3_signed_feedback import read,write,sha


def analyze(root):
    original_brief=core.brief_event
    def brief(event,tokens,tokenizer):
        result=original_brief(event,tokens,tokenizer)
        if 'collective_visual_effect' in event:
            result.update(collective_visual_effect=event['collective_visual_effect'],
                collective_visual_support_veto=event['collective_visual_support_veto'])
        return result
    with patch.object(core,'brief_event',brief):core.analyze(root)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    decisions=[];batches=[]
    for shard in (0,1):
        for group in source_groups(root/f'coco100/collective.shard{shard}.jsonl'):
            batches.append(group[0]['sar_ems_statistics'])
            for row in group:
                tokens=row['response_token_ids']
                for value in row['sar_ems_statistics']['full_collective_decisions']:
                    decisions.append(dict(value,image_id=row['image_id'],
                        target_text=tokenizer.decode([value['token']],skip_special_tokens=True),
                        prefix_context=tokenizer.decode(tokens[max(0,value['step']-12):value['step']],skip_special_tokens=True)))
    summary=dict(passed=True,source_sha256=sha(Path(__file__)),
        scope='Nonzero original negative-head budgets before the veto; all-SAM-positive backward cost is separately counted, including no final budget.',
        decision_count=len(decisions),vetoed=sum(d['veto'] for d in decisions),
        retained_for_actual_replay=sum(not d['veto'] for d in decisions),
        vetoed_target_texts=dict(Counter(d['target_text'] for d in decisions if d['veto'])),
        actual_backward_calls=sum(b['gradient_probe_calls'] for b in batches))
    write(root/'collective_activity.json',summary)
    write(root/'collective_gate_cases.json',decisions)
    print(__import__('json').dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
