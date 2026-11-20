"""Export every saved causal event, including unchanged final captions.

The existing changed-caption summaries intentionally omit some unchanged
captions. This CPU-only export adds readable token contexts without model
inference, refitting, rescoring, or modifying any original result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.analyze_qwen3_soft_conflict_route_pilot import brief_event


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda:handle.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def export(root):
    from transformers import AutoTokenizer
    import torch
    read=lambda p:json.loads(p.read_bytes())
    assert read(root/'run_status.json')['status']=='complete'
    assert read(root/'post_audit_status.json')['status']=='passed'
    destination=root/'complete_event_briefs.json'
    assert not destination.exists()
    plan=read(root/'protocol.json');ids=read(root/'cohort.json')[plan['cohort']]
    inputs={str(p):sha(p) for p in [Path(__file__),
        REPO/'scripts/analyze_qwen3_soft_conflict_route_pilot.py',
        root/'protocol.json',root/'cohort.json',root/'post_audit_status.json']}
    tokenizer=AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b',local_files_only=True)
    images={str(image):{} for image in ids}
    for method in plan['variants']:
        count=0
        for shard in (0,1):
            path=root/f'{plan["cohort"]}/{method}.shard{shard}.jsonl';inputs[str(path)]=sha(path)
            for line in path.read_bytes().splitlines():
                row=json.loads(line);image=str(row['image_id']);assert method not in images[image]
                images[image][method]=[brief_event(event,row['response_token_ids'],tokenizer)
                    for event in row['sar_ems_statistics'].get('full_joint_events',[])]
                count+=1
        assert count==len(ids)
    assert all(set(methods)==set(plan['variants']) for methods in images.values())
    assert not torch.cuda.is_initialized()
    for path,value in inputs.items():assert sha(Path(path))==value,path
    result=dict(passed=True,images=images,input_sha256=inputs,
        scope='All raw saved events, including captions whose final tokens equal SADI; not a new trajectory.',
        no_new_model_calls=True,cuda_initialized=False)
    with destination.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps(dict(passed=True,images=len(images),methods=len(plan['variants']),
        events={method:sum(len(methods[method]) for methods in images.values()) for method in plan['variants']},
        immediate_accepted={method:sum(e['accepted'] and e['immediate_token_changed']
            for methods in images.values() for e in methods[method]) for method in plan['candidates']},
        sha256=sha(destination))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    export(parser.parse_args().root)
