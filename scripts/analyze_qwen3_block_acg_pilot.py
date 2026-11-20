"""Paired object-count changes and joint-step diagnostics for manual review."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
from scripts.audit_qwen3_visual_feedback_mentions import compare
from scripts.run_qwen3_signed_feedback import read,write,sha,source_groups


def analyze(root):
    plan=read(root/'protocol.json')
    assert read(root/'mechanism_audit.json')['passed'] and read(root/'independent_summary.json')['passed']
    raw,groups={},{}
    for name in plan['variants']:
        groups[name]=[g for s in (0,1) for g in source_groups(root/f'coco100/{name}.shard{s}.jsonl')]
        raw[name]={r['image_id']:r for g in groups[name] for r in g}
    ids=read(root/'cohort.json')['coco100'];draws=np.random.default_rng(43).integers(0,len(ids),size=(5000,len(ids)))
    summaries,cases,activity={},{},{}
    for mode in ('sentence','legacy'):
        path,=(root/f'evaluations/{mode}_v1/coco100').glob('chair_per_image*.jsonl')
        rows=[json.loads(l) for l in path.read_text().splitlines()];assert [r['image_id'] for r in rows]==ids
        scores={name:[r['methods'][name] for r in rows] for name in plan['variants']}
        summaries[mode]={name:{ref:compare(scores[ref],scores[name],ids,draws) for ref in plan['variants'] if ref!=name} for name in plan['candidates']}
        if mode=='sentence':
            by_id={r['image_id']:r['methods'] for r in rows}
            for name in plan['candidates']:
                cases[name]={}
                for image in ids:
                    if raw[name][image]['generated_text']==raw['sadi'][image]['generated_text']:continue
                    a,b=by_id[image]['sadi'],by_id[image][name]
                    cases[name][str(image)]=dict(false_mentions_changed=Counter(map(tuple,a['hallucinated_objects']))!=Counter(map(tuple,b['hallucinated_objects'])),
                        methods={m:dict(text=raw[m][image]['generated_text'],scores=by_id[image][m],events=raw[m][image]['sar_ems_statistics'].get('recycle_events',[])) for m in plan['variants']})
    for name in plan['candidates']:
        batches=[g[0]['sar_ems_statistics'] for g in groups[name]]
        events=[e for r in raw[name].values() for e in r['sar_ems_statistics']['recycle_events']]
        kept=[e for e in events if e['accepted']]
        activity[name]=dict(proposals=len(events),kept=len(kept),
            active_images=sum(any(e['accepted'] for e in r['sar_ems_statistics']['recycle_events']) for r in raw[name].values()),
            changed_text_images=len(cases[name]),false_multiset_changed_images=sum(r['false_mentions_changed'] for r in cases[name].values()),
            gradient_calls=sum(s['gradient_probe_calls'] for s in batches),
            semantic_head_calls=sum(s['static_head_calls'] for s in batches),static_hits=sum(s['shared_static_hits'] for s in batches),
            kept_immediate_token_changes=sum(e['immediate_token_changed'] for e in kept),
            positive_margin_kept_without_token_change=sum(e['native_top2_gap']>0 and not e['immediate_token_changed'] for e in kept),
            mean_native_top2_gap=sum(e['native_top2_gap'] for e in events)/max(1,len(events)),
            mean_accepted_log_probability_change=sum(e['actual_log_probability_change'] for e in kept)/max(1,len(kept)))
    write(root/'paired_mentions.json',summaries);write(root/'changed_text_cases.json',cases);write(root/'activity.json',activity)
    write(root/'analysis_provenance.json',dict(passed=True,source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(root/'protocol.json'),image_review='pending',advance_to_500=False))
    print(json.dumps(dict(activity=activity,screen=read(root/'screening.json')),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);analyze(p.parse_args().root)
