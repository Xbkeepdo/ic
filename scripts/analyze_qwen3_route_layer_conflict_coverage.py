"""Check whole-layer visual/text signs at shared pre-intervention native points."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def analyze(repo,output):
    assert not output.exists()
    source=repo/'outputs/qwen3_conflict_route_pilot_20261119'
    labels_path=repo/'outputs/qwen3_full_risk_f_trace_20261118/labelled_positions.json'
    names=['sadi','route_vri','route_signal','route_roll']
    paths=[source/'protocol.json',source/'final_provenance.json',labels_path,Path(__file__)]
    paths += [source/f'coco100/{name}.shard{s}.jsonl' for name in names for s in (0,1)]
    inputs={str(p):sha(p) for p in paths};assert read(source/'final_provenance.json')['passed']
    output.mkdir()
    protocol=dict(fixed_at=time.time(),source_sha256=inputs,neural_generation=False,refitting=False,
        scope='Post-hoc closed development100; only points before any accepted action in all three methods.',
        hypothesis='Head-level conflict may coexist with positive total visual support. Test a single whole-layer conflict criterion, not a threshold search.',
        fixed_layer_criterion='sum_h visual_effect<0 and sum_h textual_effect>0',
        membership='Require all three native effect vectors exactly equal and original SADI prefix/target agreement.',
        labels='Archived automatic mention labels, not verified physical hallucination ground truth.',
        followup='This diagnostic alone does not establish generation utility; no change to the completed mild run or reserved500.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    labelled={(r['image_id'],r['step'],r['token_id']):r for r in read(labels_path)}
    rows={name:{} for name in names}
    for name in names:
        for shard in (0,1):
            for line in (source/f'coco100/{name}.shard{shard}.jsonl').read_bytes().splitlines():
                r=json.loads(line);s=r['sar_ems_statistics']
                rows[name][r['image_id']]=dict(tokens=r['response_token_ids'],
                    events=s.get('full_joint_events',[]),decisions={d['step']:d for d in s.get('full_class_decisions',[])})
    records=[];excluded=[]
    for image in read(source/'cohort.json')['coco100']:
        methods={n:rows[n][image] for n in names[1:]}
        cut=min((e['step'] for r in methods.values() for e in r['events'] if e['accepted']),default=513)
        events={n:{e['step']:e for e in r['events'] if e['step']<=cut} for n,r in methods.items()}
        for step in sorted(set.intersection(*(set(v) for v in events.values()))):
            selected={n:table[step] for n,table in events.items()};reference=selected[names[1]]
            assert len({e['token'] for e in selected.values()})==1
            assert reference['token']==rows['sadi'][image]['tokens'][step]
            assert all(r['tokens'][:step]==rows['sadi'][image]['tokens'][:step] for r in methods.values())
            if not all(selected[n]['layer_details'][str(i)][key]==reference['layer_details'][str(i)][key]
                for n in names[2:] for i in range(19,36) for key in ['visual_effect','textual_effect']):
                excluded.append(dict(image_id=image,step=step,reason='Native effect vectors not bitwise equal.'));continue
            layer=[]
            for i in range(19,36):
                d=reference['layer_details'][str(i)];vis=d['visual_effect'];txt=d['textual_effect']
                total_vis=math.fsum(vis);total_txt=math.fsum(txt)
                head_count=sum(v<0 and t>0 for v,t in zip(vis,txt,strict=True))
                layer.append(dict(layer=i,total_visual_effect=total_vis,total_textual_effect=total_txt,
                    head_conflict_count=head_count,whole_layer_conflict=total_vis<0 and total_txt>0))
            key=(image,step,reference['token']);label=labelled.get(key)
            records.append(dict(image_id=image,step=step,token_id=reference['token'],
                canonical=methods[names[1]]['decisions'][step]['canonical'],
                automatic_hallucination=label['automatic_hallucination'] if label else None,
                context=label['context'] if label else None,
                layers_with_head_conflict=sum(d['head_conflict_count']>0 for d in layer),
                layers_with_whole_layer_conflict=sum(d['whole_layer_conflict'] for d in layer),layers=layer))
    groups={}
    for name,value in [('automatic_real',False),('automatic_hallucination',True),('unlabelled',None)]:
        group=[r for r in records if r['automatic_hallucination'] is value]
        groups[name]=dict(points=len(group),head_conflict_layers=sum(r['layers_with_head_conflict'] for r in group),
            whole_layer_conflict_layers=sum(r['layers_with_whole_layer_conflict'] for r in group),
            no_whole_layer_conflict_points=sum(r['layers_with_whole_layer_conflict']==0 for r in group))
    result=dict(passed=True,protocol_sha256=sha(output/'protocol.json'),shared_native_points=len(records),
        excluded_not_exact=len(excluded),groups=groups,records=records,exclusions=excluded,
        limitation='Only common pre-action points are covered. Missing later native counterfactuals cannot be inferred from altered trajectories. Automatic labels retain semantic ambiguity.',
        new_generation_or_threshold_search=False,validation_untouched=True)
    for n,v in inputs.items():assert sha(Path(n))==v,n
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','exclusions']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repo',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();analyze(args.repo,args.output)
