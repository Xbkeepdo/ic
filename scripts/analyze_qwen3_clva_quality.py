"""Separate full-cohort hallucination counts from repetition/denominator effects."""

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.audit_qwen3_fresh_caption500 import calculate
from scripts.qwen3_caption_groups_with_cache import source_groups


def prepare(root):
    target=root/'quality_analysis_request.json'
    assert not target.exists() and not (root/'quality_audit.json').exists()
    write(target,dict(fixed_at=time.time(),protocol_sha256=sha(root/'protocol.json'),
        source_sha256={str(Path(__file__)):sha(Path(__file__)),
            str(REPO/'scripts/audit_qwen3_fresh_caption500.py'):sha(REPO/'scripts/audit_qwen3_fresh_caption500.py')},
        ngram=8,repetition_count=4,primary='Full cohort only; unchanged sentence and legacy metrics.',
        all_pairs=True,diagnostics=['Token caps, max repeated8gram, unique/repeated real and false mentions.',
            'Drop each pair top3 increased real repetitions and top3 false-mention reductions, tie by image id.',
            'Common nonlooping and noncapped subsets defined by all arms; descriptive and output-dependent, never a replacement test set.'],
        decision='No physical correction claimed from seat/color aliases, repeated true-mention dilution, raw-method gains or single-image loops.'))


def run(root):
    request=read(root/'quality_analysis_request.json');plan=read(root/'protocol.json')
    assert request['protocol_sha256']==sha(root/'protocol.json')
    for name,wanted in request['source_sha256'].items():assert sha(Path(name))==wanted,name
    assert read(root/'analysis_provenance.json')['passed'] and not (root/'quality_audit.json').exists()
    ids=read(root/'cohort.json')[plan['cohort']];names=plan['variants']
    raw={n:{r['image_id']:r for s in range(plan['shards'])
        for g in source_groups(root/f'{plan["cohort"]}/{n}.shard{s}.jsonl') for r in g} for n in names}
    repetitions={n:{} for n in names}
    for name in names:
        for image in ids:
            tokens=raw[name][image]['response_token_ids'];counts=Counter(tuple(tokens[i:i+8]) for i in range(max(0,len(tokens)-7)))
            repetitions[name][image]=max(counts.values(),default=0)
    report=dict(passed=True,request_sha256=sha(root/'quality_analysis_request.json'),
        repeated8gram_threshold=4,repetition_is_descriptive=True,eval={})
    for mode in ['sentence','legacy']:
        folder=root/f'evaluations/{mode}_v1/{plan["cohort"]}'
        metric_file,=folder.glob('metrics.*.json');metrics=read(metric_file)
        ground=read(Path(metrics['chair_cache']))['objects']
        per_file,=folder.glob('chair_per_image*.jsonl')
        import json
        rows=[json.loads(line) for line in per_file.read_bytes().splitlines()]
        assert [r['image_id'] for r in rows]==ids
        scores={n:{r['image_id']:r['methods'][n] for r in rows} for n in names}
        true_unique={n:{} for n in names};activity={}
        for name in names:
            for image in ids:
                s=scores[name][image];value=s['object_recall']*len(ground[str(image)])
                assert abs(value-round(value))<1e-8 and round(value)<=s['real_mentions']
                true_unique[name][image]=round(value)
            summary=calculate([scores[name][i] for i in ids])
            for k in ['chair_s','chair_i','object_recall','object_f1']:
                assert abs(summary[k]-metrics['methods'][name][k])<1e-12
            activity[name]=dict(summary,unique_real_image_classes=sum(true_unique[name].values()),
                repeated_real_mentions=sum(scores[name][i]['real_mentions']-true_unique[name][i] for i in ids),
                caps=[i for i in ids if len(raw[name][i]['response_token_ids'])==plan['max_new_tokens']],
                looped_images=[i for i in ids if repetitions[name][i]>=4],
                max_repeated8gram=max(repetitions[name].values()),mean_tokens=sum(len(raw[name][i]['response_token_ids']) for i in ids)/len(ids))
        def summarize(selected):
            return {n:calculate([scores[n][i] for i in selected]) for n in names} if selected else None
        pairs={}
        for base,name in combinations(names,2):
            changes={i:dict(false_delta=scores[name][i]['hallucinated_mentions']-scores[base][i]['hallucinated_mentions'],
                true_repetition_delta=scores[name][i]['real_mentions']-true_unique[name][i]
                    -scores[base][i]['real_mentions']+true_unique[base][i],
                false_classes_base=scores[base][i]['hallucinated_objects'],false_classes_new=scores[name][i]['hallucinated_objects'],
                true_class_delta=true_unique[name][i]-true_unique[base][i]) for i in ids}
            false_order=sorted(ids,key=lambda i:(changes[i]['false_delta'],i))[:3]
            repetition_order=sorted(ids,key=lambda i:(-changes[i]['true_repetition_delta'],i))[:3]
            pairs[f'{name}__minus__{base}']=dict(
                fixed=[i for i in ids if scores[base][i]['chair_s']>scores[name][i]['chair_s']],
                introduced=[i for i in ids if scores[base][i]['chair_s']<scores[name][i]['chair_s']],
                false_count_delta=sum(c['false_delta'] for c in changes.values()),
                largest_false_reductions=[dict(image_id=i,**changes[i]) for i in false_order],
                largest_real_repetition_increases=[dict(image_id=i,**changes[i]) for i in repetition_order],
                drop_top3_false_reductions=dict(excluded=false_order,metrics=summarize([i for i in ids if i not in false_order])),
                drop_top3_real_repetition_increases=dict(excluded=repetition_order,metrics=summarize([i for i in ids if i not in repetition_order])),
                changed_cases={i:dict(**c,text_base=raw[base][i]['generated_text'],text_new=raw[name][i]['generated_text'])
                    for i,c in changes.items() if c['false_classes_base']!=c['false_classes_new'] or c['true_class_delta']!=0})
        common_nonloop=[i for i in ids if all(repetitions[n][i]<4 for n in names)]
        common_noncapped=[i for i in ids if all(len(raw[n][i]['response_token_ids'])<plan['max_new_tokens'] for n in names)]
        report['eval'][mode]=dict(activity=activity,pairs=pairs,
            common_nonloop=dict(ids=common_nonloop,metrics=summarize(common_nonloop)),
            common_noncapped=dict(ids=common_noncapped,metrics=summarize(common_noncapped)),
            interpretation='Full cohort primary. Subsets and removals diagnose output-dependent repetition and concentration, not confirm generalization.')
    report['per_image_max_repeated8gram']=repetitions
    write(root/'quality_audit.json',report)
    print({n:{k:v for k,v in a.items() if k in ['chair_s','chair_i','object_recall','object_f1','hallucinated_mentions',
        'object_mentions','unique_real_image_classes','repeated_real_mentions','mean_tokens','max_repeated8gram']}|
        dict(looped_images=len(a['looped_images']),caps=len(a['caps'])) for n,a in report['eval']['sentence']['activity'].items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare(args.root) if args.prepare else run(args.root)
