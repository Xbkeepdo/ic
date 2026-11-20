"""Audit whether visual feedback changes which objects are mentioned or only their counts."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def summarize(rows):
    hallucinated = [{term[1] for term in row['hallucinated_objects']} for row in rows]
    false_mentions = sum(row['hallucinated_mentions'] for row in rows)
    total_mentions = sum(row['object_mentions'] for row in rows)
    unique = sum(map(len,hallucinated))
    np.testing.assert_equal(sum(bool(classes) for classes in hallucinated), sum(row['chair_s'] for row in rows))
    return {'images':len(rows), 'hallucinated_images':sum(bool(classes) for classes in hallucinated),
            'false_mentions':false_mentions, 'total_mentions':total_mentions, 'false_image_classes':unique,
            'repeated_false_mentions':false_mentions-unique,'chair_i':false_mentions/max(1,total_mentions)}


def compare(left, right, ids, draws):
    def false_set(row):
        return {x[1] for x in row['hallucinated_objects']}

    left_sets, right_sets = list(map(false_set,left)), list(map(false_set,right))
    removed, added = Counter(), Counter()
    for a,b in zip(left_sets,right_sets,strict=True):
        removed.update(a-b)
        added.update(b-a)
    removed_mentions = np.array([a['hallucinated_mentions']-b['hallucinated_mentions'] for a,b in zip(left,right,strict=True)])
    diff_classes = np.array([len(b)-len(a) for a,b in zip(left_sets,right_sets,strict=True)])
    label_diff = np.array([b['chair_s']-a['chair_s'] for a,b in zip(left,right,strict=True)])
    rank = np.argsort(-removed_mentions,kind='stable')
    largest = [{'image_id':ids[i],'base':left[i]['hallucinated_mentions'],'candidate':right[i]['hallucinated_mentions']}
               for i in rank[:3]]
    return {
        'net_changed_hallucinated_images':int(label_diff.sum()),
        'fixed_hallucinated_images':int((label_diff<0).sum()),
        'introduced_hallucinated_images':int((label_diff>0).sum()),
        'removed_image_classes':removed, 'introduced_image_classes':added,
        'net_added_image_classes':int(diff_classes.sum()),
        'net_added_image_classes_ci95':np.quantile(diff_classes[draws].sum(1),[.025,.975]).tolist(),
        'net_added_false_mentions':int(-removed_mentions.sum()),
        'net_added_repeated_mentions':int((-removed_mentions-diff_classes).sum()),
        'largest_reductions':largest,
        'drop_largest_reduction':{'base':summarize([r for i,r in enumerate(left) if i!=rank[0]]),
                                  'candidate':summarize([r for i,r in enumerate(right) if i!=rank[0]])},
        'drop_top3':{'base':summarize([left[i] for i in rank[3:]]),'candidate':summarize([right[i] for i in rank[3:]])},
    }


def main(root):
    protocol=load(root/'protocol.json')
    assert load(root/'run_status.json')['status']=='complete'
    ids=load(root/'cohort.json')[protocol['cohort']]
    draws=np.random.default_rng(43).integers(0,len(ids),size=(5000,len(ids)))
    output={}
    for mode in ('sentence','legacy'):
        path,=(root / f'evaluations/{mode}_v1/{protocol["cohort"]}').glob('chair_per_image*.jsonl')
        rows=[json.loads(line) for line in path.read_bytes().splitlines()]
        assert [r['image_id'] for r in rows]==ids
        values={name:[r['methods'][name] for r in rows] for name in protocol['variants']}
        pairs={}
        for name,records in values.items():
            if name not in protocol['candidates']:
                continue
            for baseline in ('sadi','sadi_low_visual_uniform_29','sadi_sar_ems_next_19_20'):
                if baseline in values and name!=baseline:
                    pairs[f'{name} vs {baseline}']=compare(values[baseline],records,ids,draws)
        output[mode]={'metrics':{name:summarize(records) for name,records in values.items()},'pairs':pairs}
    (root/'object_mention_audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    for mode,v in output.items():
        print(mode,v['metrics'])
        for name,stat in v['pairs'].items():
            print(name,{k:v for k,v in stat.items() if k.startswith(('net_','largest_'))})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
