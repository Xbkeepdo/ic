"""Separate first object mentions from repeats and test held-out signal ranking."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


COLUMNS = ('signed_support', 'drift', 'ffn_gain', 'logit_gap', 'visual_mass')
LABELS = ('first_real', 'first_false', 'repeat_real', 'repeat_false')


def scores(y, pred):
    return {'auc': float(roc_auc_score(y, pred)), 'ap': float(average_precision_score(y, pred)),
            'false_rate': float(y.mean()), 'n': len(y)}


def within_category(events, y, pred):
    groups = defaultdict(list)
    for i, event in enumerate(events):
        groups[event['class']].append(i)
    weighted, pairs = 0.0, 0
    for indices in groups.values():
        selected = y[indices]
        n_pos = int(selected.sum())
        n_pairs = n_pos * (len(indices)-n_pos)
        if n_pairs:
            weighted += roc_auc_score(selected, pred[indices])*n_pairs
            pairs += n_pairs
    return {'auc': weighted/pairs if pairs else None, 'pairs': pairs}


def main(root):
    assert json.loads((root/'status.json').read_bytes())['status'] == 'complete'
    plan = json.loads((root/'protocol.json').read_bytes())
    events = []
    for path in sorted(root.glob('trace.shard*.jsonl')):
        events.extend(json.loads(line) for line in path.read_bytes().splitlines())
    assert len({(e['image_id'],e['step']) for e in events}) == len(events)
    assert all(set(e['features']) == {str(layer) for layer in plan['layers']} for e in events)
    n_match = sum(e['replay_matches_target'] for e in events)
    rows = [dict(event, **m) for event in events if event['replay_matches_target']
            for m in event['mentions'] if len(event['mentions']) == 1]
    output = {'positions': len(events), 'matching_positions': n_match, 'rows_in_analysis':len(rows),
              'images':len({e['image_id'] for e in rows}), 'disjoint_split': plan['split'], 'layers':{}}
    for name in LABELS:
        selected = [r for r in rows if r['repeat']==name.startswith('repeat') and r['hallucinated']==name.endswith('false')]
        output[name] = {'n':len(selected),'images':len({e['image_id'] for e in selected}),
                        'nouns':sum(e['token_is_noun'] for e in selected)}

    first = [r for r in rows if not r['repeat']]
    for layer in map(str,plan['layers']):
        summary = {'subsets':{},'gate':{}}
        for cohort, filtered in [('all', first), ('top8_noun', [r for r in first if r['token_is_noun'] and r['features'][layer]['in_top8']]),
                                 ('source_top_noun', [r for r in first if r['token_is_noun'] and r['features'][layer]['is_source_top']])]:
            y = np.array([r['hallucinated'] for r in filtered], dtype=np.int64)
            fold = np.array([r['fold'] for r in filtered])
            values = np.array([[r['features'][layer][k] for k in COLUMNS] for r in filtered],dtype=float)
            assert np.isfinite(values).all()
            means = dict(zip(COLUMNS, values.mean(0).tolist(), strict=True)) if len(y) else {}
            stat = {'n':len(y),'hallucinated':int(y.sum()),'means':means,
                    'individual':{},'models':{}}
            if len(set(y)) < 2 or min(sum(y[fold==0]),sum(y[fold==1])) < 5:
                summary['subsets'][cohort]=stat
                continue
            for col,sign in ((0,-1),(1,1),(2,1),(3,1),(4,-1)):
                pred=sign*values[:,col]
                stat['individual'][COLUMNS[col]]={**scores(y,pred),'by_coco_class':within_category(filtered,y,pred)}
            for name,cols in [('write',[0]),('drift',[1]),('write_drift',[0,1]),('gain_gap',[2,3]),('joint',[0,1,2,3])]:
                preds=np.zeros(len(y)); fits=[]; passed=np.zeros(len(y),dtype=bool)
                for validate in (0,1):
                    test=fold==validate
                    x=values[:,cols]; scaler=StandardScaler().fit(x[~test]); z=scaler.transform(x)
                    model=LogisticRegression(C=0.3,solver='lbfgs').fit(z[~test],y[~test])
                    preds[test]=model.decision_function(z[test]); train_scores=model.decision_function(z[~test])
                    threshold=np.quantile(train_scores[y[~test]==0],0.90,method='higher')
                    passed[test]=preds[test]>threshold
                    fits.append({'validate_fold':validate,'n_train':int((~test).sum()),'n_test':int(test.sum()),
                                 'test_scores':scores(y[test],preds[test]),'coefs':model.coef_[0].tolist(),
                                 'means':scaler.mean_.tolist(),'scales':scaler.scale_.tolist(),'intercept':float(model.intercept_[0]),
                                 'threshold':float(threshold)})
                stat['models'][name]={**scores(y,preds),'by_coco_class':within_category(filtered,y,preds),
                                      'false_trigger':float(passed[y==1].mean()),'real_trigger':float(passed[y==0].mean()),
                                      'folds':fits}
            summary['subsets'][cohort]=stat
        for label in LABELS:
            selected=[r for r in rows if r['repeat']==label.startswith('repeat') and r['hallucinated']==label.endswith('false')]
            summary['gate'][label]={'count':len(selected),
                'top8_noun':sum(r['token_is_noun'] and bool(r['features'][layer]['in_top8']) for r in selected),
                'top1_noun_positive_gain':sum(r['token_is_noun'] and bool(r['features'][layer]['is_source_top'])
                                              and r['features'][layer]['ffn_gain']>0 for r in selected)}
        output['layers'][layer]=summary
    path=root/'analysis.json'
    path.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print('ANALYZED',len(events),'positions,',n_match,'match,',len(first),'first mentions')
    for layer,s in output['layers'].items():
        print(layer,s['gate'])
        for subset,v in s['subsets'].items():
            print(subset,v['n'],v['hallucinated'],{n:(round(x['auc'],3),round(x['false_trigger'],3),round(x['real_trigger'],3))
                                      for n,x in v['models'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    main(p.parse_args().root)
