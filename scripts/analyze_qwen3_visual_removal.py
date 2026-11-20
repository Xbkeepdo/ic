"""Audit pairwise visual and FFN contrast on first mentions, with category controls."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def paired_auc(rows, features):
    labels = np.array([m['hallucinated'] for _, m in rows], dtype=int)
    by_category = defaultdict(list)
    by_image = defaultdict(list)
    for i, (event, mention) in enumerate(rows):
        by_category[mention['class']].append(i)
        by_image[event['image_id']].append(i)

    def score(indices):
        y, x = labels[indices], features[indices]
        n_pos = int(y.sum())
        if n_pos in (0, len(y)):
            return 0, 0
        return roc_auc_score(y, x), n_pos * (len(y)-n_pos)

    def aggregate(groups):
        scores = [score(ids) for ids in groups.values()]
        denom = sum(n for _, n in scores)
        return {'auc': sum(a*n for a,n in scores)/denom if denom else None, 'pairs': denom}

    return {'pooled': score(np.arange(len(rows)))[0], 'within_category': aggregate(by_category),
            'within_image': aggregate(by_image)}


def analyze(root):
    plan = json.loads((root/'protocol.json').read_bytes())
    assert json.loads((root/'status.json').read_bytes())['status'] == 'complete'
    labels = Path(next(path for path in plan['contrast_sources'] if path.endswith('trace.shard0.jsonl'))).parent
    old = {(r['image_id'],r['step']):r for p in labels.glob('trace.shard*.jsonl')
           for r in [json.loads(line) for line in p.read_bytes().splitlines()]}
    new = [json.loads(line) for p in root.glob('trace.shard*.jsonl') for line in p.read_bytes().splitlines()]
    assert len(new) == len(old) and {(r['image_id'],r['step']) for r in new} == set(old)
    assert all(r['counterfactual_replay_matches'] and len(r['contrast']) == len(plan['contrast_layers']) for r in new)
    for r in new:
        assert r['mentions'] == old[r['image_id'],r['step']]['mentions']
        assert r['target_token_id'] == old[r['image_id'],r['step']]['target_token_id']
    report = {'positions': len(new), 'replay_matches': sum(r['counterfactual_replay_matches'] for r in new), 'layers': {}}
    first = [(r, r['mentions'][0]) for r in new if len(r['mentions']) == 1 and not r['mentions'][0]['repeat']]
    for layer in map(str, plan['contrast_layers']):
        groups = {'all': first, 'plausible_entity': [(r,m) for r,m in first if r['token_is_noun']
                  and r['contrast'][layer]['target_in_top8'] and r['contrast'][layer]['target_gap'] <= 2
                  and r['contrast'][layer]['rival_is_entity']]}
        report['layers'][layer] = {}
        for name, rows in groups.items():
            y = np.array([m['hallucinated'] for _, m in rows], dtype=int)
            fold = np.array([r['fold'] for r,_ in rows])
            f = {k: np.array([r['contrast'][layer][k] for r,_ in rows]) for k in (
                'visual_target', 'visual_margin', 'ffn_target', 'ffn_margin', 'visual_fixed_ffn_margin', 'post_margin')}
            f['visual_mass'] = np.array([r['features'][layer]['visual_mass'] for r,_ in rows])
            f['drift'] = np.array([r['features'][layer]['drift'] for r,_ in rows])
            s = {'n': len(y), 'false': int(y.sum()), 'within': {}, 'means': {}, 'crossfit': {}}
            if len(set(y)) < 2:
                report['layers'][layer][name] = s
                continue
            for key, values in f.items():
                sign = -1 if key.startswith('visual') else 1
                s['within'][key] = paired_auc(rows, sign*values)
                s['means'][key] = [float(values[y == label].mean()) for label in (0, 1)]
            for cols in (('visual_target',), ('visual_margin', 'ffn_margin'), ('visual_mass',),
                         ('visual_mass','visual_margin','ffn_margin'), ('visual_mass','drift','visual_margin','ffn_margin')):
                pred = np.zeros(len(y))
                valid = True
                coefficients = []
                for j in (0, 1):
                    test = fold == j
                    if not test.any() or len(set(y[~test])) < 2:
                        valid = False
                        break
                    data = np.column_stack([f[k] for k in cols])
                    scaler = StandardScaler().fit(data[~test])
                    data = scaler.transform(data)
                    model = LogisticRegression(C=.3).fit(data[~test], y[~test])
                    pred[test] = model.decision_function(data[test])
                    coefficients.append(model.coef_[0].tolist())
                if valid:
                    s['crossfit']['+'.join(cols)] = {**paired_auc(rows, pred), 'coefficients': coefficients}
            report['layers'][layer][name] = s
    (root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('AUDITED', len(new), 'all matched, first mentions', len(first))
    for layer, values in report['layers'].items():
        for name, stat in values.items():
            print(layer, name, stat['n'], stat['false'])
            for section in ('within', 'crossfit'):
                print(section, {k:[round(v['pooled'],3), (round(v['within_category']['auc'],3) if v['within_category']['pairs'] else None)]
                                for k,v in stat[section].items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    analyze(p.parse_args().root)
