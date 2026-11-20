"""Audit final-layer write contrasts before choosing a generation experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_qwen3_visual_removal import paired_auc


def records(root):
    return [json.loads(line) for path in sorted(root.glob('trace.shard*.jsonl'))
            for line in path.read_bytes().splitlines()]


def analyze(root):
    plan = json.loads((root / 'protocol.json').read_bytes())
    assert json.loads((root / 'status.json').read_bytes())['status'] == 'complete'
    for name, digest in plan['contrast_sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    original_root = Path(next(p for p in plan['contrast_sources']
                              if p.endswith('trace.shard0.jsonl'))).parent
    original = {(r['image_id'], r['step']): r for r in records(original_root)}
    current = records(root)
    assert len(current) == len(original)
    assert {(r['image_id'], r['step']) for r in current} == set(original)
    for event in current:
        old = original[event['image_id'], event['step']]
        assert event['counterfactual_replay_matches']
        assert event['target_token_id'] == old['target_token_id']
        assert event['mentions'] == old['mentions']
        assert set(event['contrast']) == {'35'}

    first = [(r, r['mentions'][0]) for r in current
             if len(r['mentions']) == 1 and not r['mentions'][0]['repeat']]
    groups = {'first': first,
              'first_entity': [(r, m) for r, m in first if r['token_is_noun']]}
    report = dict(positions=len(current), replay_matches=len(current), groups={})
    for name, rows in groups.items():
        y = np.array([m['hallucinated'] for _, m in rows], dtype=bool)
        visual = np.array([r['contrast']['35']['visual_target'] for r, _ in rows])
        gain = np.array([r['contrast']['35']['ffn_target'] for r, _ in rows])
        features = {'low_visual': -visual, 'ffn_gain': gain,
                    'ffn_minus_visual': gain - visual,
                    'low_visual_margin': -np.array([r['contrast']['35']['visual_margin'] for r, _ in rows]),
                    'ffn_margin': np.array([r['contrast']['35']['ffn_margin'] for r, _ in rows])}
        gates = {'visual_below_0_35': visual < 0.35,
                 'visual_negative': visual < 0,
                 'positive_ffn_exceeds_visual': (gain > 0) & (gain > visual)}
        stat = dict(n=len(rows), hallucinated=int(y.sum()),
                    auc={k: paired_auc(rows, v) for k, v in features.items()},
                    gates={k: dict(real=int((v & ~y).sum()), hallucinated=int((v & y).sum()))
                           for k, v in gates.items()},
                    quantiles={k: {label: np.quantile(v[y == positive], [.1, .25, .5, .75, .9]).tolist()
                                   for label, positive in (('real', False), ('hallucinated', True))}
                               for k, v in (('visual', visual), ('ffn_gain', gain))})
        report['groups'][name] = stat
    (root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
