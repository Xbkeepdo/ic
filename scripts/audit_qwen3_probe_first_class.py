"""Recompute every canonical-history decision from saved generated prefixes."""

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('NLTK_DATA', str(REPO / 'outputs/generation_time_evidence_drift_llava/nltk_data'))
from features.coco_object_history import ObjectClassHistory
from scripts.audit_qwen3_probe_recycle_events import audit as physical_audit
from scripts.run_qwen3_signed_feedback import read, sha, write


def audit(root):
    from transformers import AutoTokenizer
    physical_audit(root)
    plan = read(root / 'protocol.json')
    source = Path(plan['source'])
    assert (root / 'object_lexicon.json').read_bytes() == (source / 'object_lexicon.json').read_bytes()
    # The copied dictionary is checked against the inherited frozen source,
    # even though the new output dictionary has no separate manifest entry.
    assert sha(source / 'object_lexicon.json') == plan['source_sha256'][str(source / 'object_lexicon.json')]
    tokenizer = AutoTokenizer.from_pretrained('/root/rivermind-data/models/qwen3_vl_8b', local_files_only=True)
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(tokenizer, read(root / 'object_lexicon.json')['token_ids'])
    name, = plan['candidates']
    report = dict(passed=True, images=0, decisions=0, blocked=0, proposals=0, accepted=0,
                  own_lexicon_equals_frozen_source=True)
    for shard in (0, 1):
        for row in map(json.loads, (root / plan['cohort'] / f'{name}.shard{shard}.jsonl').read_bytes().splitlines()):
            report['images'] += 1
            tokens, stats = row['response_token_ids'], row['sar_ems_statistics']
            assert stats['first_object_class_gate'] and stats['history_uses_generated_prefix_only']
            decisions = {d['step']: d for d in stats['class_decisions']}
            assert len(decisions) == len(stats['class_decisions'])
            assert sum(d['blocked'] for d in decisions.values()) == stats['blocked_by_class_history']
            for step, decision in decisions.items():
                target = decision['token']
                assert target in classes and target not in tokens[:step]
                prefix = tokenizer.decode(tokens[:step], skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
                prior = parser.classes(prefix)
                assert decision['prior_classes'] == sorted(prior)
                assert decision['canonical'] == classes[target]
                assert decision['blocked'] == (classes[target] in prior)
                if decision['blocked']:
                    assert tokens[step] == target
                    report['blocked'] += 1
                report['decisions'] += 1
            for event in stats['recycle_events']:
                assert not decisions[event['step']]['blocked']
                assert decisions[event['step']]['token'] == event['token']
                report['proposals'] += 1
                report['accepted'] += int(event['accepted'])
    assert report['images'] == plan['images']
    write(root / 'first_class_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
