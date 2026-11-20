"""Describe the fixed original gate's omissions using existing native SADI caches."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(repo, output):
    assert not output.exists(), 'Keep existing diagnostic evidence'
    trace = repo/'outputs/qwen3_full_risk_f_trace_20261118'
    frozen = repo/'outputs/qwen3_evidence_head_routing_pilot_20261119'
    predictions = repo/'outputs/qwen3_risk_semantic_gate_transfer_20261119/predictions.json'
    paths = [trace/'labelled_positions.json', trace/'unlabelled_positions.json',
             trace/'final_provenance.json', predictions, frozen/'protocol.json',
             frozen/'final_provenance.json', frozen/'cohort.json', Path(__file__)]
    inputs = {str(path):sha(path) for path in paths}
    assert read(trace/'final_provenance.json')['passed']
    assert read(frozen/'final_provenance.json')['passed']
    old = read(frozen/'protocol.json'); threshold = old['real_threshold']
    assert threshold == 0.418672114610672
    labelled = read(trace/'labelled_positions.json')
    unlabelled = read(trace/'unlabelled_positions.json')
    predicted = read(predictions)
    key = lambda row:(row['image_id'],row['step'],row['token_id'])
    lookup = {key(row):row for row in predicted}
    assert len(lookup)==len(predicted)==len(labelled)+len(unlabelled)==209
    assert len(labelled)==191 and sum(row['automatic_hallucination'] for row in labelled)==29
    output.mkdir()
    protocol = dict(fixed_at=time.time(), source_sha256=inputs,
        operation='Cached original108 gate omissions; no model, training, feature extraction or threshold search.',
        original_threshold=threshold, planned_eligibility='Keep lexical309 and first-token/class rules, omit learned probe.',
        motivation='Test eligibility separately from the already-tested head allocation and fixed operator.',
        previous_negative='Semantic-final31 omitted I/El gate but changed spatial attention unconditionally; no claim that omission alone is novel.',
        followup='Only after inspecting missed physical-object candidates, freeze any prospective generation protocol. Reuse neither the old9 no-positive certificates nor outputs as new-policy captions.',
        validation_untouched=True)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for row in labelled+unlabelled:
        cached=lookup[key(row)]
        assert cached['canonical']==row['canonical']
        rows.append(dict(image_id=row['image_id'],step=row['step'],token_id=row['token_id'],
            canonical=row['canonical'],context=cached['context'],automatic_hallucination=cached['automatic_hallucination'],
            original108_p_real=cached['original108_p_real'],
            original108_positive=cached['original108_positive'],
            no_learned_gate_eligible=True))
        assert rows[-1]['original108_positive']==(rows[-1]['original108_p_real']<threshold)
    missed=[row for row in rows if row['automatic_hallucination'] and not row['original108_positive']]
    assert len(missed)==16
    result=dict(passed=True,protocol_sha256=sha(output/'protocol.json'),cached_candidates=209,labelled=191,
        automatic_false=29,automatic_false_not_triggered=len(missed),missed_candidates=missed,
        original_positive=sum(row['original108_positive'] for row in rows),
        labelled_real_not_triggered=sum(row['automatic_hallucination'] is False and not row['original108_positive'] for row in rows),
        semantic_scope='Automatic errors can denote colours, parts, toys, places, photographed objects and visible but unannotated objects.',
        image_review='pending',advance_to_generation=None,neural_generation=False,cuda_used=False,
        raw_captions_and_scores_unchanged=True)
    for name,value in inputs.items():assert sha(Path(name))==value,name
    (output/'predictions.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='missed_candidates'}))
    print('MISSED_IDS',sorted({row['image_id'] for row in missed}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repo',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();analyze(args.repo,args.output)
