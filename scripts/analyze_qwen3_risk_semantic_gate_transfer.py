"""Test the frozen risk/E_l gate on existing SADI features, with no neural replay."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_risk_semantic_probe import FullRiskSemanticProbe
from features.qwen3_risk_i_el_probe import FullRiskIElProbe

TRAIN = REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'
TRACE = REPO/'outputs/qwen3_full_risk_f_trace_20261118'
I_EL = REPO/'outputs/qwen3_sadi_full_i_el_trace'
ROOT = REPO/'outputs/qwen3_risk_semantic_gate_transfer_20261119'
BASE = REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def run():
    assert not ROOT.exists(), 'Preserve existing diagnostic outputs'
    assert read(TRACE/'final_provenance.json')['passed']
    assert read(BASE/'final_provenance.json')['passed']
    assert read(I_EL/'independent_audit.json')['passed']
    old = read(BASE/'protocol.json'); metrics = read(TRAIN/'risk+E_l/seed43/metrics.json')
    paths = [Path(__file__), REPO/'features/qwen3_risk_semantic_probe.py',
        REPO/'features/qwen3_online_i_el.py', REPO/'features/qwen3_risk_i_el_probe.py',
        TRACE/'protocol.json', TRACE/'labelled_positions.json', TRACE/'unlabelled_positions.json',
        TRACE/'unlabelled_context_scores.json', TRACE/'final_provenance.json',
        I_EL/'protocol.json', I_EL/'independent_audit.json',
        REPO/'outputs/qwen3_static_semantic_gate_transfer_20261119/protocol.json',
        REPO/'outputs/qwen3_static_semantic_gate_transfer_20261119/analysis.json',
        REPO/'outputs/qwen3_static_semantic_gate_transfer_20261119/human_review.json',
        BASE/'protocol.json', BASE/'cohort.json', BASE/'final_provenance.json',
        TRAIN/'compact_features.npz', TRAIN/'protocol.json']
    for directory in (TRAIN/'risk+E_l/seed43', Path(old['probe'])):
        paths.extend(directory/n for n in ('model.pt', 'input_scaler.npz', 'metrics.json'))
    paths.extend(I_EL/f'features.shard{s}.jsonl' for s in (0, 1))
    inputs = {str(p):sha(p) for p in paths}
    assert metrics['threshold_selection'] == 'train_f1'
    threshold = metrics['decision_threshold']
    assert threshold == metrics['thresholds']['train_f1']
    ROOT.mkdir()
    protocol = dict(fixed_at=time.time(), source_sha256=inputs, no_generation=True, no_refit=True,
        candidate='Frozen original risk+E_l/seed43,36 risk and36 semantic inputs; I omitted.',
        threshold_p_real=threshold, threshold_origin='Original3200-image training F1, not current SADI labels.',
        original108_threshold=old['real_threshold'],
        eligibility='Same209 archived lexical/first-class SADI candidate slots;191 aligned labels and18 unlabelled contexts.',
        primary='Compare risk/E_l and original108 on identical cached candidates; old threshold fixed.',
        followup='Consider a generation pilot only if an added trigger is an original-image-confirmed unsupported physical object and labelled-real trigger count does not exceed original108. Audit unlabelled contexts separately. Otherwise retain diagnosis, no threshold search.',
        physical_truth='CHAIR mention labels and annotation support are not human physical-truth labels.',
        untouched_validation='No reserved500 generation, consumption, score reading or redraw.')
    write(ROOT/'protocol.json', protocol)
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    semantic = FullRiskSemanticProbe(TRAIN/'risk+E_l/seed43').eval()
    original = FullRiskIElProbe(old['probe']).eval()
    with np.load(TRAIN/'compact_features.npz', allow_pickle=False) as d:
        training = {k:d[k].copy() for k in d.files}
    test = training['test_mask']
    with torch.inference_mode():
        old_real = semantic(torch.from_numpy(np.concatenate((training['risk'],training['E_l']),1))).numpy()
    reference_auc = roc_auc_score(training['label'][test], old_real[test])
    assert abs(reference_auc-metrics['auc']) < 1e-6
    ids = read(BASE/'cohort.json')['coco100']
    assert not set(ids) & set(map(int, training['image_id']))
    assert read(TRACE/'protocol.json')['ids'] == ids
    assert read(I_EL/'cohort.json')['trace'] == ids
    labelled = read(TRACE/'labelled_positions.json'); unlabelled = read(TRACE/'unlabelled_positions.json')
    rows = labelled+unlabelled
    assert len(rows) == 209 and len({(r['image_id'],r['step']) for r in rows}) == len(rows)
    semantic_input = np.asarray([r['native_features'][:36]+r['E_l'] for r in rows], dtype=np.float32)
    original_input = np.asarray([r['native_features'][:36]+r['native_I']+r['E_l'] for r in rows], dtype=np.float32)
    assert semantic_input.shape == (209,72) and original_input.shape == (209,108)
    with torch.inference_mode():
        sem_probability = semantic(torch.from_numpy(semantic_input)).numpy()
        orig_probability = original(torch.from_numpy(original_input)).numpy()
        assert torch.equal(torch.from_numpy(sem_probability), semantic(torch.from_numpy(original_input)))
    context = {(r['image'],r['step']):r['context'] for r in read(TRACE/'unlabelled_context_scores.json')}
    predictions = []
    for index,row in enumerate(rows):
        predictions.append(dict(image_id=row['image_id'],step=row['step'],token_id=row['token_id'],
            canonical=row['canonical'],context=row.get('context',context.get((row['image_id'],row['step']))),
            automatic_hallucination=row.get('automatic_hallucination'),
            risk_semantic_p_real=float(sem_probability[index]),original108_p_real=float(orig_probability[index]),
            risk_semantic_positive=bool(sem_probability[index]<threshold),
            original108_positive=bool(orig_probability[index]<old['real_threshold']),
            semantic_features=row['native_features'][:36]+row['E_l']))
    def summarize(records):
        labels = np.asarray([r['automatic_hallucination'] for r in records], dtype=bool)
        result = dict(positions=len(records),automatic_hallucinations=int(labels.sum()),methods={})
        for key,score in (('risk_semantic','risk_semantic_p_real'),('original108','original108_p_real')):
            p=np.asarray([r[score] for r in records]); chosen=np.asarray([r[key+'_positive'] for r in records])
            result['methods'][key]=dict(auroc=float(roc_auc_score(labels,1-p)),
                hallucination_aupr=float(average_precision_score(labels,1-p)),
                real_aupr=float(average_precision_score(~labels,p)),
                automatic_false_triggers=int((chosen & labels).sum()),labelled_real_triggers=int((chosen & ~labels).sum()),
                triggered_positions=int(chosen.sum()))
        return result
    summary = summarize(predictions[:len(labelled)])
    eligible = [r for r in predictions if r['risk_semantic_positive'] and not r['original108_positive']]
    removed = [r for r in predictions if r['original108_positive'] and not r['risk_semantic_positive']]
    iel_rows = [json.loads(line) for s in (0,1) for line in (I_EL/f'features.shard{s}.jsonl').read_bytes().splitlines()]
    known = {(r['image_id'],r['step'],r['target_token_id']):r for r in iel_rows}
    diffs = [float(np.max(np.abs(semantic_input[i,36:]-np.asarray(known[(r['image_id'],r['step'],r['token_id'])]['features'][36:],dtype=np.float32))))
        for i,r in enumerate(rows) if (r['image_id'],r['step'],r['token_id']) in known]
    for n,v in inputs.items():
        assert sha(Path(n)) == v,n
    write(ROOT/'predictions.json', predictions)
    write(ROOT/'analysis.json', dict(passed=True, protocol_sha256=sha(ROOT/'protocol.json'),
        original_risk_semantic_probe_test_auc=float(reference_auc),reference_auc=metrics['auc'],
        labelled=summary, unlabelled_counts={name:sum(r[name+'_positive'] for r in predictions[len(labelled):]) for name in ('risk_semantic','original108')},
        added_triggers=eligible,removed_triggers=removed,
        original_cache_overlap_positions=len(diffs),maximum_E_l_difference_between_archived_readers=max(diffs,default=None),
        labelled_real_nonincrease=summary['methods']['risk_semantic']['labelled_real_triggers']<=summary['methods']['original108']['labelled_real_triggers'],
        new_image_confirmed_physical_correction=None,image_review='pending',
        neural_generation_performed=False,refitting_performed=False,cuda_initialized=torch.cuda.is_initialized()))
    print(json.dumps(dict(passed=True,labelled=summary,
        unlabelled_counts=read(ROOT/'analysis.json')['unlabelled_counts'],
        added_triggers=[{k:v for k,v in r.items() if k!='semantic_features'} for r in eligible],
        removed_triggers=[{k:v for k,v in r.items() if k!='semantic_features'} for r in removed]),indent=2))


if __name__ == '__main__':
    run()
