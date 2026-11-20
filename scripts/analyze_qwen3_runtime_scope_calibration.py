"""Match source-real calibration to the fixed runtime target vocabulary.

Only archived source scores, source masks and cached target scores are used.
No network is trained or run. Source quantiles are fixed before target
trigger counts are read; every model and training seed is reported.
"""

import argparse
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def locations():
    base=REPO/'outputs'
    return dict(native=base/'qwen3_native_sadi_source_probe_20261119',
        moment=base/'qwen3_native_source_moment_gates_20261119',
        balanced=base/'qwen3_token_balanced_source_probe_20261120',
        native_target=base/'qwen3_native_sadi_source_gate_transfer_20261119',
        balanced_target=base/'qwen3_token_balanced_source_gate_transfer_20261120',
        lexicon=base/'qwen3_mass_matched_visual_pilot_20261119/object_lexicon.json',
        scope=base/'qwen3_source_runtime_scope_audit_20261120/scope_audit.json')


def prepare(root):
    assert not root.exists()
    paths=locations()
    source=Path(read(paths['native']/'protocol.json')['original_source'])
    inputs=[Path(__file__),REPO/'scripts/verify_qwen3_teacher_sadi_capture.py',
            source/'compact_features.npz',paths['lexicon'],paths['scope']]
    for name in ['native','moment','balanced']:
        inputs.extend(paths[name]/file for file in ['protocol.json','source_predictions.npz'])
        inputs.append(paths[name]/('analysis.json' if name=='moment' else 'verification.json'))
    inputs.extend(paths[name]/file for name in ['native_target','balanced_target','moment']
                  for file in ['analysis.json','predictions.json'])
    root.mkdir()
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256={str(p):sha(p) for p in set(inputs)},
        source=str(source),locations={k:str(v) for k,v in paths.items()},quantile=.05,quantile_method='linear',
        calibration='Original source test_mask AND real label AND target_token_id in the actual fixed309 runtime vocabulary.',
        target_policy='Report all209 stored slots; only runtime-vocabulary members can trigger. Report out-of-scope slots explicitly.',
        candidates=['raw_reference_vocab','frozen_native_vocab','moment_native_vocab',
                    'native43_vocab','native44_vocab','native45_vocab',
                    'balanced43_vocab','balanced44_vocab','balanced45_vocab'],
        primary_models=['raw_reference_vocab','frozen_native_vocab','moment_native_vocab','native43_vocab','balanced43_vocab'],
        no_best_seed_selection=True,no_current_threshold_fit=True,
        pilot_screen='Verified fork353411/step97/token22435 is covered and automatic-real triggers fewer than the original reference on the same runtime-eligible slots.',
        scope='Vocabulary-conditioned calibration only. The forced source prefix and native-argmax mismatch are not corrected by this audit.',
        no_model_calls=True,no_new_generation=True,no_automatic500=True))
    print(dict(prepared=True,root=str(root)),flush=True)


def run(root):
    import numpy as np
    plan=read(root/'protocol.json')
    assert not (root/'source_calibration.json').exists()
    for path,expected in plan['source_sha256'].items():assert sha(Path(path))==expected,path
    paths={k:Path(v) for k,v in plan['locations'].items()}
    old=np.load(Path(plan['source'])/'compact_features.npz',allow_pickle=False)
    ids=old['target_token_id'];eligible=np.isin(ids,read(paths['lexicon'])['token_ids'])
    full=old['test_mask']&(old['label']==1);cal=full&eligible
    assert full.sum()==1848 and cal.sum()==1518
    archives={name:np.load(paths[name]/'source_predictions.npz',allow_pickle=False) for name in ['native','moment','balanced']}
    for name in ['native','balanced']:
        verified=read(paths[name]/'verification.json');assert verified['passed']
        assert sha(paths[name]/'source_predictions.npz')==verified['source_predictions_sha256']
        np.testing.assert_array_equal(archives[name]['row'],np.arange(len(ids)))
        np.testing.assert_array_equal(archives[name]['label'],old['label'])
    assert sha(paths['moment']/'source_predictions.npz')==read(paths['moment']/'analysis.json')['source_predictions_sha256']
    source_scores=dict(raw_reference_vocab=archives['moment']['original_on_raw'],
        frozen_native_vocab=archives['moment']['frozen_on_native'],moment_native_vocab=archives['moment']['native_moments'])
    for seed in [43,44,45]:
        source_scores[f'native{seed}_vocab']=archives['native'][f'seed{seed}']
        source_scores[f'balanced{seed}_vocab']=archives['balanced'][f'seed{seed}']
    assert set(source_scores)==set(plan['candidates'])
    thresholds={name:float(np.quantile(score[cal],.05,method='linear')) for name,score in source_scores.items()}
    all_thresholds={name:float(np.quantile(score[full],.05,method='linear')) for name,score in source_scores.items()}
    assert all(np.isfinite(score).all() for score in source_scores.values())
    # Seal the source-derived thresholds before loading any target predictions.
    write(root/'source_calibration.json',dict(thresholds=thresholds,prior_all_source_thresholds=all_thresholds,
        source_real_targets=int(full.sum()),runtime_eligible_source_real_targets=int(cal.sum()),
        excluded_source_real_targets=int((full&~eligible).sum()),no_current_labels_used=True))
    native=read(paths['native_target']/'predictions.json')
    balanced=read(paths['balanced_target']/'predictions.json')
    moment=read(paths['moment']/'predictions.json')
    def key(row):return row['image_id'],row['step'],row['token_id']
    maps={name:{key(r):r for r in values} for name,values in [('native',native),('balanced',balanced),('moment',moment)]}
    keys=[key(row) for row in native]
    assert len(keys)==len(set(keys))==209 and all(set(m)==set(keys) for m in maps.values())
    allowed=set(read(paths['lexicon'])['token_ids'])
    reference_tau=read(paths['native_target']/'analysis.json')['thresholds']['original108']
    predictions=[]
    for row in native:
        current=key(row);m=maps['moment'][current];b=maps['balanced'][current]
        scores=dict(raw_reference_vocab=m['scores']['original108'],frozen_native_vocab=m['scores']['original108'],
                    moment_native_vocab=m['scores']['native_moments'])
        for seed in [43,44,45]:
            scores[f'native{seed}_vocab']=row['scores'][f'native_seed{seed}']
            scores[f'balanced{seed}_vocab']=b['scores'][f'native_seed{seed}']
        within=row['token_id'] in allowed
        predictions.append(dict(image_id=row['image_id'],step=row['step'],token_id=row['token_id'],
            canonical=row['canonical'],context=row['context'],automatic_hallucination=row['automatic_hallucination'],
            runtime_eligible=within,scores=scores,
            original_reference_score=row['scores']['original108'],
            original_reference_trigger=bool(within and row['scores']['original108']<reference_tau),
            triggered={name:bool(within and score<thresholds[name]) for name,score in scores.items()}))
    def counts(field):
        selected=[r for r in predictions if field(r)]
        return dict(triggered=len(selected),automatic_false_triggered=sum(r['automatic_hallucination'] is True for r in selected),
            automatic_real_triggered=sum(r['automatic_hallucination'] is False for r in selected),
            unaligned_triggered=sum(r['automatic_hallucination'] is None for r in selected))
    reference=counts(lambda r:r['original_reference_trigger'])
    totals={name:counts(lambda r:r['triggered'][name]) for name in source_scores}
    fork=[r for r in predictions if key(r)==(353411,97,22435)]
    assert len(fork)==1 and fork[0]['runtime_eligible']
    screens={name:bool(fork[0]['triggered'][name] and totals[name]['automatic_real_triggered']<reference['automatic_real_triggered']) for name in source_scores}
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),thresholds=thresholds,
        reference_threshold=reference_tau,reference=reference,counts=totals,initial_pilot_retention_screens=screens,
        primary_models=plan['primary_models'],out_of_runtime_scope=[key(r) for r in predictions if not r['runtime_eligible']],
        verified_fork_reference=fork[0],no_model_calls=True,no_current_fit=True,no_new_generation=True,
        no_mitigation_claim=True,transitions={name:dict(
            added=[r for r in predictions if r['triggered'][name] and not r['original_reference_trigger']],
            removed=[r for r in predictions if not r['triggered'][name] and r['original_reference_trigger']]) for name in source_scores})
    for path,expected in plan['source_sha256'].items():assert sha(Path(path))==expected,path
    write(root/'predictions.json',predictions);write(root/'analysis.json',result)
    print({k:v for k,v in result.items() if k not in ['transitions','verified_fork_reference']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root)
