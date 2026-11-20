"""Keep full SADI and test source gate adaptation with a fixed attention action.

Only gate weights, standardization or source calibration change. The existing
uniform attention action and its acceptance/rollback code remain unchanged.
Each gate has a separate output root; source-calibrated eligibility is checked
before any GPU launch. No independent500 is launched by this script.
"""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_risk_semantic_gate_pilot as base
from scripts import run_qwen3_signed_feedback as shared

SOURCE=shared.OUT/'qwen3_mass_matched_visual_pilot_20261119'
PLANNED=shared.OUT/'qwen3_native_sadi_gate_pilot_plan_20261119'
FIT=shared.OUT/'qwen3_native_sadi_source_probe_20261119'
NATIVE_DIAG=shared.OUT/'qwen3_native_sadi_source_gate_transfer_20261119'
MOMENT_DIAG=shared.OUT/'qwen3_native_source_moment_gates_20261119'
RESERVED=shared.OUT/'qwen3_mass_matched_visual_reserved500_20261119'
GATES=('retrained','calibration','moments')
AUDITS=['scripts/audit_qwen3_native_sadi_gate_pilot.py',
        'scripts/audit_qwen3_caption_with_native_cache_v2.py',
        'scripts/analyze_qwen3_native_sadi_gate_pilot.py',
        'scripts/audit_qwen3_caption_denominator.py']
FILES=['scripts/run_qwen3_native_sadi_gate_pilot.py',*AUDITS,
       'scripts/run_qwen3_risk_semantic_gate_pilot.py',
       'features/qwen3_risk_i_el_probe.py','features/qwen3_online_i_el.py',
       'features/qwen3_native_risk_i_el.py','features/qwen3_causal_visual_tilt.py',
       'features/qwen3_bounded_visual_tilt.py',
       'scripts/audit_qwen3_risk_semantic_gate_pilot.py']
ACTIVE='retrained'


def output():
    return shared.OUT/f'qwen3_native_gate_{ACTIVE}_pilot_20261119'


def declare():
    assert not PLANNED.exists()
    paths=[REPO/name for name in FILES]
    paths.extend(path/'protocol.json' for path in [SOURCE,FIT,NATIVE_DIAG,MOMENT_DIAG])
    paths.extend(RESERVED/name for name in ['cohort.json','reservation.json','batch_plan.json'])
    hashes={str(path):shared.sha(path) for path in paths}
    PLANNED.mkdir()
    shared.write(PLANNED/'recipe.json',dict(fixed_at=time.time(),source_sha256=hashes,
        gates=list(GATES),primary='retrained seed43; no best-seed selection',
        eligibility='Each selected gate must pass its already fixed initial coverage/preservation screen. No automatic fallback or500.',
        data='Same old development100, same22 ordered batch groups, same images/prompt/max512/B16.',
        base_layers=[5,18],anchor=19,attention_layers=list(range(20,36)),
        attention_action='For selected rows only, current visual logits z become z+0.5*abs(z) on each20--35 layer. Nonvisual logits unchanged.',
        gate='Actual native-final argmax, unseen eligible object class, p_real below its source-calibrated threshold; original108 risk/I/E_l features.',
        acceptance='Original-target final log probability decreases; otherwise restore native hidden/current KV. History KV unchanged.',
        component_scope='The attention coefficient is fixed and does not use per-layer Q/P. SAR/EMS-derived signals guide WHEN to act through the detector. This is not refreshed-next spatial guidance.',
        controls=['sadi','svar','causal_fixed'],controls_reused_byte_for_byte=True,
        common_native_probability_audit_limit=1e-5,
        generation_screen='Full sentence Cs/Ci and false counts lower vs SADI, F1 loss<=1.5pp and recall loss<=2pp; assess original108 fixed and SVAR too; inspect genuine physical fixes and matched controls before independent500.',
        analysis='Both evaluators, full100 paired bootstrap, all fixes/introductions and top3 nonzero false/recall changes; no independent claim from old100.'))
    for name in FILES:
        target=PLANNED/'source'/name;target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(REPO/name,target)
    print(dict(declared=True,path=str(PLANNED),no_model_calls=True),flush=True)


def gate_config():
    if ACTIVE=='retrained':
        diagnostic=NATIVE_DIAG;score='native_seed43'
        assert shared.read(FIT/'verification.json')['passed']
        results=shared.read(diagnostic/'analysis.json')
        assert results['primary_seed']==43 and results['initial_pilot_retention_screen']
        return diagnostic,score,FIT/'seed43',results['thresholds'][score]
    diagnostic=MOMENT_DIAG
    results=shared.read(diagnostic/'analysis.json')
    score={'calibration':'calibration_only','moments':'native_moments'}[ACTIVE]
    assert results['passed'] and results['initial_pilot_retention_screens'][score]
    original=Path(shared.read(diagnostic/'protocol.json')['original_probe'])
    probe=original if ACTIVE=='calibration' else diagnostic/'native_moments'
    return diagnostic,score,probe,results['thresholds'][score]


def prepare():
    root=output();assert not root.exists(),'Inspect existing roots/processes, not an implicit resume.'
    planned=shared.read(PLANNED/'recipe.json')
    for filename,expected in planned['source_sha256'].items():assert shared.sha(Path(filename))==expected,filename
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert not (RESERVED/'consumed.json').exists()
    diagnostic,score,probe,threshold=gate_config()
    name=f'native_{ACTIVE}_fixed'
    old=shared.read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    hashes.update(planned['source_sha256'])
    paths=[PLANNED/'recipe.json',diagnostic/'protocol.json',diagnostic/'analysis.json',diagnostic/'predictions.json',
           SOURCE/'protocol.json',SOURCE/'analysis_protocol.json',SOURCE/'final_provenance.json',
           probe/'model.pt',probe/'input_scaler.npz']
    paths.extend(REPO/path for path in FILES)
    paths.extend(RESERVED/path for path in ['cohort.json','reservation.json','batch_plan.json'])
    for path in paths:hashes[str(path)]=shared.sha(path)
    for filename,expected in hashes.items():assert shared.sha(REPO/filename)==expected,filename
    root.mkdir();(root/'logs').mkdir();(root/'coco100').mkdir()
    for item in ['cohort.json','object_lexicon.json']:shutil.copyfile(SOURCE/item,root/item)
    for shard in [0,1]:
        for control in ['sadi','svar','causal_fixed']:
            path=SOURCE/f'coco100/{control}.shard{shard}.jsonl'
            hashes[str(path)]=shared.sha(path);shutil.copyfile(path,root/'coco100'/path.name)
    recipe=dict(planned,gate_variant=ACTIVE,gate_diagnostic_score_key=score,
        checkpoint=str(probe),source_real_threshold=threshold,
        unchanged_anchor=19,budget_source_layers=list(range(19,35)),
        intervention_layers=list(range(20,36)),gamma=1.,coefficient='Uniform0.5',
        predeclared_recipe_sha256=shared.sha(PLANNED/'recipe.json'))
    plan=dict(stage='native-source-gate-development',images=100,shards=2,cohort='coco100',max_batch=16,max_new_tokens=512,
        variants=['sadi','svar','causal_fixed',name],candidates=[name],copied_controls=['sadi','svar','causal_fixed'],
        cached_controls=['sadi'],base_by_candidate={name:'sadi'},mode_by_candidate={name:'fixed'},
        refresh_by_candidate={name:False},source=str(SOURCE),source_protocol_sha256=shared.sha(SOURCE/'protocol.json'),
        source_sha256=hashes,fixed_at=time.time(),probe=str(probe),real_threshold=threshold,
        eos_token_ids=old['eos_token_ids'],recipe=recipe,gate_variant=ACTIVE,
        gate_diagnostic=str(diagnostic),gate_diagnostic_score_key=score,
        common_native_probability_limit=planned['common_native_probability_audit_limit'],
        initial_pilot_retention_screen_passed=True)
    shared.write(root/'protocol.json',plan)
    analysis_files=[*AUDITS,'scripts/analyze_qwen3_causal_visual_tilt_pilot.py',
        'scripts/analyze_qwen3_scalar_visual_pilot.py','scripts/audit_qwen3_visual_feedback_mentions.py',
        'scripts/audit_qwen3_fresh_caption500.py']
    shared.write(root/'analysis_protocol.json',dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(root/'protocol.json'),before_inspecting_pilot_metrics=True,
        source_sha256={str(REPO/n):shared.sha(REPO/n) for n in analysis_files},
        primary=planned['analysis'],cases='All fixes/introductions and top3 nonzero false/recall changes, original photos.'))
    return plan


def decorate(stats,eos):
    base.causal.decorate_statistics(stats,False,eos)
    for item in stats:
        item.update(gate_feature_components=['risk','I','E_l'],gate_feature_indices=list(range(108)),
            feature_reader_output_width=108,probe_input_width=108,source_gate_variant=ACTIVE,
            acg_scope='Source-adapted108 gate; unchanged uniform20--35 attention action. No refreshed per-layer evidence coefficient.')


def execute(worker):
    from features import qwen3_risk_semantic_probe as legacy_probe
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ,{'IC_NATIVE_SOURCE_GATE':ACTIVE}))
        for key,value in dict(OUT=output(),NAME=f'native_{ACTIVE}_fixed',prepare=prepare,
                              decorate=decorate,AUDITS=AUDITS,__file__=__file__).items():
            stack.enter_context(patch.object(base,key,value))
        stack.enter_context(patch.object(legacy_probe,'FullRiskSemanticProbe',FullRiskIElProbe))
        base.run() if worker is None else base.worker(worker)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare',action='store_true')
    parser.add_argument('--gate',choices=GATES,default=os.environ.get('IC_NATIVE_SOURCE_GATE','retrained'))
    parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    ACTIVE=args.gate
    declare() if args.declare else execute(args.worker)
