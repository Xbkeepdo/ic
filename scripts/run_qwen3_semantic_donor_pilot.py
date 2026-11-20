"""Two attention transport policies after complete SADI, on development100."""

import argparse
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_probe_first_class_pilot as base
from scripts import run_qwen3_svar_first_class_pilot as svar
from scripts import run_qwen3_history_route_pilot as driver
from scripts.run_qwen3_semantic_final_pilot import build_wrapper

OUT = shared.OUT/'qwen3_semantic_donor_pilot_20261118'
TRACE = shared.OUT/'qwen3_full_risk_f_trace_20261118'
PREFLIGHT = shared.OUT/'qwen3_semantic_donor_preflight_v2/verification.json'
CPU = shared.OUT/'qwen3_semantic_donor_cpu/verification.json'
MODES = {'sadi_sam_v2_history_19': 'history', 'sadi_sam_v2_semantic_donor_19': 'semantic'}
FILES = ['features/qwen3_native_risk_i_el.py', 'features/qwen3_semantic_donor_feedback.py',
         'features/qwen3_semantic_donor_transport.py', 'scripts/verify_qwen3_semantic_donor_gpu.py',
         'scripts/verify_qwen3_semantic_donor_transport.py', 'scripts/run_qwen3_semantic_donor_pilot.py',
         'scripts/audit_qwen3_semantic_donor_pilot.py', 'scripts/run_qwen3_semantic_final_pilot.py']


def prepare():
    assert not OUT.exists(), 'Inspect existing run before any restart'
    assert shared.read(TRACE/'final_provenance.json')['passed']
    hashes = dict(shared.read(base.OUT/'protocol.json')['source_sha256'])
    for path in [TRACE/'protocol.json', TRACE/'cached_sam_v2/protocol.json', TRACE/'source_calibration/protocol.json']:
        hashes.update(shared.read(path)['source_sha256'])
        hashes[str(path)] = shared.sha(path)
    assert shared.read(PREFLIGHT)['passed'] and shared.read(CPU)['passed']
    for name, expected in shared.read(PREFLIGHT)['source_sha256'].items():
        assert shared.sha(REPO/name) == expected, name
    probe = REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b/risk+I+E_l/seed43'
    calibration = TRACE/'source_calibration/analysis.json'
    threshold = shared.read(calibration)['calibration']['SAM_V2']['threshold']
    for path in [CPU, PREFLIGHT, calibration, TRACE/'human_review.json',
                 base.OUT/'object_lexicon.json', base.OUT/'protocol.json', *(REPO/n for n in FILES)]:
        hashes[str(path)] = shared.sha(path)
    for name, expected in hashes.items():
        assert shared.sha(Path(name)) == expected, name
    OUT.mkdir(); (OUT/'coco100').mkdir(); (OUT/'logs').mkdir()
    shutil.copyfile(base.OUT/'cohort.json', OUT/'cohort.json')
    shutil.copyfile(base.OUT/'object_lexicon.json', OUT/'object_lexicon.json')
    controls = ['sadi', base.NAME]
    for shard in (0, 1):
        for name, origin in [(n, base.OUT) for n in controls]+[('svar', svar.OUT)]:
            path = origin/f'coco100/{name}.shard{shard}.jsonl'
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT/'coco100'/path.name)
    plan = dict(stage='semantic-donor-pilot', images=100, cohort='coco100', max_batch=16, max_new_tokens=512,
        model='Qwen3-VL-8B-Instruct', fixed_at=time.time(), variants=[*controls, 'svar', *MODES],
        candidates=list(MODES), cached_controls=['sadi'], copied_controls=controls,
        additional_copied_controls={'svar': str(svar.OUT)}, source_sha256=hashes,
        source=str(base.OUT), probe=str(probe), real_threshold=threshold,
        recipe=dict(base_layers=[5, 18], source_layer=19, intervention_layer=19, target_read_layer=35,
            indexing='zero based', gate='Original native108 SAM-V2 p_real below old-domain real5th percentile; unseen token and first generated object class.',
            calibration='Old disjoint783 images /1848 real positions only; no target-label threshold fit.',
            risk='Original K32/tau.03 exact EMD, cosine cost and both endpoint penalties coefficient1.',
            input='risk0..35,I0..35,E_l0..35; current trajectory recomputed at every eligible position.',
            support='S_j=(max_alias p_j(alias)/max309 p_j(object))*(V*max_alias p_j(alias)/(1+V*max_alias p_j(alias))).',
            unsupported='U=sum_j Normalize(actual visual-write norms)_j*(1-S_j); zero-write fallback U=1.',
            drift='HellingerSquared(negative final-gradient visual effects, FFN hpre visual preference tau.03).',
            history_control='fraction=.5*drift; move positive generated-history effects to negative visual effects, same original FFN spatial deficit.',
            semantic='fraction=.5*drift*U; same history donors plus positive visual effects weighted1-S_j.',
            acceptance='Original final token log probability strictly decreases after replay, else exact hidden/current-tail-KV rollback.',
            replay='Layer19 cached QKV; full-batch layers20--35 current query only. Prefix KV untouched.',
            cache='Selected309 fully normalized semantic probabilities shared across policies within fixed ordered group after exact hpre/value equality; lazy image/layer cache. No dynamic feature reuse.',
            cost='One image encoder per group and policy; one gradient per selected step, exact36 small OT per eligible target; no fullV/attention/hidden saved.',
            limitation='Gate-only screen previously failed17 false positives; this is a new joint gate+attention policy experiment, not a passed detector screen.'),
        development_screen='Against SADI both Cs/Ci and hallucinated count decrease; F1 loss<=1.5pp, recall loss<=2pp, leave-largest Ci no worse; image-confirmed identity correction required.',
        followup='If supported by actual outputs and images: support-history budget ablation and per-head dose-matched spatial permutation, then independent500. Development100 is not independent evidence.')
    shared.write(OUT/'protocol.json', plan)
    return plan


def worker(gpu):
    import json
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_semantic_donor_feedback import Qwen3BatchSemanticDonor
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    plan = shared.read(OUT/'protocol.json')
    for name, expected in plan['source_sha256'].items():
        assert shared.sha(Path(name)) == expected, name
    wrapper = build_wrapper()
    allowed = frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    probe = FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    groups = shared.source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    paths = {name: OUT/f'coco100/{name}.shard{gpu}.jsonl' for name in MODES}
    assert not any(p.exists() for p in paths.values())
    with torch.inference_mode():
        for index, group in enumerate(groups):
            ids = [r['image_id'] for r in group]
            (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [i for i, _ in inputs] == ids and span[1]-span[0] == group[0]['visual_token_count']
            if index == 0:
                tokens, _, _ = original.generate_group(wrapper, span, inputs, allowed, 'sadi', 32)
                assert tokens == [r['response_token_ids'][:32] for r in group]
                shared.write(OUT/f'verification.gpu{gpu}.json', dict(passed=True, image_ids=ids,
                    actual_batch=len(ids), exact_native_prefix_tokens=32))
            static = {}
            for name, policy in MODES.items():
                def factory(model, current_span, nouns, mode, strength):
                    return Qwen3BatchSemanticDonor(model, current_span, allowed, probe=probe,
                        semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer, history_parser=parser,
                        transport_mode=policy, shared_static=static, real_threshold=plan['real_threshold'])
                with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                    tokens, stats, seconds = original.generate_group(wrapper, span, inputs, allowed, shared.POSITIVE, 512)
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                with paths[name].open('a', encoding='utf-8') as stream:
                    for image_id, current, text, info in zip(ids, tokens, texts, stats, strict=True):
                        assert 0 < len(current) <= 512 and info['maximum_probe_hidden_difference'] == 0
                        stream.write(json.dumps(dict(image_id=image_id, model=MODEL, variant=name,
                            generated_text=text.strip(), response_token_ids=current, seconds=seconds/len(ids),
                            batch_size=len(ids), batch_wall_seconds=seconds, visual_token_count=span[1]-span[0],
                            sar_ems_statistics=info, decoder='native_qwen3_generate_same_span_batch'), ensure_ascii=False)+'\n')
                print(f'[{gpu}] {name} group={index} batch={len(ids)} seconds={seconds:.3f}', flush=True)
            del static
    assert all(shared.group_keys(shared.source_groups(p)) == shared.group_keys(groups) for p in paths.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), patch.object(driver, 'prepare', prepare):
            driver.run()
