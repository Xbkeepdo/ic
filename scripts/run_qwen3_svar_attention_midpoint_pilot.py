"""Compare geometric and linear midpoint attention on exactly cached100 batches."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_probe_first_class_pilot as base
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_svar_first_class_pilot as original

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-svar-attention-midpoint-pilot')
SOURCE, TRACE = original.OUT, base.TRACE
RISK = original.NAME
MODES = ('half', 'geometric')
NAMES = ('svar_first_class_half_19', 'svar_first_class_geometric_19')


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', RISK), patch.object(repeat, 'NAME', NAMES[0]):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for shard in (0, 1):
        path = SOURCE / 'coco100' / f'svar.shard{shard}.jsonl'
        plan['source_sha256'][str(path)] = shared.sha(path)
        shutil.copyfile(path, OUT / 'coco100' / path.name)
    plan['copied_controls'].append('svar')
    plan['cached_controls'] = ['svar']
    previous = shared.read(SOURCE / 'protocol.json')
    for filename in ('features/qwen3_probe_svar_first_class.py', 'features/qwen3_svar_attention_midpoint.py',
                     'scripts/verify_qwen3_attention_midpoint_math.py', 'scripts/verify_qwen3_svar_attention_midpoint.py',
                     'scripts/run_qwen3_svar_attention_midpoint_pilot.py', 'scripts/audit_qwen3_svar_attention_midpoint.py'):
        plan['source_sha256'][str(REPO / filename)] = shared.sha(REPO / filename)
    for path in (OUT / 'object_lexicon.json', SOURCE / 'object_lexicon.json', Path(previous['semantic_preflight'])):
        plan['source_sha256'][str(path)] = shared.sha(path)
    for shard, groups in enumerate(shared.read(TRACE / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = TRACE / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-svar-attention-midpoint-pilot', candidates=list(NAMES),
        variants=[*plan['copied_controls'], *NAMES], geometry_modes=list(MODES),
        primary_baseline='svar', source_trace=str(TRACE), probe=previous['probe'], max_batch=16,
        actuation_preflight=str(OUT / 'midpoint_preflight.json'), semantic_preflight=previous['semantic_preflight'],
        fixed_at=time.time(),
        mathematical_check=dict(command='scripts/verify_qwen3_attention_midpoint_math.py',
            observed_exit_code=0, scipy_KL_optimum=.14494897628064868,
            maximum_mass_error=1.6676494851708412e-7, scales=[1, 1e-8, 1e-14, 1e-24],
            scope='Observed CPU integration result before preparing the run; includes zeroed-row KV alignment.'),
        selection='Two prospective attention-proposal variants after the fixed-evidence FFN-acceptance pilot failed. '
                  'Do not change FFN thresholds. Compare both with native SVAR and original risk; '
                  'the half-step control distinguishes conservative strength from geometric reweighting.',
        development_screen='For either candidate: native-SVAR CHAIRs/Ci/false mentions strictly lower; '
                           'F1 loss<=1.5pp, recall loss<=2pp, leave-largest-reduction Ci no worse, '
                           'plus at least one image-reviewed genuine unsupported-object correction. '
                           'Report both variants, including negative outcomes; no post-hoc threshold change.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gradient_scope={name: previous['recipe']['gradient_scope'][RISK] for name in NAMES},
        reference='Compute the original risk-gated Q/P-guided attention proposal b=a+delta once.',
        half='a_new=a+.5*delta.',
        geometric='On changed support G per head, a_new_G=m_G*sqrt(a_G*b_G)/sum sqrt(a_G*b_G); '
                  'all other entries fixed. Equal-weight KL(q||a)+KL(q||b) midpoint.',
        finite_precision='Clip only sign errors at the original support boundary; keep the1e-6 per-head mass assertion.',
        directional_bound='Keep exactly the reference signed support, so target-opposing visual routes gain '
                          'and target-supporting history routes lose. Record a nonpositive upper bound, '
                          'not the old reference numeric prediction as a new numeric prediction.',
        unchanged='Risk/class gate, base5--18, source/action19, gradients, FFN guidance, actual-probability '
                  'acceptance, tail20--35 replay and original image-only E_l caches.',
        extra_cost='Algebra on cached scores and the existing budget only; no extra gradient, model, MLP or LM-head forwards.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_svar_attention_midpoint import Qwen3BatchSvarAttentionMidpoint
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config

    torch.set_num_threads(1)
    plan, trace = shared.read(OUT / 'protocol.json'), shared.read(TRACE / 'protocol.json')
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL), hf_name=str(MODEL_ROOT))
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    history = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    if gpu == 0:
        from scripts.verify_qwen3_svar_attention_midpoint import verify
        shared.write(OUT / 'midpoint_preflight.json', verify(wrapper, probe, allowed))
    groups = shared.source_groups(SOURCE / f'coco100/svar.shard{gpu}.jsonl')
    assert len(groups) == len(trace['groups'][gpu])
    files = {name: OUT / f'coco100/{name}.shard{gpu}.jsonl' for name in NAMES}
    assert not any(path.exists() for path in files.values()), 'Inspect existing output; no partial resume.'
    verified = False
    with torch.inference_mode():
        for index, group in enumerate(groups):
            ids = [row['image_id'] for row in group]
            (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [image for image, _ in inputs] == ids == trace['groups'][gpu][index]['ids']
            path = TRACE / f'semantics/shard{gpu}.batch{index}.npz'
            assert shared.sha(path) == plan['source_sha256'][str(path)]
            with np.load(path, allow_pickle=False) as saved:
                assert saved['image_ids'].tolist() == ids and saved['span'].tolist() == list(span)
                semantic_tokens = saved['token_ids'].tolist()
                cache = torch.from_numpy(saved['E_l'].copy()).to(wrapper.device)
            if not verified:
                tokens, _, _ = common.generate_group(wrapper, span, inputs, nouns, 'svar', 64)
                assert tokens == [row['response_token_ids'][:64] for row in group]
                shared.write(OUT / f'verification.gpu{gpu}.json', dict(exact_svar_prefix=True, images=ids, tokens=64))
                verified = True
            for geometry_mode, name in zip(MODES, NAMES, strict=True):
                def factory(model, current_span, _nouns, mode, strength):
                    assert mode == 'next' and strength == 5.
                    return Qwen3BatchSvarAttentionMidpoint(model, current_span, allowed, probe=probe,
                        semantic_tokens=semantic_tokens, cached_semantics=cache, guided=True,
                        tokenizer=wrapper.tokenizer, history_parser=history, geometry_mode=geometry_mode)
                with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                    tokens, stats, seconds = original.generate_group(wrapper, span, inputs, nouns,
                        'sadi_sar_ems_next_19_20', plan['max_new_tokens'])
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                with files[name].open('a', encoding='utf-8') as stream:
                    for image, token_ids, text, info in zip(ids, tokens, texts, stats, strict=True):
                        assert 0 < len(token_ids) <= plan['max_new_tokens']
                        assert info['local_recompute_events'] == info['guided_attention_calls']
                        assert not info['diagnostic_force_probe_gate'] and not info['diagnostic_force_guidance']
                        row = dict(image_id=image, model=MODEL, variant=name, generated_text=text.strip(),
                            response_token_ids=token_ids, seconds=seconds / len(ids), batch_size=len(ids),
                            batch_wall_seconds=seconds, visual_token_count=span[1] - span[0],
                            sar_ems_statistics=info, decoder='native_qwen3_generate_same_span_batch')
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f'[{gpu}] {name} batch={len(ids)} seconds={seconds:.3f}', flush=True)
    assert all(shared.group_keys(shared.source_groups(path)) == shared.group_keys(groups) for path in files.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare):
            driver.run()
    else:
        worker(args.worker)
