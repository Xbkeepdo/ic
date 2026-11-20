"""Two target-semantic guidance policies after complete SADI on cached100."""

import argparse
import os
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

OUT = shared.OUT / 'qwen3_semantic_final_pilot_20261118'
SOURCE = base.OUT
CPU = shared.OUT / 'qwen3_semantic_final_cpu/verification.json'
PREFLIGHT = shared.OUT / 'qwen3_semantic_final_preflight/verification.json'
MODES = {'sadi_final_semantic_sar_31': 'full_sar',
         'sadi_final_semantic_increment_31': 'semantic_increment'}
FILES = ['features/qwen3_semantic_final_feedback.py',
         'scripts/verify_qwen3_semantic_final.py',
         'scripts/verify_qwen3_semantic_final_gpu.py',
         'scripts/run_qwen3_semantic_final_pilot.py',
         'scripts/audit_qwen3_semantic_final_pilot.py']


def prepare():
    assert not OUT.exists(), 'Inspect existing run rather than relaunching'
    previous = shared.read(SOURCE / 'protocol.json')
    hashes = dict(previous['source_sha256'])
    for name, expected in hashes.items():
        assert shared.sha(Path(name)) == expected, name
    for path in (CPU, PREFLIGHT):
        assert shared.read(path)['passed'], path
    for name, expected in shared.read(PREFLIGHT)['source_sha256'].items():
        assert shared.sha(REPO / name) == expected, name
    OUT.mkdir()
    (OUT / 'coco100').mkdir()
    (OUT / 'logs').mkdir()
    shutil.copyfile(SOURCE / 'cohort.json', OUT / 'cohort.json')
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    controls = ['sadi', base.NAME]
    for shard in (0, 1):
        for name, origin in [(n, SOURCE) for n in controls]+[('svar', svar.OUT)]:
            path = origin / f'coco100/{name}.shard{shard}.jsonl'
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT / 'coco100' / path.name)
    for path in [CPU, PREFLIGHT, SOURCE / 'protocol.json', SOURCE / 'object_lexicon.json',
                 OUT / 'object_lexicon.json', *(REPO / name for name in FILES),
                 REPO / 'scripts/audit_qwen3_cached_caption_pilot.py',
                 REPO / 'scripts/audit_qwen3_caption_denominator.py']:
        hashes[str(path)] = shared.sha(path)
    plan = dict(stage='semantic-final-pilot', source=str(SOURCE), cohort='coco100', images=100,
        shards=2, max_new_tokens=512, max_batch=16, model='Qwen3-VL-8B-Instruct',
        variants=[*controls, 'svar', *MODES], candidates=list(MODES),
        cached_controls=['sadi'], copied_controls=controls,
        additional_copied_controls={'svar': str(svar.OUT)},
        source_sha256=hashes, source_protocol_sha256=shared.sha(SOURCE / 'protocol.json'),
        fixed_at=time.time(),
        recipe=dict(base_variant='sadi', base_layers=[5, 18], read_action_layer=31,
            target='Actual native final logits argmax, not a provisional logit-lens argmax.',
            gate='Unseen lexical token309; class absent from generated prefix; local final-norm FFN target gain>0. No learned probe.',
            log_s='log(1+V*p(y|final_norm(hpre_visual31)))',
            evidence={'full_sar': 'Normalize(||sum_heads a_hj W_Oh v_hj|| * (1+log_s_j))',
                      'semantic_increment': 'Normalize(||sum_heads a_hj W_Oh v_hj|| * log_s_j), the F_log-I component'},
            preference='softmax(cos(hpre_visual31,u31)/.03)',
            budget='5*Hellinger_squared(Q,P)*tanh(gain/2)*relu(Q-P)/max(relu(Q-P),1e-6)',
            action='Existing spatial-disagreement logit shift, per-head visual mass preserved.',
            acceptance='Apply finite positive budget; no original-target probability-decrease acceptance.',
            gradient_scope={name: 'none' for name in MODES},
            cache='Per-image selected309 semantic columns and FP32 write Gram built lazily once; share across the two policies after exact image-prefix checks.',
            cost='Extra full-vocabulary head projection in image chunks64 once per built image, plus one head call per replay step; no backward or extra encoder forward.',
            replay='Full batch shape, layer31 current query using cached QKV; layers32--35 current KV only.',
            saved='Captions and scalar events only; no hidden/attention arrays saved.'),
        development_screen='Vs SADI Cs/Ci/false count lower; F1 loss<=1.5pp, recall loss<=2pp, leave-largest Ci no worse and image-confirmed object correction.',
        followup='Only if numerical and image review support a candidate, run the predeclared spatial permutation control, then an independent500.',
        planned_spatial_control='Same eligible policy and budget; roll the existing complete spatial shift by M//3 before visual mass correction. Trajectories and eventual dose can differ.',
        limitations='Development100 reused for comparison, not independent evidence; EMS uses a Hellinger surrogate, not original exact32-point OT risk.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def build_wrapper():
    import torch
    from models import build_model
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config
    torch.set_num_threads(1)
    cfg = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    cfg['hf_name'] = str(MODEL_ROOT)
    wrapper = build_model(MODEL, cfg, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    return wrapper


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_semantic_final_feedback import Qwen3BatchSemanticFinal
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    plan = shared.read(OUT / 'protocol.json')
    for name, expected in plan['source_sha256'].items():
        assert shared.sha(Path(name)) == expected, name
    wrapper = build_wrapper()
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    groups = shared.source_groups(OUT / f'coco100/sadi.shard{gpu}.jsonl')
    paths = {name: OUT / f'coco100/{name}.shard{gpu}.jsonl' for name in MODES}
    assert not any(p.exists() for p in paths.values())
    with torch.inference_mode():
        for index, group in enumerate(groups):
            ids = [r['image_id'] for r in group]
            (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [i for i, _ in inputs] == ids and span[1]-span[0] == group[0]['visual_token_count']
            if index == 0:
                tokens, _, _ = original.generate_group(wrapper, span, inputs, allowed, 'sadi', 32)
                assert tokens == [r['response_token_ids'][:32] for r in group]
                shared.write(OUT / f'verification.gpu{gpu}.json', dict(passed=True, image_ids=ids,
                    actual_batch=len(ids), exact_native_prefix_tokens=32))
            static = {}
            for name, mode in MODES.items():
                def factory(model, current_span, nouns, mode_, strength):
                    return Qwen3BatchSemanticFinal(model, current_span, allowed, tokenizer=wrapper.tokenizer,
                        history_parser=parser, semantic_mode=mode, static_cache=static, strength=strength)
                # The established generation entry point passes mode/strength as keywords.
                def adapted(model, current_span, nouns, mode, strength):
                    return factory(model, current_span, nouns, mode, strength)
                with patch.object(original, 'Qwen3BatchLayerFeedback', adapted):
                    tokens, stats, seconds = original.generate_group(wrapper, span, inputs, allowed, shared.POSITIVE, 512)
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                with paths[name].open('a', encoding='utf-8') as stream:
                    for image_id, current, text, info in zip(ids, tokens, texts, stats, strict=True):
                        assert 0 < len(current) <= 512 and info['gradient_probe_calls'] == 0
                        assert info['local_recompute_events'] == info['guided_attention_calls']
                        stream.write(__import__('json').dumps(dict(image_id=image_id, model=MODEL, variant=name,
                            generated_text=text.strip(), response_token_ids=current, seconds=seconds/len(ids),
                            batch_size=len(ids), batch_wall_seconds=seconds, visual_token_count=span[1]-span[0],
                            sar_ems_statistics=info, decoder='native_qwen3_generate_same_span_batch'), ensure_ascii=False)+'\n')
                print(f'[{gpu}] {name} group={index} batch={len(ids)} seconds={seconds:.3f}', flush=True)
            del static
    assert all(shared.group_keys(shared.source_groups(path)) == shared.group_keys(groups) for path in paths.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        prep = prepare
        if args.prepared:
            plan = shared.read(OUT / 'protocol.json')
            assert shared.read(OUT / 'preparation_check.json')['protocol_sha256'] == shared.sha(OUT / 'protocol.json')
            for name, expected in plan['source_sha256'].items():
                assert shared.sha(Path(name)) == expected, name
            prep = lambda: plan
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), patch.object(driver, 'prepare', prep):
            driver.run()
