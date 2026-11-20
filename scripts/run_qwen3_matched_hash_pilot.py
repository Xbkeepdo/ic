"""Queue three fixed feature-blind gate controls on the cached100 batches."""

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

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-matched-hash-pilot')
SOURCE, TRACE = base.OUT, base.TRACE
SEEDS = (43, 44, 45)
NAMES = tuple(f'sadi_first_class_hash_{seed}_19' for seed in SEEDS)


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', base.NAME), patch.object(repeat, 'NAME', NAMES[0]):
        plan = repeat.prepare()
    calls = positives = 0
    for shard in (0, 1):
        for group in shared.source_groups(SOURCE / f'coco100/{base.NAME}.shard{shard}.jsonl'):
            for row in group:
                stats = row['sar_ems_statistics']
                calls += stats['probe_calls']
                positives += stats['probe_positive_calls']
    assert (positives, calls) == (23, 208)
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    for name in ('features/qwen3_probe_matched_hash.py', 'scripts/run_qwen3_matched_hash_pilot.py',
                 'scripts/audit_qwen3_matched_hash.py'):
        plan['source_sha256'][str(REPO / name)] = shared.sha(REPO / name)
    plan['source_sha256'][str(OUT / 'object_lexicon.json')] = shared.sha(OUT / 'object_lexicon.json')
    trace = shared.read(TRACE / 'protocol.json')
    for shard, groups in enumerate(trace['groups']):
        for index in range(len(groups)):
            path = TRACE / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    previous = shared.read(SOURCE / 'protocol.json')
    replication = shared.OUT / 'generation_time_evidence_drift_qwen3_replication_first_class500/run_status.json'
    plan.update(stage='head-gradient-matched-hash-pilot', candidates=list(NAMES), max_batch=16,
        variants=[*plan['copied_controls'], *NAMES], source_trace=str(TRACE), probe=previous['probe'],
        actuation_preflight=previous['actuation_preflight'],
        calibration=dict(positives=positives, eligible=calls, rate=positives / calls,
            scope='Existing first-class100 runtime counts only, no labels or outcome fitting.'),
        gate_seeds=list(SEEDS), fixed_at=time.time(),
        replication500_status_at_design=shared.read(replication)['status'],
        selection='Contribution control, not best-seed selection. Report all three fixed seeds and mean. Any500 continuation must retain all seeds.',
        expected_rate_only='Expected eligibility rate matched; realized proposals and moved mass need not be equal after trajectories diverge.')
    plan['recipe'].update(exclude_prior_token_id=True,
        gate='First-token and first-class eligibility, then SHA256(seed|image_id|step|token)<23/208; detector values logged but not used.',
        gradient_scope={name: previous['recipe']['gradient_scope'][base.NAME] for name in NAMES},
        unchanged='Full target-conditioned attention action, FFN spatial guidance, strength, acceptance and cache rollback.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_matched_hash import Qwen3BatchMatchedHashGate
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
    groups = shared.source_groups(SOURCE / f'coco100/sadi.shard{gpu}.jsonl')
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
                cache = torch.from_numpy(saved['E_l'].copy())
            if not verified:
                tokens, _, _ = common.generate_group(wrapper, span, inputs, nouns, 'sadi', 64)
                assert tokens == [row['response_token_ids'][:64] for row in group]
                shared.write(OUT / f'verification.gpu{gpu}.json', dict(exact_sadi_prefix=True, images=ids, tokens=64))
                verified = True
            for seed, name in zip(SEEDS, NAMES, strict=True):
                def factory(model, current_span, _nouns, mode, strength):
                    assert mode == 'next' and strength == 5.
                    return Qwen3BatchMatchedHashGate(model, current_span, allowed, probe=probe,
                        semantic_tokens=semantic_tokens, cached_semantics=cache, guided=True,
                        tokenizer=wrapper.tokenizer, history_parser=history, image_ids=ids,
                        gate_seed=seed, gate_rate=plan['calibration']['rate'])
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
