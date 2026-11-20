"""Add all fixed hash controls to the already observed SVAR500, reusing its caches."""

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
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_svar_matched_hash_pilot as pilot

OUT = shared.OUT / 'generation_time_evidence_drift_qwen3_svar_hash_cached500'
SOURCE = shared.OUT / 'generation_time_evidence_drift_qwen3_fresh_svar_first_class500'
SEEDS, NAMES, RISK = pilot.SEEDS, pilot.NAMES, pilot.RISK


def prepare():
    import numpy as np
    assert not OUT.exists(), 'Keep existing outputs; inspect before any restart.'
    assert shared.read(SOURCE / 'run_status.json')['status'] == 'complete'
    assert shared.read(SOURCE / 'post_audit_status.json')['status'] == 'passed'
    assert shared.read(pilot.OUT / 'matched_hash_audit.json')['passed']
    previous, development = shared.read(SOURCE / 'protocol.json'), shared.read(pilot.OUT / 'protocol.json')
    assert previous['max_new_tokens'] == 512 and previous['max_batch'] == 16
    assert development['calibration']['rate'] == 55 / 258
    ids = shared.read(SOURCE / 'cohort.json')['coco500']
    assert len(ids) == len(set(ids)) == 500
    hashes = {}
    for inherited in (previous, development):
        for filename, expected in inherited['source_sha256'].items():
            path = Path(filename)
            if not path.is_absolute():
                path = REPO / path
            assert shared.sha(path) == expected, str(path)
            assert str(path) not in hashes or hashes[str(path)] == expected
            hashes[str(path)] = expected
    cache_lookup = {}
    for path in sorted((SOURCE / 'semantics').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            key = tuple(data['image_ids'].tolist())
            assert key not in cache_lookup
            cache_lookup[key] = dict(path=str(path), ids=list(key), span=data['span'].tolist(),
                                     targets=data['token_ids'].tolist())
        hashes[str(path)] = shared.sha(path)
    controls = ['sadi', RISK, 'svar']
    batches = {}
    for shard in (0, 1):
        groups = shared.source_groups(SOURCE / 'coco500' / f'svar.shard{shard}.jsonl')
        batches[str(shard)] = []
        for group in groups:
            key = tuple(row['image_id'] for row in group)
            cached = cache_lookup[key]
            assert cached['span'][1] - cached['span'][0] == group[0]['visual_token_count']
            batches[str(shard)].append(cached)
        for name in controls:
            path = SOURCE / 'coco500' / f'{name}.shard{shard}.jsonl'
            assert shared.group_keys(shared.source_groups(path)) == shared.group_keys(groups)
            hashes[str(path)] = shared.sha(path)
    assert sum(len(v) for v in batches.values()) == len(cache_lookup) == 64
    for filename in ('scripts/run_qwen3_svar_hash_cached500.py',
                     'scripts/audit_qwen3_svar_hash_cached500.py',
                     'scripts/summarize_qwen3_matched_hash_controls.py'):
        hashes[str(REPO / filename)] = shared.sha(REPO / filename)
    for path in (SOURCE / 'protocol.json', SOURCE / 'cohort.json', SOURCE / 'post_audit_status.json',
                 pilot.OUT / 'protocol.json', pilot.OUT / 'matched_hash_audit.json',
                 pilot.OUT / 'risk_vs_fixed_hash_mean.json'):
        hashes[str(path)] = shared.sha(path)
    OUT.mkdir()
    (OUT / 'coco500').mkdir()
    (OUT / 'logs').mkdir()
    for filename in ('cohort.json', 'object_lexicon.json', 'training_vocabulary.json'):
        shutil.copyfile(SOURCE / filename, OUT / filename)
        hashes[str(OUT / filename)] = shared.sha(OUT / filename)
    for shard in (0, 1):
        for name in controls:
            filename = f'{name}.shard{shard}.jsonl'
            shutil.copyfile(SOURCE / 'coco500' / filename, OUT / 'coco500' / filename)
    plan = dict(source=str(SOURCE), source_protocol_sha256=shared.sha(SOURCE / 'protocol.json'),
        source_sha256=hashes, stage='head-gradient-svar-hash-cached500', cohort='coco500', images=500,
        shards=2, model='Qwen3-VL-8B-Instruct', max_new_tokens=512, max_batch=16,
        variants=[*controls, *NAMES], candidates=list(NAMES), copied_controls=controls,
        cached_controls=['svar'], primary_baseline='svar', source_trace=str(SOURCE),
        probe=development['probe'], calibration=development['calibration'],
        actuation_preflight=development['actuation_preflight'],
        semantic_preflight=development['semantic_preflight'],
        gate_seeds=list(SEEDS), cache_batches=batches, fixed_at=time.time(),
        observed_cohort_before_design=True,
        selection='Contribution ablation on the already observed SVAR500, not a newly sampled independent method test. '
                  'Retain every fixed seed43/44/45; no best-seed promotion or retuning. '
                  'On development100 the risk mean was better but image-paired intervals crossed zero.',
        reuse_policy='Copy all three previous500 arms byte-for-byte; load each of the64 image-only E_l caches '
                     'from its frozen source. Generate only the three new hash arms on the original ordered batches.',
        expected_rate_only='Keep55/258 from development100. Do not recalibrate using observed500 outcomes. '
                           'Realized counts and moved attention mass need not match.',
        recipe=dict(development['recipe']))
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_svar_matched_hash import Qwen3BatchSvarMatchedHash
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config

    torch.set_num_threads(1)
    plan = shared.read(OUT / 'protocol.json')
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL), hf_name=str(MODEL_ROOT))
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    allowed = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    history = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    groups = shared.source_groups(SOURCE / f'coco500/svar.shard{gpu}.jsonl')
    assert len(groups) == len(plan['cache_batches'][str(gpu)])
    files = {name: OUT / f'coco500/{name}.shard{gpu}.jsonl' for name in NAMES}
    assert not any(path.exists() for path in files.values()), 'Inspect existing output; no partial resume.'
    verified = False
    with torch.inference_mode():
        for index, group in enumerate(groups):
            ids = [row['image_id'] for row in group]
            (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [image for image, _ in inputs] == ids == plan['cache_batches'][str(gpu)][index]['ids']
            path = Path(plan['cache_batches'][str(gpu)][index]['path'])
            assert shared.sha(path) == plan['source_sha256'][str(path)]
            with np.load(path, allow_pickle=False) as saved:
                assert saved['image_ids'].tolist() == ids and saved['span'].tolist() == list(span)
                semantic_tokens = saved['token_ids'].tolist()
                cache = torch.from_numpy(saved['E_l'].copy())
            if not verified:
                tokens, _, _ = common.generate_group(wrapper, span, inputs, nouns, 'svar', 64)
                assert tokens == [row['response_token_ids'][:64] for row in group]
                shared.write(OUT / f'verification.gpu{gpu}.json', dict(exact_svar_prefix=True, images=ids, tokens=64))
                verified = True
            for seed, name in zip(SEEDS, NAMES, strict=True):
                def factory(model, current_span, _nouns, mode, strength):
                    assert mode == 'next' and strength == 5.
                    return Qwen3BatchSvarMatchedHash(model, current_span, allowed, probe=probe,
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
