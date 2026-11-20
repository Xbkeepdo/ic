"""Transfer the unchanged first-class layer19 action from SADI to SVAR."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_probe_first_class_pilot as base
from scripts import run_qwen3_repeat_guidance as repeat
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-svar-first-class-pilot')
SOURCE, TRACE = base.OUT, base.TRACE
NAME = 'svar_probe_first_class_recycle_19'


def prepare():
    assert shared.read(SOURCE / 'first_class_audit.json')['passed']
    with patch.object(repeat, 'OUT', OUT), patch.object(repeat, 'SOURCE', SOURCE), \
            patch.object(repeat, 'PREVIOUS', base.NAME), patch.object(repeat, 'NAME', NAME):
        plan = repeat.prepare()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    previous = shared.read(SOURCE / 'protocol.json')
    for name in ('scripts/run_qwen3_svar_first_class_pilot.py', 'scripts/audit_qwen3_svar_first_class.py',
                 'scripts/verify_qwen3_early_recycle.py', 'features/qwen3_svar_layer_feedback.py',
                 'features/qwen3_image_semantic_cache.py', 'scripts/audit_qwen3_probe_first_class.py',
                 'scripts/audit_qwen3_visual_feedback_mentions.py', 'scripts/audit_qwen3_head_feedback.py',
                 'scripts/audit_qwen3_cached_caption_pilot.py', 'scripts/audit_qwen3_caption_denominator.py'):
        plan['source_sha256'][str(REPO / name)] = shared.sha(REPO / name)
    plan['source_sha256'][str(OUT / 'object_lexicon.json')] = shared.sha(OUT / 'object_lexicon.json')
    for shard, groups in enumerate(shared.read(TRACE / 'protocol.json')['groups']):
        for index in range(len(groups)):
            path = TRACE / f'semantics/shard{shard}.batch{index}.npz'
            plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-svar-first-class-pilot', max_batch=16,
        variants=[*plan['copied_controls'], 'svar', NAME], primary_baseline='svar',
        source_trace=str(TRACE), probe=previous['probe'],
        actuation_preflight=str(OUT / 'svar_recycle_preflight.json'),
        semantic_preflight=str(OUT / 'svar_semantic_reuse_preflight.json'),
        native_svar='Fresh paired controls: no cached SVAR captions match these100 IDs and ordered batches.',
        semantic_reuse='Reuse all22 old image-only E_l caches; GPU0 checks its first native-SVAR batch against the exact cached values.',
        selection='Transfer only the base operator; do not tune the detector, vocabulary, attention action or acceptance on SVAR outputs.')
    plan['recipe'].update(exclude_prior_token_id=True, base_variant='svar',
        base_operator='Exact BF16 in-place SVAR visual logits add0.5*mean_heads(abs(logits)) at layers5--18.',
        local_I='Read scores before the local SVAR edit; residuals and generation remain SVAR.',
        unchanged='Original first-class I/E_l gate, layer19 final-target gradient, FFN spatial drift, rho=.5D, probability-decrease acceptance and tail replay.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def worker(gpu):
    import numpy as np
    import torch
    from features import qwen3_sar_ems_layer_feedback as feedback
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_image_semantic_cache import ImageSemanticCache
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
    from features.qwen3_svar_layer_feedback import svar_visual_logits
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts import verify_qwen3_early_recycle as check
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
    native_controller = check.Qwen3BatchLateAttention

    def svar_baseline(model, span, targets, _variant):
        return native_controller(model, span, targets, 'svar')

    def first_class(model, span, targets, **kwargs):
        return Qwen3BatchProbeFirstClass(model, span, targets, tokenizer=wrapper.tokenizer,
                                        history_parser=history, **kwargs)

    with torch.inference_mode():
        if gpu == 0:
            with patch.object(check, 'Qwen3BatchLateAttention', svar_baseline), \
                    patch.object(check, 'Qwen3BatchProbeEarlyRecycle', first_class), \
                    patch.object(feedback, 'sadi_visual_logits', svar_visual_logits):
                preflight = check.verify(wrapper, probe, allowed)
            preflight['baseline_variant'] = 'svar'
            shared.write(OUT / 'svar_recycle_preflight.json', preflight)
        groups = shared.source_groups(SOURCE / f'coco100/sadi.shard{gpu}.jsonl')
        files = {name: OUT / f'coco100/{name}.shard{gpu}.jsonl' for name in ('svar', NAME)}
        assert not any(path.exists() for path in files.values()), 'Inspect existing output; no partial resume.'
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
            if index == 0:
                tokens, _, _ = common.generate_group(wrapper, span, inputs, nouns, 'sadi', 64)
                assert tokens == [row['response_token_ids'][:64] for row in group]
                shared.write(OUT / f'verification.gpu{gpu}.json', dict(exact_copied_sadi_prefix=True, images=ids, tokens=64))
            for name in ('svar', NAME):
                if name == 'svar':
                    if gpu == 0 and index == 0:
                        with ImageSemanticCache(wrapper.model, span, semantic_tokens).capture() as reader:
                            tokens, stats, seconds = common.generate_group(wrapper, span, inputs, nouns, name, 512)
                        measured = torch.stack([reader.semantics[i] for i in range(36)]).cpu()
                        assert torch.equal(measured, cache), float((measured - cache).abs().max())
                        shared.write(OUT / 'svar_semantic_reuse_preflight.json', dict(passed=True,
                            source=str(path), images=ids, exact_E_l=True, max_abs_difference=0.,
                            timing='First native-SVAR batch includes the cache-reuse check; not a pure speed benchmark.'))
                    else:
                        tokens, stats, seconds = common.generate_group(wrapper, span, inputs, nouns, name, 512)
                else:
                    def factory(model, current_span, _nouns, mode, strength):
                        assert mode == 'next' and strength == 5.
                        return Qwen3BatchProbeFirstClass(model, current_span, allowed, probe=probe,
                            semantic_tokens=semantic_tokens, cached_semantics=cache, guided=True,
                            tokenizer=wrapper.tokenizer, history_parser=history)
                    with patch.object(original, 'Qwen3BatchLayerFeedback', factory), \
                            patch.object(feedback, 'sadi_visual_logits', svar_visual_logits):
                        tokens, stats, seconds = original.generate_group(wrapper, span, inputs, nouns,
                            'sadi_sar_ems_next_19_20', 512)
                    for info in stats:
                        info.update(base_variant='svar', local_I_before_base_edit='svar')
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                with files[name].open('a', encoding='utf-8') as stream:
                    for image, token_ids, text, info in zip(ids, tokens, texts, stats, strict=True):
                        assert 0 < len(token_ids) <= 512
                        stream.write(json.dumps(dict(image_id=image, model=MODEL, variant=name,
                            generated_text=text.strip(), response_token_ids=token_ids, seconds=seconds / len(ids),
                            batch_size=len(ids), batch_wall_seconds=seconds, visual_token_count=span[1] - span[0],
                            sar_ems_statistics=info, decoder='native_qwen3_generate_same_span_batch'), ensure_ascii=False) + '\n')
                print(f'[{gpu}] {name} batch={len(ids)} seconds={seconds:.3f}', flush=True)
        assert all(shared.group_keys(shared.source_groups(path)) == shared.group_keys(groups) for path in files.values())


def evaluate(output, plan):
    for filename, expected in plan['source_sha256'].items():
        assert shared.sha(Path(filename)) == expected, filename
    for mode in ('sentence', 'legacy'):
        subprocess.run([sys.executable, str(REPO / 'scripts/evaluate_caption_chair.py'),
            '--generation-root', str(output), '--cohort', 'coco100', '--limit', '100',
            '--variants', *plan['variants'], '--evaluator', mode], cwd=REPO, check=True)
    folder = output / 'evaluations/sentence_v1/coco100'
    metric, = folder.glob('metrics.*.json')
    per, = folder.glob('chair_per_image*.jsonl')
    values = shared.read(metric)['methods']
    rows = [json.loads(line)['methods'] for line in per.read_bytes().splitlines()]
    base_value, candidate = values['svar'], values[NAME]
    false = {name: sum(r[name]['hallucinated_mentions'] for r in rows) for name in ('svar', NAME)}
    largest = max(range(100), key=lambda i: rows[i]['svar']['hallucinated_mentions'] - rows[i][NAME]['hallucinated_mentions'])
    trimmed = {name: sum(r[name]['hallucinated_mentions'] for i, r in enumerate(rows) if i != largest) /
               max(1, sum(r[name]['object_mentions'] for i, r in enumerate(rows) if i != largest)) for name in ('svar', NAME)}
    checks = dict(CHAIRs_decreases=candidate['chair_s'] < base_value['chair_s'],
        CHAIRi_decreases=candidate['chair_i'] < base_value['chair_i'], false_mentions_decrease=false[NAME] < false['svar'],
        F1_loss_at_most_1_5pp=candidate['object_f1'] >= base_value['object_f1'] - .015,
        recall_loss_at_most_2pp=candidate['object_recall'] >= base_value['object_recall'] - .02,
        CHAIRi_no_worse_without_largest_reduction=trimmed[NAME] <= trimmed['svar'])
    shared.write(output / 'screening.json', dict(primary_baseline='svar', checks=checks,
        advance_to_500=[NAME] if all(checks.values()) else [], false_mentions=false, leave_one_out_chair_i=trimmed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare), patch.object(shared, 'evaluate', evaluate):
            driver.run()
    else:
        worker(args.worker)
