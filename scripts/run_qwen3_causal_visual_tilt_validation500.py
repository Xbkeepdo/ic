"""Prospective500 validation of the original108-gated uniform adjacent tilt."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.run_qwen3_block_acg_fresh500 import ids_sha
from scripts.run_qwen3_causal_visual_tilt_pilot import decorate_statistics

SOURCE = shared.OUT/'qwen3_causal_visual_tilt_pilot_20261119'
RESERVED = shared.OUT/'qwen3_causal_visual_tilt_reserved500_20261119'
OUT = shared.OUT/'qwen3_causal_visual_tilt_validation500_20261119'
VARIANTS = ['sadi', 'svar', 'causal_fixed']
FILES = ['scripts/run_qwen3_causal_visual_tilt_validation500.py',
         'scripts/audit_qwen3_causal_visual_tilt_validation500.py',
         'scripts/analyze_qwen3_causal_visual_tilt_validation500.py']
AUDITS = ['scripts/audit_qwen3_causal_visual_tilt_validation500.py',
          'scripts/audit_qwen3_caption_with_native_cache_v2.py',
          'scripts/analyze_qwen3_causal_visual_tilt_validation500.py',
          'scripts/audit_qwen3_caption_denominator.py']


def prepare():
    assert not OUT.exists(), 'Inspect existing outputs and workers; never overwrite or redraw'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    assert shared.read(SOURCE/'post_audit_status.json')['status'] == 'passed'
    human = shared.read(SOURCE/'human_review.json')
    assert human['passed'] and human['review_complete']
    assert human['advance_candidates_to_new_frozen_test'] == ['causal_fixed']
    assert 'causal_fixed' in shared.read(SOURCE/'screening.json')['advance_to_500']
    for name, value in human['input_sha256'].items():
        assert shared.sha(SOURCE/name) == value, name
    reservation = shared.read(RESERVED/'reservation.json'); selection = shared.read(RESERVED/'selection.json')
    ids = shared.read(RESERVED/'cohort.json')['coco500']
    assert reservation['before_causal_pilot_metrics'] and reservation['reserve_only']
    assert len(ids) == len(set(ids)) == 500
    assert ids_sha(ids) == reservation['sample_sha256'] == selection['sample_sha256']
    assert not set(ids) & set(selection['excluded_ids'])
    assert not set(ids) & set(shared.read(SOURCE/'cohort.json')['coco100'])
    assert not set(ids) & set(shared.read(shared.OUT/'qwen3_bounded_visual_tilt_validation500_20261119/cohort.json')['coco500'])
    assert not (RESERVED/'consumed.json').exists(), 'Reserved cohort already assigned'
    for info in selection['sources']:
        assert shared.sha(Path(info['path'])) == info['file_sha256']
    for name, value in reservation['source_sha256'].items():
        assert shared.sha(REPO/name) == value, name
    old = shared.read(SOURCE/'protocol.json'); hashes = dict(old['source_sha256'])
    hashes.update(shared.read(SOURCE/'analysis_protocol.json')['source_sha256'])
    paths = [SOURCE/name for name in ('protocol.json', 'analysis_protocol.json', 'human_review.json',
        'screening.json', 'post_audit_status.json', 'independent_summary.json', 'final_provenance.json',
        'mechanism_audit.json', 'analysis_provenance.json')]
    paths += list((SOURCE/'coco100').glob('*.jsonl'))
    paths += [RESERVED/name for name in ('cohort.json', 'selection.json', 'reservation.json')]
    paths += [REPO/name for name in [*FILES, *AUDITS,
        'scripts/audit_qwen3_bounded_visual_tilt_validation500.py',
        'scripts/analyze_qwen3_bounded_visual_tilt_validation500.py']]
    for path in paths:
        hashes[str(path)] = shared.sha(path)
    for name, value in hashes.items():
        assert shared.sha(REPO/name) == value, name
    OUT.mkdir(); (OUT/'logs').mkdir(); (OUT/'coco500').mkdir()
    for name in ('cohort.json', 'selection.json'):
        shutil.copyfile(RESERVED/name, OUT/name)
    shutil.copyfile(SOURCE/'object_lexicon.json', OUT/'object_lexicon.json')
    recipe = dict(old['recipe'])
    recipe.update(cache='No generation reuse. Image hpre/GQA/semantics cached in RAM within each batch.',
        mode='Only fixed .5, refresh=False, gamma=1; original108 feature gate unchanged.',
        unchanged_from_pilot=True)
    plan = dict(stage='independent-causal500', model='Qwen3-VL-8B-Instruct', images=500, shards=2,
        cohort='coco500', max_batch=16, max_new_tokens=512, prompt='Describe this image.',
        variants=VARIANTS, candidates=['causal_fixed'], copied_controls=[], cached_controls=['sadi'],
        base_by_candidate={'causal_fixed': 'sadi'}, mode_by_candidate={'causal_fixed': 'fixed'},
        refresh_by_candidate={'causal_fixed': False}, source=str(SOURCE),
        source_protocol_sha256=shared.sha(SOURCE/'protocol.json'), source_sha256=hashes,
        fixed_at=time.time(), sample_sha256=ids_sha(ids), reservation=str(RESERVED),
        reservation_sha256=shared.sha(RESERVED/'reservation.json'),
        probe=old['probe'], real_threshold=old['real_threshold'], eos_token_ids=old['eos_token_ids'], recipe=recipe,
        generation='All three methods fresh on identical ordered same-span batches; sorted IDs/stride2 shards, BF16/eager/greedy.',
        primary='Full500 sentence Cs/Ci, counts and recall/F1; paired against both SADI and SVAR.',
        interpretation='This candidate uses our original108 gate with uniform enhancement; it does not establish a benefit of Q/P-directed or refreshed budgets.',
        hypothesis='Test whether the gated uniform intervention transfers beyond the development100 and its repeated-table case.',
        failure_policy='No automatic OOM splits, rebatching or retries; preserve completed outputs and inspect actual process state.',
        no_validation_tuning=True)
    shared.write(OUT/'protocol.json', plan)
    analysis_files = [*AUDITS, 'scripts/analyze_qwen3_bounded_visual_tilt_validation500.py',
        'scripts/audit_qwen3_visual_feedback_mentions.py', 'scripts/audit_qwen3_fresh_caption500.py',
        'scripts/qwen3_caption_groups_with_cache.py']
    shared.write(OUT/'analysis_protocol.json', dict(fixed_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'), before_inspecting_validation_metrics=True,
        source_sha256={str(REPO/name): shared.sha(REPO/name) for name in analysis_files},
        primary='Full500 primary; report all three methods and both evaluators, paired5000 seed43 uncertainty.',
        cases='All fixes/introductions and top3 nonzero false/recall changes, ties by ID; original photos.',
        sensitivity='Leave-largest/top3, repeated false and real mentions, shared uncapped outputs are secondary.',
        decision='No success from numerical CHAIR alone; inspect object identity and compare both SADI and SVAR. Attribute to the gated uniform component only.'))
    shared.write(RESERVED/'consumed.json', dict(assigned_to=str(OUT), assigned_at=time.time(),
        generation_protocol_sha256=shared.sha(OUT/'protocol.json'), sample_sha256=ids_sha(ids)))
    print('CAUSAL_RESERVED500_ASSIGNED', ids_sha(ids), flush=True)
    return plan


def verify_first_group(rows):
    native, candidate = rows['sadi'], rows['causal_fixed']; exact = 0
    for a, b in zip(native, candidate, strict=True):
        assert a['image_id'] == b['image_id']
        stat = b['sar_ems_statistics']; kept = [e for e in stat['full_joint_events'] if e['accepted']]
        cut = min((e['step'] for e in kept), default=len(b['response_token_ids']))
        assert a['response_token_ids'][:cut] == b['response_token_ids'][:cut]
        for event in stat['full_joint_events']:
            if event['step'] <= cut:
                assert event['token'] == a['response_token_ids'][event['step']]
        if not kept:
            assert a['response_token_ids'] == b['response_token_ids'] and a['generated_text'] == b['generated_text']
            exact += 1
    return dict(passed=True, image_ids=[r['image_id'] for r in native], no_kept_rows_exact_to_sadi=exact,
        native_prefix_exact_until_first_accepted_proposal=True, same_ordered_batch=True,
        implementation_preflight_reused='qwen3_causal_visual_tilt_preflight_20261119: B2 and B16 passed; neural source unchanged.')


def worker(gpu):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_causal_visual_tilt import Qwen3BatchCausalVisualTilt
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    plan = shared.read(OUT/'protocol.json')
    for name, value in plan['source_sha256'].items():
        assert shared.sha(REPO/name) == value, name
    wrapper = build_wrapper(); probe = FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed = frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ'); assert nouns
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    eos = plan['eos_token_ids']; eos = set(eos if isinstance(eos, list) else [eos])
    ids = shared.read(OUT/'cohort.json')['coco500'][gpu::2]
    paths = {name: OUT/f'coco500/{name}.shard{gpu}.jsonl' for name in VARIANTS}
    assert not any(path.exists() for path in paths.values())
    memory = []
    with torch.inference_mode():
        for index, (span, group) in enumerate(common.same_span_batches(wrapper, ids, COCO_ROOT, 16)):
            images = [image for image, _ in group]; static = {}; first = {}
            for name in VARIANTS:
                torch.cuda.reset_peak_memory_stats(wrapper.device)
                if name in ('sadi', 'svar'):
                    tokens, stats, seconds = common.generate_group(wrapper, span, group, nouns, name, 512)
                    calls = None
                else:
                    holder = {}
                    def factory(model, current_span, _nouns, mode, strength):
                        controller = Qwen3BatchCausalVisualTilt(model, current_span, allowed, probe=probe,
                            semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer, history_parser=parser,
                            shared_static=static, real_threshold=plan['real_threshold'],
                            tilt_mode='fixed', refresh=False, gamma=1.)
                        holder['controller'] = controller
                        return controller
                    with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                        tokens, stats, seconds = original.generate_group(wrapper, span, group, allowed, shared.POSITIVE, 512)
                    calls = holder['controller'].steps; holder.clear()
                    decorate_statistics(stats, False, eos)
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                rows = [dict(image_id=image, model=MODEL, variant=name, response_token_ids=t, generated_text=text.strip(),
                    seconds=seconds/len(images), batch_wall_seconds=seconds, batch_size=len(images),
                    visual_token_count=span[1]-span[0], sar_ems_statistics=stat,
                    decoder='native_qwen3_generate_same_span_batch', generation_reused=False,
                    actual_model_calls_in_batch=calls)
                    for image, t, text, stat in zip(images, tokens, texts, stats, strict=True)]
                with paths[name].open('a', encoding='utf-8') as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                if index == 0:
                    first[name] = rows
                memory.append(dict(group=index, variant=name, batch_size=len(images), image_ids=images,
                    visual_tokens=span[1]-span[0], peak=torch.cuda.max_memory_allocated(wrapper.device)))
                shared.write(OUT/f'memory.gpu{gpu}.json', memory)
                del stats; gc.collect(); torch.cuda.empty_cache()
                print(f'[{gpu}] group={index} {name} B={len(images)} seconds={seconds:.3f}', flush=True)
            if index == 0:
                shared.write(OUT/f'first_batch.gpu{gpu}.json', verify_first_group(first))
            del first; static.clear(); gc.collect(); torch.cuda.empty_cache()
    for path in paths.values():
        assert {row['image_id'] for group in shared.source_groups(path) for row in group} == set(ids)


def run():
    with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), patch.object(driver, 'prepare', prepare):
        driver.run()
    commands = [[name, str(OUT)] for name in AUDITS]
    shared.write(OUT/'post_audit_status.json', dict(status='running', commands=commands, started=time.time()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
    try:
        for index, args in enumerate(commands):
            with (OUT/f'post_audit{index}.log').open('x') as handle:
                subprocess.run([sys.executable, *args], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        assert all(shared.read(OUT/name)['passed'] for name in
                   ('mechanism_audit.json', 'independent_summary.json', 'analysis_provenance.json'))
        assert (OUT/'denominator_audit.json').is_file()
        shared.write(OUT/'post_audit_status.json', dict(status='passed', commands=commands, finished=time.time()))
    except BaseException as exc:
        shared.write(OUT/'post_audit_status.json', dict(status='failed', error=repr(exc), commands=commands, failed=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    run() if args.worker is None else worker(args.worker)
