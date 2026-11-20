"""Freeze both midpoint candidates before sampling a fourth unused500 cohort."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_fresh_svar_first_class500 as reference

fresh = reference.fresh
RUN = fresh.OUT / 'generation_time_evidence_drift_qwen3_fresh_svar_midpoint500'
PILOT = fresh.OUT / 'generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-svar-attention-midpoint-pilot'
REFERENCE_PILOT = reference.PILOT
RISK = reference.NAME
GEOMETRY = {'svar_first_class_half_19': 'half', 'svar_first_class_geometric_19': 'geometric'}
METHODS = ('svar', RISK, *GEOMETRY)


def configure():
    reference.RUN = RUN
    reference.configure()
    fresh.VARIANTS, fresh.SEED, fresh.__file__ = METHODS, 20261122, __file__
    prior = fresh.read(PILOT / 'protocol.json')
    reference_plan = fresh.read(REFERENCE_PILOT / 'protocol.json')
    files = [str(Path(path).relative_to(REPO)) if Path(path).is_relative_to(REPO) else path
             for path in prior['source_sha256'] if Path(path).suffix in ('.py', '.yaml', '.yml')]
    files += [str(PILOT / name) for name in ('protocol.json', 'post_audit_status.json',
              'review_screening.json', 'visual_case_review.json', 'object_lexicon.json',
              'training_vocabulary.json', 'midpoint_preflight.json')]
    files += [reference_plan['actuation_preflight'], reference_plan['semantic_preflight'],
              'scripts/run_qwen3_fresh_svar_midpoint500.py',
              'scripts/audit_qwen3_svar_midpoint500.py', 'scripts/audit_qwen3_primary_denominator.py']
    fresh.SOURCES = tuple(dict.fromkeys((*fresh.SOURCES, *files)))
    prepare_reference = fresh.prepare

    def prepare():
        screening = fresh.read(PILOT / 'review_screening.json')
        assert screening['advance_to_500'] == list(GEOMETRY)
        assert fresh.read(PILOT / 'post_audit_status.json')['status'] == 'passed'
        assert fresh.read(PILOT / 'run_status.json')['status'] == 'complete'
        for path in prior['source_sha256']:
            if Path(path).suffix in ('.py', '.yaml', '.yml'):
                assert fresh.sha(Path(path).read_bytes()) == prior['source_sha256'][path], path
        # The algorithm selection and seed are fixed before inspecting this cohort.
        frozen_before_selection = time.time()
        prepare_reference()
        plan = fresh.read(RUN / 'protocol.json')
        for filename in ('object_lexicon.json', 'training_vocabulary.json'):
            assert (RUN / filename).read_bytes() == (PILOT / filename).read_bytes()
        plan.update(stage='fresh-svar-midpoint500', variants=list(METHODS), candidates=list(GEOMETRY),
            candidate_method=list(GEOMETRY)[0], reference_method=RISK, geometry_modes=list(GEOMETRY.values()),
            source=str(PILOT), source_protocol_sha256=fresh.sha((PILOT / 'protocol.json').read_bytes()),
            prior_protocol=str(PILOT / 'protocol.json'),
            prior_protocol_sha256=fresh.sha((PILOT / 'protocol.json').read_bytes()),
            copied_controls=[], primary_baseline='svar', recipe=prior['recipe'],
            algorithm_fixed_before_selection=frozen_before_selection,
            primary='Both midpoint candidates vs SVAR; original risk is a full matched reference. '
                    'Publish all500, both candidates, both evaluators and paired intervals regardless of outcome.',
            advancement='Exploratory fourth-cohort follow-up. Both100 pilots passed the numerical screen. '
                        'The only image-reviewed positive is a small soap-label portrait detail at47121; '
                        'most other measured fixes remain lexical/context boundaries. Do not overstate this evidence.',
            candidate='Unchanged half and geometric attention budgets from the frozen100 protocol.',
            semantic_preflight=str(RUN / 'svar_cache_preflight.json'),
            actuation_preflight=str(PILOT / 'midpoint_preflight.json'),
            reference_actuation_preflight=reference_plan['actuation_preflight'],
            reference_semantic_preflight=reference_plan['semantic_preflight'],
            semantic_cache='Image-only E_l captured during the SVAR arm of each actual batch, '
                           'cached once on GPU and reused by all three guided arms. No extra SADI generation.',
            batch_policy='Four freshly generated arms share the same realized ordered batches. '
                         'Any OOM discards all four arms of that group and splits all together.',
            timing='SVAR includes shared E_l setup; do not use its time as a pure baseline speed measurement.',
            analysis_primary='Explicit SVAR baseline for mentions and denominator sensitivity. '
                             'No SADI arm is generated on these500.',
            uncertainty='5000 shared-image paired bootstrap draws, seed43. Each candidate interval is exploratory '
                        'and marginal, not family-wise adjusted; do not select a single winner after testing.')
        ids = set(fresh.read(RUN / 'cohort.json')['coco500'])
        previous_names = ('fresh_first_class500', 'replication_first_class500', 'fresh_svar_first_class500')
        previous = [set(fresh.read(fresh.OUT / ('generation_time_evidence_drift_qwen3_' + name) /
                                  'cohort.json')['coco500']) for name in previous_names]
        with np.load(Path(plan['probe']).parents[1] / 'compact_features.npz', allow_pickle=False) as data:
            training = set(map(int, data['image_id']))
        development = set(fresh.read(PILOT / 'cohort.json')['coco100'])
        assert ids.isdisjoint(training | development | set.union(*previous))
        fresh.write(RUN / 'protocol.json', plan)
        fresh.write(RUN / 'cohort_independence.json', dict(passed=True, images=500,
            old_probe_images=len(training), overlap_old_probe_images=len(ids & training),
            overlap_development100=len(ids & development),
            previous_cohorts=list(previous_names), overlap_previous500=[len(ids & p) for p in previous],
            excluded_images=len(fresh.read(RUN / 'selection.json')['excluded_ids']),
            sample_sha256=plan['sample_sha256'], source_count=len(plan['source_sha256'])))
        print('FROZEN_BOTH_MIDPOINT500', plan['sample_sha256'], flush=True)

    fresh.prepare = prepare


def cache_preflight(wrapper, nouns, semantic_tokens):
    import torch
    from features.qwen3_image_semantic_cache import ImageSemanticCache
    from scripts import generation_time_evidence_drift_qwen3_batched as common

    (span, group), = common.same_span_batches(wrapper, [485532, 558955], fresh.COCO, 16)
    observed = []
    with torch.inference_mode():
        for capture in (False, True):
            logits = []
            with ExitStack() as stack:
                reader = stack.enter_context(ImageSemanticCache(wrapper.model, span, semantic_tokens).capture()) if capture else None
                handle = wrapper.model.register_forward_hook(lambda _m, _a, out: logits.append(out.logits[:, -1].cpu()))
                stack.callback(handle.remove)
                tokens, _, _ = common.generate_group(wrapper, span, group, nouns, 'svar', 16)
            if reader is not None:
                assert len(reader.semantics) == 36
            observed.append((tokens, logits))
    assert observed[0][0] == observed[1][0]
    assert all(torch.equal(a, b) for a, b in zip(observed[0][1], observed[1][1], strict=True))
    return dict(passed=True, images=[485532, 558955], steps=16,
                exact_native_svar_logits_and_tokens=True, semantic_targets=len(semantic_tokens))


def worker(shard):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_image_semantic_cache import ImageSemanticCache
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass
    from features.qwen3_svar_attention_midpoint import Qwen3BatchSvarAttentionMidpoint
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config

    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL), hf_name=str(MODEL_ROOT))
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    plan = fresh.read(RUN / 'protocol.json')
    probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
    semantic_tokens = sorted(fresh.read(RUN / 'training_vocabulary.json')['token_ids'])
    allowed = frozenset(fresh.read(RUN / 'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    native_generate = common.generate_group
    cache = None
    verified_batch_prefix = False
    if shard == 0:
        fresh.write(RUN / 'svar_cache_preflight.json', cache_preflight(wrapper, nouns, semantic_tokens))

    def generate(model_wrapper, span, group, noun_ids, method, cap):
        nonlocal cache, verified_batch_prefix
        ids = [image for image, _ in group]
        if method == 'svar':
            cache = None
            prefix = None
            if not verified_batch_prefix:
                prefix, _, _ = native_generate(model_wrapper, span, group, noun_ids, method, 64)
            with ImageSemanticCache(wrapper.model, span, semantic_tokens).capture() as reader:
                result = native_generate(model_wrapper, span, group, noun_ids, method, cap)
            if prefix is not None:
                assert prefix == [row[:64] for row in result[0]]
                fresh.write(RUN / f'cache_prefix.gpu{shard}.json', dict(passed=True, images=ids,
                    batch_size=len(ids), steps=64, exact_native_svar_prefix_with_cache=True))
                verified_batch_prefix = True
            values = torch.stack([reader.semantics[i] for i in range(36)])
            assert values.device == torch.device(wrapper.device)
            cache = dict(ids=ids, span=tuple(span), E_l=values)
            key = fresh.sha(str((ids, tuple(span))).encode())[:20]
            np.savez_compressed(RUN / 'semantics' / f'shard{shard}.{key}.npz', image_ids=ids,
                                span=span, token_ids=semantic_tokens, E_l=values.cpu().numpy())
            return result
        assert method in (RISK, *GEOMETRY) and cache is not None
        assert cache['ids'] == ids and cache['span'] == tuple(span)

        def factory(model, current_span, _nouns, mode, strength):
            assert mode == 'next' and strength == 5.
            kwargs = dict(probe=probe, semantic_tokens=semantic_tokens, cached_semantics=cache['E_l'],
                          guided=True, tokenizer=wrapper.tokenizer, history_parser=parser)
            if method == RISK:
                return Qwen3BatchSvarFirstClass(model, current_span, allowed, **kwargs)
            return Qwen3BatchSvarAttentionMidpoint(model, current_span, allowed,
                                                  geometry_mode=GEOMETRY[method], **kwargs)

        with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
            return original.generate_group(model_wrapper, span, group, noun_ids,
                                           'sadi_sar_ems_next_19_20', cap)

    ids = fresh.read(RUN / 'cohort.json')['coco500'][shard::2]
    files = {name: RUN / 'coco500' / f'{name}.shard{shard}.jsonl' for name in METHODS}
    assert not any(path.exists() for path in files.values()), 'Inspect existing output; no partial resume.'
    (RUN / 'semantics').mkdir(exist_ok=True)
    existing = {name: set() for name in METHODS}
    with torch.inference_mode(), patch.object(common, 'generate_group', generate):
        for span, group in common.same_span_batches(wrapper, ids, fresh.COCO, fresh.BATCH):
            common.write_group(wrapper, span, group, nouns, existing, files, fresh.CAP, shard)
    assert all(values == set(ids) for values in existing.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    configure()
    if args.worker is not None:
        worker(args.worker)
    elif args.evaluate:
        fresh.evaluate()
    else:
        fresh.run()
