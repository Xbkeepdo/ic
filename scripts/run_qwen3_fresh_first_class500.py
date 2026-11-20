"""Validate frozen class-first attention recycling on 500 unused COCO images."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_fresh_caption500 as fresh
from scripts import run_qwen3_fresh_head500 as selection

NAME = 'sadi_probe_first_class_recycle_19'
RUN = fresh.OUT / 'generation_time_evidence_drift_qwen3_fresh_first_class500'
PILOT = fresh.OUT / 'generation_time_evidence_drift_qwen3_signed_feedback_head-gradient-probe-first-class-pilot'


def configure():
    selection.RUN, selection.CANDIDATE, selection.PREVIOUS = RUN, NAME, PILOT / 'protocol.json'
    prior = fresh.read(PILOT / 'protocol.json')
    generation_sources = tuple(str(Path(path).relative_to(REPO)) if Path(path).is_relative_to(REPO) else path
        for path in prior['source_sha256'] if Path(path).suffix in ('.py', '.yaml', '.yml'))
    fresh.SOURCES += generation_sources + (
        'features/qwen3_image_semantic_cache.py', 'scripts/run_qwen3_fresh_first_class500.py',
        str(PILOT / 'training_vocabulary.json'), str(PILOT / 'object_lexicon.json'),
        str(Path(prior['probe']) / 'model.pt'), str(Path(prior['probe']) / 'input_scaler.npz'),
    )
    selection.configure()
    fresh.SOURCES = tuple(dict.fromkeys(fresh.SOURCES))
    fresh.CAP, fresh.SEED = 512, 20261119
    fresh.__file__ = __file__
    previous_prepare = fresh.prepare

    def prepare():
        assert fresh.read(PILOT / 'run_status.json')['status'] == 'complete'
        assert fresh.read(PILOT / 'first_class_audit.json')['passed']
        previous_prepare()
        protocol = fresh.read(RUN / 'protocol.json')
        protocol.update(candidate='Frozen first-object-class I/E_l gate and layer19 attention recycling.',
            pilot='100-image development result: CHAIRs unchanged; CHAIRi7.939->7.706; F174.950->75.112. Two fewer false mentions include one color-token artifact.',
            advancement='Exploratory independent follow-up of a weak signal; did not pass the strict CHAIRs-decrease pilot screen.',
            semantic_cache='During native SADI generation, cache original image-only E_l for all496 training targets. Reuse only for identical actual ordered batches.',
            batch_policy='All methods freshly generated on matching realized groups. Any OOM discards all arms for that group, then splits every arm together.',
            timing='SADI wall time includes shared image-semantic setup; report it explicitly, not as a pure SADI timing benchmark.')
        fresh.write(RUN / 'protocol.json', protocol)
    fresh.prepare = prepare


def preflight(wrapper, nouns, semantic_tokens):
    import torch
    from features.qwen3_image_semantic_cache import ImageSemanticCache
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    span, group = next(common.same_span_batches(wrapper, [485532, 558955], fresh.COCO, 16))
    assert len(group) == 2

    def run(read_semantics):
        logits = []
        with ExitStack() as stack:
            if read_semantics:
                reader = stack.enter_context(ImageSemanticCache(wrapper.model, span, semantic_tokens).capture())
            handle = wrapper.model.register_forward_hook(lambda _m, _a, out: logits.append(out.logits[:, -1].cpu()))
            stack.callback(handle.remove)
            tokens, _, _ = common.generate_group(wrapper, span, group, nouns, 'sadi', 16)
        if read_semantics:
            assert len(reader.semantics) == 36
        return tokens, logits

    with torch.inference_mode():
        before, after = run(False), run(True)
        assert before[0] == after[0]
        assert all(torch.equal(a, b) for a, b in zip(before[1], after[1], strict=True))
    return dict(passed=True, images=[485532, 558955], steps=len(before[1]),
                exact_native_sadi_logits_and_tokens=True, semantic_targets=len(semantic_tokens))


def worker(shard):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_image_semantic_cache import ImageSemanticCache
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
    from models import build_model
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, MODEL, MODEL_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config

    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL), hf_name=str(MODEL_ROOT))
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    prior = fresh.read(PILOT / 'protocol.json')
    probe = FullIElProbe(prior['probe']).to(wrapper.device).eval()
    semantic_tokens = sorted(fresh.read(PILOT / 'training_vocabulary.json')['token_ids'])
    allowed = frozenset(fresh.read(PILOT / 'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    nouns = lexical_nouns(wrapper.tokenizer, word_start='Ġ')
    native_generate = common.generate_group
    cache = None
    if shard == 0:
        fresh.write(RUN / 'semantic_preflight.json', preflight(wrapper, nouns, semantic_tokens))

    def generate(model_wrapper, span, group, noun_ids, method, cap):
        nonlocal cache
        ids = [image for image, _ in group]
        if method == 'sadi':
            cache = None
            with ImageSemanticCache(wrapper.model, span, semantic_tokens).capture() as reader:
                result = native_generate(model_wrapper, span, group, noun_ids, method, cap)
            values = torch.stack([reader.semantics[i] for i in range(36)]).cpu()
            cache = dict(ids=ids, span=tuple(span), E_l=values)
            key = fresh.sha(str((ids, tuple(span))).encode())[:20]
            np.savez_compressed(RUN / 'semantics' / f'shard{shard}.{key}.npz', image_ids=ids,
                                span=span, token_ids=semantic_tokens, E_l=values.numpy())
            return result
        if method == 'svar':
            return native_generate(model_wrapper, span, group, noun_ids, method, cap)
        assert method == NAME and cache is not None
        assert cache['ids'] == ids and cache['span'] == tuple(span)

        def factory(model, current_span, _nouns, mode, strength):
            assert mode == 'next' and strength == 5.
            return Qwen3BatchProbeFirstClass(model, current_span, allowed, probe=probe,
                semantic_tokens=semantic_tokens, cached_semantics=cache['E_l'], guided=True,
                tokenizer=wrapper.tokenizer, history_parser=parser)

        with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
            return original.generate_group(model_wrapper, span, group, noun_ids,
                                           'sadi_sar_ems_next_19_20', cap)

    ids = fresh.read(RUN / 'cohort.json')['coco500'][shard::2]
    files = {name: RUN / 'coco500' / f'{name}.shard{shard}.jsonl' for name in fresh.VARIANTS}
    assert not any(path.exists() for path in files.values()), 'Inspect existing output; no partial resume'
    (RUN / 'semantics').mkdir(exist_ok=True)
    existing = {name: set() for name in fresh.VARIANTS}
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
