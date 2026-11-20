"""Generation-entry no-op, rollback, cache reuse and operation-count checks."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from features.coco_object_history import ObjectClassHistory
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features.qwen3_semantic_final_feedback import Qwen3BatchSemanticFinal
from features.llava_sar_ems_attention import visual_write_norms
from scripts.run_qwen3_semantic_final_pilot import build_wrapper, SOURCE, FILES
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from models.dgst_capture import apply_decoder_final_norm


def verify(wrapper, allowed):
    (span, group), = same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16)
    inputs = {k: torch.cat([data[k] for _, data in group]).to(wrapper.device) for k in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')

    def run(controller):
        logits, tails = [], []
        counts = dict(model=0, head=0, backward=0, ffn31=0, encoder=0)
        def count(field):
            def hook(*_):
                counts[field] += 1
            return hook
        def backward(*args, **kwargs):
            counts['backward'] += 1
            raise AssertionError('This method must not use backward')
        def observe(_module, _args, value):
            logits.append(value.logits[:, -1].cpu().clone())
            kv = value.past_key_values
            assert [kv.get_seq_length(i) for i in range(36)] == [prompt+len(logits)-1]*36
            tails.append(tuple(t[:, :, -1:].cpu().clone() for i in range(32, 36)
                for t in (kv.layers[i].keys, kv.layers[i].values)))
        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            stack.enter_context(patch.object(torch.autograd, 'grad', backward))
            for module, field in [(wrapper.model, 'model'), (wrapper.model.get_output_embeddings(), 'head'),
                    (wrapper.model.model.language_model.layers[31].mlp, 'ffn31'),
                    (wrapper.model.model.visual, 'encoder')]:
                stack.callback(module.register_forward_pre_hook(count(field)).remove)
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            result = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        tokens = [trim_eos(row, eos) for row in result[:, prompt:].tolist()]
        return tokens, logits, tails, counts, controller.statistics(tokens)

    baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
    cache = {}
    report = dict(passed=True, base_layers_inclusive=[5, 18], action_layer=31,
                  image_ids=[i for i, _ in group], max_new_tokens=32, cases={})
    for name, options in [('disabled', dict(disable_guidance=True)),
            ('zero', dict(diagnostic_zero_replay=True)),
            ('reject', dict(diagnostic_reject_all=True, diagnostic_force_guidance=True)),
            ('live_sar', {}), ('live_increment', dict(semantic_mode='semantic_increment')),
            ('forced_sar', dict(diagnostic_force_guidance=True)),
            ('forced_increment', dict(semantic_mode='semantic_increment', diagnostic_force_guidance=True)),
            ('permuted', dict(permute=True, diagnostic_force_guidance=True))]:
        controller = Qwen3BatchSemanticFinal(wrapper.model, span, allowed, tokenizer=wrapper.tokenizer,
            history_parser=parser, static_cache=cache, **options)
        current = run(controller)
        exact = current[0] == baseline[0] and len(current[1]) == len(baseline[1]) and all(
            torch.equal(a, b) for a, b in zip(current[1], baseline[1], strict=True))
        exact_kv = len(current[2]) == len(baseline[2]) and all(torch.equal(a, b)
            for before, after in zip(current[2], baseline[2], strict=True)
            for a, b in zip(before, after, strict=True))
        if name in ('disabled', 'zero', 'reject'):
            assert exact and exact_kv, name
        if name in ('reject', 'forced_sar', 'forced_increment', 'permuted'):
            assert sum(s['guided_attention_calls'] for s in current[4]) > 0, name
        counts = current[3]
        assert counts['model'] == len(current[1]) and counts['backward'] == 0
        assert counts['encoder'] == baseline[3]['encoder'] == 1
        assert counts['head'] == counts['model']+controller.replay_steps+controller.semantic_head_calls
        assert counts['ffn31'] == counts['model']+controller.replay_steps
        for row, (tokens, stat) in enumerate(zip(current[0], current[4], strict=True)):
            events = stat['semantic_events']
            assert len(events) == stat['guided_attention_calls'] == stat['local_recompute_events']
            if not events:
                assert tokens == baseline[0][row]
            else:
                first = events[0]['step']
                assert tokens[:first] == baseline[0][row][:first]
                assert all(torch.equal(current[1][step][row], baseline[1][step][row]) for step in range(first))
        report['cases'][name] = dict(exact_native_logits_tokens=exact, exact_current_tail_KV=exact_kv,
            counts=counts, events=sum(s['guided_attention_calls'] for s in current[4]),
            cache_builds=controller.cache_builds, cache_reuses=controller.cache_reuses,
            changed_sequences=sum(a != b for a, b in zip(current[0], baseline[0], strict=True)))
    assert cache, 'Live semantic cache was never built'
    errors = []
    for row, terms in cache.items():
        # Independent full semantic formula, retaining no logits after this check.
        normalized = apply_decoder_final_norm(wrapper.model.model.language_model.norm, terms.hpre)
        reference = []
        for begin in range(0, len(normalized), 64):
            logits = wrapper.model.get_output_embeddings()(normalized[begin:begin+64]).float()
            log_z = logits.logsumexp(-1, keepdim=True)
            reference.append(torch.log1p((logits[:, list(terms.tokens)]-log_z).exp()*logits.shape[-1]))
        semantic = torch.cat(reference)
        torch.testing.assert_close(terms.log_s, semantic, rtol=2e-6, atol=2e-6)
        attention = torch.linspace(.001, .009, terms.values.shape[1], device=terms.values.device)
        attention = attention.expand(terms.values.shape[:2])
        weight = wrapper.model.model.language_model.layers[31].self_attn.o_proj.weight.float()
        direct = torch.stack([weight @ (terms.values[:, j].float()*attention[:, j, None]).flatten()
                              for j in range(terms.values.shape[1])]).norm(dim=-1)
        actual = visual_write_norms(terms.gram, attention)
        torch.testing.assert_close(actual, direct, rtol=2e-5, atol=1e-6)
        errors.append(dict(row=row, semantic_max_error=float((terms.log_s-semantic).abs().max()),
                           write_norm_max_error=float((actual-direct).abs().max())))
    report['static_cache_formula_checks'] = errors
    report['source_sha256'] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FILES}
    return report


if __name__ == '__main__':
    out = REPO / 'outputs/qwen3_semantic_final_preflight'
    out.mkdir(exist_ok=True)
    try:
        wrapper = build_wrapper()
        with torch.inference_mode():
            allowed = frozenset(json.loads((SOURCE / 'object_lexicon.json').read_bytes())['token_ids'])
            report = verify(wrapper, allowed)
        (out / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')
        (out / 'status.json').write_text(json.dumps(dict(status='passed'))+'\n')
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        (out / 'status.json').write_text(json.dumps(dict(status='failed', error=str(exc)))+'\n')
        raise
