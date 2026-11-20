"""Check complete-SADI parity, rollback and cost of the target-FFN readout."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from features.coco_object_history import ObjectClassHistory
from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
from features.qwen3_target_ffn import Qwen3BatchTargetFFN
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper, probe, allowed):
    (span, group), = same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16)
    inputs = {key: torch.cat([data[key] for _, data in group]).to(wrapper.device) for key in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    parser = ObjectClassHistory(Path(__file__).resolve().parents[1] / 'coco-labeling/coco_chair.py')
    cached = None

    def run(controller):
        nonlocal cached
        logits, tails = [], []
        counts = dict(model=0, source_ffn=0, lm_head=0, backward=0)
        original_grad = torch.autograd.grad

        def gradient(*args, **kwargs):
            counts['backward'] += 1
            return original_grad(*args, **kwargs)

        def count(name):
            def observe(*_args):
                counts[name] += 1
            return observe

        def output(_module, _args, value):
            logits.append(value.logits[:, -1].cpu())
            kv = value.past_key_values
            assert [kv.get_seq_length(i) for i in range(36)] == [prompt + len(logits) - 1] * 36
            tails.append(tuple(t[:, :, -1:].cpu().clone() for i in range(20, 36)
                for t in (kv.layers[i].keys, kv.layers[i].values)))

        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            stack.enter_context(patch.object(torch.autograd, 'grad', gradient))
            for module, field in ((wrapper.model, 'model'),
                    (wrapper.model.model.language_model.layers[19].mlp, 'source_ffn'),
                    (wrapper.model.get_output_embeddings(), 'lm_head')):
                handle = module.register_forward_pre_hook(count(field))
                stack.callback(handle.remove)
            handle = wrapper.model.register_forward_hook(output)
            stack.callback(handle.remove)
            generated = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        tokens = [trim_eos(row, eos) for row in generated[:, prompt:].tolist()]
        if hasattr(controller, 'reader') and len(controller.reader.semantics) == 36:
            cached = torch.stack([controller.reader.semantics[i] for i in range(36)]).cpu()
        return tokens, logits, tails, controller.statistics(tokens), counts

    with torch.inference_mode():
        native = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
        parent = run(Qwen3BatchProbeFirstClass(wrapper.model, span, allowed, probe=probe,
            semantic_tokens=sorted(allowed), guided=True, diagnostic_force_probe_gate=True,
            diagnostic_reject_all=True, tokenizer=wrapper.tokenizer, history_parser=parser))
        assert parent[0] == native[0]
        assert all(torch.equal(a, b) for a, b in zip(parent[1], native[1], strict=True))
        assert all(torch.equal(a, b) for before, after in zip(parent[2], native[2], strict=True)
            for a, b in zip(before, after, strict=True))
        report = dict(passed=True, baseline_variant='sadi', base_layers_inclusive=[5, 18], cases={},
            parent_reject_forward_counts=parent[4], checked_current_KV_layers=list(range(20, 36)))
        for name, options in (('disabled', dict(disable_guidance=True)), ('zero', dict(diagnostic_zero_replay=True)),
                ('reject', dict(diagnostic_reject_all=True)), ('live', {})):
            controller = Qwen3BatchTargetFFN(wrapper.model, span, allowed, probe=probe,
                semantic_tokens=sorted(allowed), cached_semantics=cached, guided=True,
                diagnostic_force_probe_gate=True, tokenizer=wrapper.tokenizer, history_parser=parser, **options)
            current = run(controller)
            assert len(current[1]) == len(native[1])
            exact_rows = 0
            for row, stat in enumerate(current[3]):
                first = min((e['step'] for e in stat['recycle_events'] if e['accepted']), default=len(current[1]))
                for step in range(first):
                    assert torch.equal(current[1][step][row], native[1][step][row])
                    assert all(torch.equal(a[row], b[row]) for a, b in zip(native[2][step], current[2][step], strict=True))
                    exact_rows += 1
                for event in stat['recycle_events']:
                    assert event['accepted'] == (event['actual_log_probability_change'] < 0 and name != 'reject')
                    assert event['target_conditioned_ffn_preference']
                    assert event['ffn_gradient_scope'] == 'source_FFN_output_to_final_log_probability'
                    assert event['predicted_log_probability_change'] <= 1e-6
                assert stat['ffn_output_gradient_reads'] == stat['target_ffn_preference_reads'] == current[4]['backward']
            exact = all(torch.equal(a, b) for a, b in zip(current[1], native[1], strict=True))
            if name != 'live':
                assert exact and current[0] == native[0]
                assert all(torch.equal(a, b) for before, after in zip(native[2], current[2], strict=True)
                    for a, b in zip(before, after, strict=True))
            proposals = sum(s['guided_attention_calls'] for s in current[3])
            if name in ('reject', 'live'):
                assert proposals > 0
            if name == 'reject':
                assert current[4] == parent[4], 'A second NN call or backward was introduced.'
            report['cases'][name] = dict(proposals=proposals,
                accepted=sum(s['accepted_guidance_calls'] for s in current[3]), forward_counts=current[4],
                exact_native_logits_tokens_and_current_tail_KV=exact,
                native_prefix_rows_with_exact_logits_and_current_KV=exact_rows,
                maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in current[3]),
                diagnostic_force_probe_gate=True)
    return report
