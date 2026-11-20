"""Exercise the new acceptance rollback against native SVAR logits and current KV."""

from contextlib import ExitStack
from pathlib import Path

import torch

from features.coco_object_history import ObjectClassHistory
from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass
from features.qwen3_svar_ffn_alignment import Qwen3BatchSvarFFNAlignment
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper, probe, allowed):
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16))
    assert len(group) == 2
    inputs = {key: torch.cat([data[key] for _, data in group]).to(wrapper.device) for key in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    history = ObjectClassHistory(Path(__file__).resolve().parents[1] / 'coco-labeling/coco_chair.py')
    cached = None

    def run(controller):
        nonlocal cached
        logits, tails = [], []
        calls = dict(model=0, source_ffn=0, lm_head=0)

        def count(name):
            def observe_call(*_args):
                calls[name] += 1
            return observe_call

        def observe(_module, _args, output):
            logits.append(output.logits[:, -1].cpu())
            cache = output.past_key_values
            assert [cache.get_seq_length(i) for i in range(36)] == [prompt + len(logits) - 1] * 36
            tails.append(tuple(value[:, :, -1:].cpu().clone() for i in range(20, 36)
                               for value in (cache.layers[i].keys, cache.layers[i].values)))

        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            for watched, field in ((wrapper.model, 'model'),
                    (wrapper.model.model.language_model.layers[19].mlp, 'source_ffn'),
                    (wrapper.model.get_output_embeddings(), 'lm_head')):
                counter = watched.register_forward_pre_hook(count(field))
                stack.callback(counter.remove)
            handle = wrapper.model.register_forward_hook(observe)
            stack.callback(handle.remove)
            generated = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        tokens = [trim_eos(row, eos) for row in generated[:, prompt:].tolist()]
        if hasattr(controller, 'reader') and len(controller.reader.semantics) == 36:
            cached = torch.stack([controller.reader.semantics[i] for i in range(36)]).cpu()
        return tokens, logits, tails, controller.statistics(tokens), calls

    with torch.inference_mode():
        baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'svar'))
        parent = run(Qwen3BatchSvarFirstClass(wrapper.model, span, allowed, probe=probe,
            semantic_tokens=sorted(allowed), guided=True, diagnostic_force_probe_gate=True,
            diagnostic_reject_all=True, tokenizer=wrapper.tokenizer, history_parser=history))
        assert parent[0] == baseline[0]
        assert all(torch.equal(a, b) for a, b in zip(parent[1], baseline[1], strict=True))
        assert all(torch.equal(a, b) for old, new in zip(parent[2], baseline[2], strict=True)
                   for a, b in zip(old, new, strict=True))
        report = dict(passed=True, baseline_variant='svar', cases={},
            parent_reject_forward_counts=parent[4], checked_current_KV_layers=list(range(20, 36)))
        for name, options in (
            ('disabled', dict(disable_guidance=True)),
            ('zero', dict(diagnostic_zero_replay=True)),
            ('reject', dict(diagnostic_reject_all=True)),
            ('feedback_reject', dict(diagnostic_reject_ffn_alignment=True)),
            ('live', {}),
        ):
            controller = Qwen3BatchSvarFFNAlignment(wrapper.model, span, allowed, probe=probe,
                semantic_tokens=sorted(allowed), cached_semantics=cached, guided=True,
                diagnostic_force_probe_gate=True, tokenizer=wrapper.tokenizer, history_parser=history, **options)
            current = run(controller)
            assert len(current[1]) == len(baseline[1])
            exact_rows = 0
            for row, stats in enumerate(current[3]):
                accepted = [e['step'] for e in stats['recycle_events'] if e['accepted']]
                first = min(accepted, default=len(current[1]))
                for step in range(first):
                    assert torch.equal(current[1][step][row], baseline[1][step][row])
                    assert all(torch.equal(old[row], new[row]) for old, new in
                               zip(baseline[2][step], current[2][step], strict=True))
                    exact_rows += 1
                if name == 'live':
                    for event in stats['recycle_events']:
                        assert event['accepted'] == (event['actual_log_probability_change'] < 0
                            and event['ffn_drift_after'] < event['drift'])
            exact_all = all(torch.equal(old, new) for old, new in zip(baseline[1], current[1], strict=True))
            if name != 'live':
                assert exact_all and current[0] == baseline[0]
                assert all(torch.equal(a, b) for old, new in zip(baseline[2], current[2], strict=True)
                           for a, b in zip(old, new, strict=True))
            proposals = sum(s['guided_attention_calls'] for s in current[3])
            feedback_rejections = sum(s['rejected_ffn_alignment_calls'] for s in current[3])
            if name in ('reject', 'feedback_reject', 'live'):
                assert proposals > 0
            if name == 'feedback_reject':
                assert feedback_rejections > 0, 'Exercise the new rollback after a parent-accepted proposal.'
            if name in ('reject', 'feedback_reject'):
                assert current[4] == parent[4], 'Feedback must add no model, source-FFN or LM-head forwards.'
            report['cases'][name] = dict(proposals=proposals, feedback_rejections=feedback_rejections,
                accepted=sum(s['accepted_guidance_calls'] for s in current[3]),
                exact_native_logits_tokens_and_current_tail_KV=exact_all,
                native_prefix_rows_with_exact_logits_and_current_KV=exact_rows,
                maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in current[3]),
                forward_counts=current[4],
                diagnostic_force_probe_gate=True)
    return report
