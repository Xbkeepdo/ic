"""Check layer19 attention replay, rejection and every current tail KV entry."""

from contextlib import ExitStack

import torch

from features.qwen3_probe_early_recycle import Qwen3BatchProbeEarlyRecycle
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper, probe, allowed):
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16))
    assert len(group) == 2
    inputs = {k: torch.cat([data[k] for _, data in group]).to(wrapper.device) for k in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    cached = None

    def run(controller):
        nonlocal cached
        logits, tail = [], []

        def observe(_module, _args, output):
            logits.append(output.logits[:, -1].cpu())
            cache = output.past_key_values
            assert [cache.get_seq_length(i) for i in range(36)] == [prompt + len(logits) - 1] * 36
            tail.append(tuple(tensor[:, :, -1:].cpu().clone() for i in range(20, 36)
                              for tensor in (cache.layers[i].keys, cache.layers[i].values)))

        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            handle = wrapper.model.register_forward_hook(observe)
            stack.callback(handle.remove)
            result = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        tokens = [trim_eos(row, eos) for row in result[:, prompt:].tolist()]
        if hasattr(controller, 'reader') and len(controller.reader.semantics) == 36:
            cached = torch.stack([controller.reader.semantics[i] for i in range(36)]).cpu()
        return tokens, logits, tail, controller.statistics(tokens)

    with torch.inference_mode():
        baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
        report = dict(passed=True, cases={}, checked_current_KV_layers=list(range(20, 36)))
        for name, options in (
            ('disabled', dict(disable_guidance=True)),
            ('zero', dict(diagnostic_zero_replay=True)),
            ('reject', dict(diagnostic_reject_all=True)),
            ('live', {}),
        ):
            controller = Qwen3BatchProbeEarlyRecycle(wrapper.model, span, allowed, probe=probe,
                semantic_tokens=sorted(allowed), cached_semantics=cached, guided=True,
                diagnostic_force_probe_gate=True, **options)
            current = run(controller)
            if name != 'live':
                assert current[0] == baseline[0] and len(current[1]) == len(baseline[1])
                assert all(torch.equal(a, b) for a, b in zip(current[1], baseline[1], strict=True))
                assert all(torch.equal(a, b) for old, new in zip(baseline[2], current[2], strict=True)
                           for a, b in zip(old, new, strict=True))
            stats = current[3]
            proposals = sum(s.get('guided_attention_calls', 0) for s in stats)
            if name in ('reject', 'live'):
                assert proposals > 0
                assert max(s['maximum_probe_hidden_difference'] for s in stats) == 0
            report['cases'][name] = dict(proposals=proposals,
                accepted=sum(s.get('accepted_guidance_calls', 0) for s in stats),
                maximum_probe_hidden_difference=max(s.get('maximum_probe_hidden_difference', 0) for s in stats),
                exact_native_logits_tokens_and_current_tail_KV=name != 'live',
                diagnostic_force_probe_gate=True)
    return report
