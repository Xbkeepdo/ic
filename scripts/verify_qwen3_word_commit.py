"""Check word-change rejection against native logits and current tail KV."""

from contextlib import ExitStack

import torch

from features.coco_object_history import ObjectClassHistory
from features.qwen3_probe_word_commit import Qwen3BatchWordChangingCommit
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper, probe, allowed):
    from pathlib import Path
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

        def observe(_module, _args, output):
            logits.append(output.logits[:, -1].cpu())
            cache = output.past_key_values
            assert [cache.get_seq_length(i) for i in range(36)] == [prompt + len(logits) - 1] * 36
            tails.append(tuple(value[:, :, -1:].cpu().clone() for i in range(20, 36)
                               for value in (cache.layers[i].keys, cache.layers[i].values)))

        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            handle = wrapper.model.register_forward_hook(observe)
            stack.callback(handle.remove)
            result = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        tokens = [trim_eos(row, eos) for row in result[:, prompt:].tolist()]
        if hasattr(controller, 'reader') and len(controller.reader.semantics) == 36:
            cached = torch.stack([controller.reader.semantics[i] for i in range(36)]).cpu()
        return tokens, logits, tails, controller.statistics(tokens)

    with torch.inference_mode():
        baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
        report = dict(passed=True, cases={}, checked_current_KV_layers=list(range(20, 36)))
        for name, options in (
            ('disabled', dict(disable_guidance=True)),
            ('zero', dict(diagnostic_zero_replay=True)),
            ('reject', dict(diagnostic_reject_all=True)),
            ('live', {}),
        ):
            controller = Qwen3BatchWordChangingCommit(wrapper.model, span, allowed, probe=probe,
                semantic_tokens=sorted(allowed), cached_semantics=cached, guided=True,
                diagnostic_force_probe_gate=True, tokenizer=wrapper.tokenizer, history_parser=history, **options)
            current = run(controller)
            assert len(current[1]) == len(baseline[1])
            checked_rows = 0
            for row, stats in enumerate(current[3]):
                accepted = [e['step'] for e in stats['recycle_events'] if e['accepted']]
                first = min(accepted, default=len(current[1]))
                for step in range(first):
                    assert torch.equal(current[1][step][row], baseline[1][step][row])
                    assert all(torch.equal(old[row], new[row]) for old, new in
                               zip(baseline[2][step], current[2][step], strict=True))
                    checked_rows += 1
            exact_all = all(torch.equal(a, b) for a, b in zip(baseline[1], current[1], strict=True))
            if name != 'live':
                assert exact_all and current[0] == baseline[0]
                assert all(torch.equal(a, b) for old, new in zip(baseline[2], current[2], strict=True)
                           for a, b in zip(old, new, strict=True))
            proposals = sum(s['guided_attention_calls'] for s in current[3])
            unchanged_rejects = sum(s['rejected_unchanged_token_calls'] for s in current[3])
            if name in ('reject', 'live'):
                assert proposals > 0
            if name == 'live':
                assert unchanged_rejects > 0, 'The new rollback branch must be exercised.'
            report['cases'][name] = dict(proposals=proposals, unchanged_token_rejections=unchanged_rejects,
                accepted=sum(s['accepted_guidance_calls'] for s in current[3]),
                exact_native_logits_tokens_and_current_tail_KV=exact_all,
                native_prefix_rows_with_exact_logits_and_current_KV=checked_rows,
                maximum_probe_hidden_difference=max(s['maximum_probe_hidden_difference'] for s in current[3]),
                diagnostic_force_probe_gate=True)
    return report
