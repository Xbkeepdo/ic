"""Check exact zero replay and cache consistency through the generation entry point."""

from contextlib import ExitStack

import torch

from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features.qwen3_sar_ems_final_target import Qwen3BatchFinalTargetReplay
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


@torch.inference_mode()
def verify(wrapper, nouns):
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16))
    assert len(group) == 2
    inputs = {k: torch.cat([data[k] for _, data in group]).to(wrapper.device) for k in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    model = wrapper.model
    args = dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def run(controller):
        logits, sizes = [], []
        calls = [0] * len(controller.decoder.layers)

        def output_hook(_module, _args, output):
            logits.append(output.logits[:, -1].cpu())
            lengths = [output.past_key_values.get_seq_length(i) for i in range(len(calls))]
            assert lengths == [prompt + len(logits) - 1] * len(calls)
            sizes.append(lengths[0])

        def count(index):
            def hook(*_):
                calls[index] += 1
            return hook

        with ExitStack() as stack:
            h = model.register_forward_hook(output_hook)
            stack.callback(h.remove)
            for i, layer in enumerate(controller.decoder.layers):
                h = layer.self_attn.register_forward_hook(count(i))
                stack.callback(h.remove)
            stack.enter_context(controller.capture())
            output = model.generate(**inputs, **args)
        tokens = [trim_eos(row, eos) for row in output[:, prompt:].tolist()]
        return tokens, logits, dict(steps=len(logits), cache_lengths=sizes,
                                   attention_calls=calls, statistics=controller.statistics(tokens))

    baseline = Qwen3BatchLateAttention(model, span, nouns, 'sadi')
    expected, logits, base = run(baseline)
    report = dict(image_ids=[i for i, _ in group], max_new_tokens=32, baseline=base, controls={})
    for name, options in (('disabled', dict(disable_guidance=True)),
                          ('zero_replay', dict(diagnostic_zero_replay=True))):
        controller = Qwen3BatchFinalTargetReplay(model, span, nouns, **options)
        tokens, current, info = run(controller)
        assert tokens == expected
        assert len(current) == len(logits) and all(torch.equal(a, b) for a, b in zip(current, logits, strict=True))
        expected_calls = [info['steps']] * len(info['attention_calls'])
        if name == 'zero_replay':
            expected_calls[32:] = [2 * info['steps'] - 1] * (len(expected_calls) - 32)
        assert info['attention_calls'] == expected_calls
        info['exact_baseline_logits_and_tokens'] = True
        report['controls'][name] = info
    controller = Qwen3BatchFinalTargetReplay(model, span, nouns)
    tokens, _, info = run(controller)
    assert sum(s['guided_attention_calls'] for s in info['statistics']) > 0
    assert all(c == info['steps'] for c in info['attention_calls'][:32])
    info['changed_token_sequences'] = sum(a != b for a, b in zip(tokens, expected, strict=True))
    report.update(live=info, passed=True)
    return report
