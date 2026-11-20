"""Change-specific GPU preflight for the new SVAR adjacent-layer adapter."""
from contextlib import ExitStack
import gc

import torch

from features import qwen3_sar_ems_layer_feedback as parent_feedback
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features.qwen3_svar_layer_feedback import Qwen3BatchSvarLayerFeedback
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


@torch.inference_mode()
def verify(wrapper, nouns):
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 32))
    assert len(group) == 2
    inputs = {key: torch.cat([item[key] for _, item in group]).to(wrapper.device)
              for key in group[0][1]}
    prompt_len = inputs["input_ids"].shape[1]
    model = wrapper.model
    decoder = model.model.language_model
    args = _generation_kwargs(wrapper, MODEL)
    args["max_new_tokens"] = 24
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def run(*, baseline=False, strength=5.0, wordset=nouns):
        controller = (Qwen3BatchLateAttention(model, span, wordset, "svar") if baseline else
                      Qwen3BatchSvarLayerFeedback(model, span, wordset, strength=strength))
        logits, lengths = [], []
        attention_calls = [0] * len(decoder.layers)
        mlp_calls = [0] * len(decoder.layers)
        base_operator_before = parent_feedback.sadi_visual_logits

        def output_hook(_module, _args, output):
            logits.append(output.logits[:, -1].detach().cpu())
            sizes = [int(output.past_key_values.get_seq_length(i)) for i in range(len(decoder.layers))]
            expected = prompt_len + len(logits) - 1
            assert sizes == [expected] * len(decoder.layers), ("KV lengths", sizes, expected)
            lengths.append(expected)

        def count(counters, index):
            def hook(*_):
                counters[index] += 1
            return hook

        with ExitStack() as stack:
            handle = model.register_forward_hook(output_hook)
            stack.callback(handle.remove)
            for index, layer in enumerate(decoder.layers):
                for module, counters in ((layer.self_attn, attention_calls), (layer.mlp, mlp_calls)):
                    handle = module.register_forward_hook(count(counters, index))
                    stack.callback(handle.remove)
            stack.enter_context(controller.capture())
            output = model.generate(**inputs, **args)
        assert parent_feedback.sadi_visual_logits is base_operator_before, "Base operator patch leaked"
        tokens = [trim_eos(row, eos) for row in output[:, prompt_len:].tolist()]
        stats = controller.statistics(tokens)
        assert attention_calls == [len(logits)] * len(decoder.layers), "Duplicate self-attention/KV path"
        assert mlp_calls == [len(logits)] * len(decoder.layers), "Unexpected same-layer FFN replay"
        if not baseline:
            assert controller.sources == (18, 19) and controller.targets == (19, 20)
            for row in stats:
                assert not row["diagnostic_force_guidance"] and row["local_recompute_events"] == 0
                assert set(row["by_layer"]) == {"19", "20"}
                for target in (19, 20):
                    layer = row["by_layer"][str(target)]
                    assert layer["source_layer"] == target - 1 and layer["target_layer"] == target
                assert sum(layer["guided_steps"] for layer in row["by_layer"].values()) == row["guided_attention_calls"]
            for events in controller.events.values():
                assert all((event["source_layer"], event["target_layer"]) in ((18, 19), (19, 20)) for event in events)
        result = {"steps": len(logits), "cache_lengths": lengths,
                  "attention_calls": attention_calls, "mlp_calls": mlp_calls,
                  "base_operator_restored": True, "statistics": stats}
        return tokens, logits, result

    baseline_tokens, baseline_logits, baseline = run(baseline=True)
    result = {"passed": True, "base_variant": "svar", "base_layers": list(range(5, 19)),
              "source_layers": [18, 19], "target_layers": [19, 20],
              "image_ids": [image_id for image_id, _ in group], "batch_size": 2,
              "max_new_tokens": 24, "baseline": baseline, "controls": {}, "live": {}}
    for label, strength, wordset in (("next_zero_strength", 0.0, nouns),
                                     ("next_empty_nouns", 5.0, frozenset())):
        tokens, logits, info = run(strength=strength, wordset=wordset)
        assert tokens == baseline_tokens, (label, "token mismatch against original SVAR")
        assert len(logits) == len(baseline_logits) and all(torch.equal(a, b) for a, b in zip(logits, baseline_logits)), (label, "logit mismatch against original SVAR")
        assert all(stats["guided_attention_calls"] == 0 for stats in info["statistics"])
        info["exact_baseline_logits_and_tokens"] = True
        result["controls"][label] = info
    tokens, logits, info = run()
    assert sum(stats["guided_attention_calls"] for stats in info["statistics"]) > 0, "No natural next-layer intervention: preflight coverage insufficient"
    info["changed_token_sequences"] = sum(a != b for a, b in zip(tokens, baseline_tokens))
    info["source_target_layers_checked"] = True
    result["live"]["next"] = info
    result["scope"] = ("Two previously observed images, four 24-token runs; exact no-op SVAR parity, "
                       "natural next-layer feedback, KV/FFN single execution and patch restoration. "
                       "No forced guidance or same-layer variant; not a B32 memory benchmark.")
    del baseline_logits, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result
