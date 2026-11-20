"""Small preflight for truncated SADI and six adjacent feedback layers."""

from contextlib import ExitStack
import gc

import torch

from features import qwen3_sar_ems_attention_batch as batch_controller
from features import qwen3_sar_ems_layer_feedback as parent_feedback
from features.qwen3_sadi_expanded_next import (
    Qwen3BatchExpandedNext, Qwen3BatchTruncatedSadi,
    SADI_LAYERS, SOURCE_LAYERS, TARGET_LAYERS,
)
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
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pairs = tuple(zip(SOURCE_LAYERS, TARGET_LAYERS, strict=True))
    assert all(target == source + 1 for source, target in pairs)
    modules = [model]
    for layer in decoder.layers:
        modules.extend((layer, layer.self_attn, layer.post_attention_layernorm, layer.mlp))

    def snapshot():
        return (
            parent_feedback.qwen.eager_attention_forward,
            parent_feedback.ATTENTION_LAYERS,
            batch_controller.ATTENTION_LAYERS,
            [(set(module._forward_hooks), set(module._forward_pre_hooks)) for module in modules],
        )

    def assert_restored(before):
        after = snapshot()
        assert all(a is b for a, b in zip(before[:3], after[:3])), "Attention/layer patch leaked"
        assert before[3] == after[3], "Controller or verification hooks leaked"

    def run(*, baseline=False, strength=5.0, wordset=nouns):
        controller = (Qwen3BatchTruncatedSadi(model, span, wordset) if baseline else
                      Qwen3BatchExpandedNext(model, span, wordset, strength=strength))
        logits, lengths, refreshes, planned = [], [], [], []
        attention_calls = [0] * len(decoder.layers)
        mlp_calls = [0] * len(decoder.layers)
        inactive_row_checks = 0
        before = snapshot()

        if not baseline:
            original_budget = controller._budget

            def budget(source):
                refreshes.append((controller.step, source))
                rows, values, details = original_budget(source)
                planned.extend((controller.step, source, row) for row in rows)
                return rows, values, details

            controller._budget = budget

        def output_hook(_module, _args, output):
            nonlocal inactive_row_checks
            logits.append(output.logits[:, -1].detach().cpu())
            sizes = [int(output.past_key_values.get_seq_length(i)) for i in range(len(decoder.layers))]
            expected = prompt_len + len(logits) - 1
            assert sizes == [expected] * len(decoder.layers), ("KV lengths", sizes, expected)
            lengths.append(expected)
            if not baseline:
                assert not controller.pending and not controller.scratch, "Unconsumed layer state"
                for row, active in enumerate(controller.active.tolist()):
                    if not active:
                        inactive_row_checks += 1
                        assert not any(event["step"] == controller.step
                                       for event in controller.events.get(row, ())), "Guided an EOS-finished row"

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
        if not baseline:
            # Remove the instance wrapper to avoid retaining a GPU controller
            # through the wrapper's closure after this run returns.
            del controller._budget
        assert_restored(before)
        tokens = [trim_eos(row, eos) for row in output[:, prompt_len:].tolist()]
        stats = controller.statistics(tokens)
        assert attention_calls == [len(logits)] * len(decoder.layers), "Duplicate self-attention/KV path"
        assert mlp_calls == [len(logits)] * len(decoder.layers), "Unexpected FFN replay"
        if not baseline:
            assert controller.sources == SOURCE_LAYERS and controller.targets == TARGET_LAYERS
            expected_refreshes = ([(step, source) for step in range(len(logits)) for source in SOURCE_LAYERS]
                                  if controller.enabled else [])
            assert refreshes == expected_refreshes, "Every enabled source must refresh once per step"
            recorded = []
            for row, events in controller.events.items():
                for event in events:
                    assert (event["source_layer"], event["target_layer"]) in pairs
                    assert event["step"] < len(tokens[row]), "Raw guidance event after row EOS"
                    recorded.append((event["step"], event["source_layer"], row))
            assert len(recorded) == len(set(recorded)), "Duplicate guidance at one target/row/step"
            assert sorted(recorded) == sorted(planned), "Fresh source budgets were not consumed exactly once"
            for row in stats:
                assert not row["diagnostic_force_guidance"] and row["local_recompute_events"] == 0
                assert set(row["by_layer"]) == {str(target) for target in TARGET_LAYERS}
                for source, target in pairs:
                    layer = row["by_layer"][str(target)]
                    assert layer["source_layer"] == source and layer["target_layer"] == target
                assert sum(layer["guided_steps"] for layer in row["by_layer"].values()) == row["guided_attention_calls"]
        info = {"steps": len(logits), "cache_lengths": lengths,
                "attention_calls": attention_calls, "mlp_calls": mlp_calls,
                "patches_and_hooks_restored": True, "statistics": stats,
                "budget_refresh_counts": {str(source): sum(s == source for _, s in refreshes)
                                          for source in SOURCE_LAYERS},
                "raw_inactive_row_checks": inactive_row_checks,
                "unfiltered_events_checked": not baseline}
        return tokens, logits, info

    baseline_tokens, baseline_logits, baseline = run(baseline=True)
    result = {"passed": True, "base_variant": "sadi_5_12", "base_layers": list(SADI_LAYERS),
              "source_layers": list(SOURCE_LAYERS), "target_layers": list(TARGET_LAYERS),
              "image_ids": [image_id for image_id, _ in group], "batch_size": 2,
              "max_new_tokens": 24, "baseline": baseline, "controls": {}, "live": {}}
    for label, strength, wordset in (("next_zero_strength", 0.0, nouns),
                                     ("next_empty_nouns", 5.0, frozenset())):
        tokens, logits, info = run(strength=strength, wordset=wordset)
        assert tokens == baseline_tokens, (label, "token mismatch against truncated SADI")
        assert len(logits) == len(baseline_logits) and all(torch.equal(a, b) for a, b in zip(logits, baseline_logits)), (label, "logit mismatch against truncated SADI")
        assert all(stats["guided_attention_calls"] == 0 for stats in info["statistics"])
        info["exact_baseline_logits_and_tokens"] = True
        result["controls"][label] = info
    tokens, logits, info = run()
    assert sum(stats["guided_attention_calls"] for stats in info["statistics"]) > 0, "No natural next intervention: insufficient preflight coverage"
    info["changed_token_sequences"] = sum(a != b for a, b in zip(tokens, baseline_tokens))
    info["source_target_layers_checked"] = True
    result["live"]["next"] = info

    # Natural 24-token runs may contain no EOS. Exercise the active-row state
    # transition explicitly, and report natural inactive-row coverage above.
    probe = Qwen3BatchExpandedNext(model, span, nouns)
    assert probe.eos_ids, "An EOS ID is required for active-row validation"
    non_eos = next(token for token in range(len(probe.eos_ids) + 1) if token not in probe.eos_ids)
    for rows, expected in (([[non_eos], [non_eos]], [True, True]),
                           ([[probe.eos_ids[0]], [non_eos]], [False, True]),
                           ([[non_eos], [non_eos]], [False, True])):
        probe._begin_step(None, (), {"input_ids": torch.tensor(rows, dtype=torch.long)})
        assert probe.active.tolist() == expected, "EOS active-row state did not persist"
    result["synthetic_eos_active_row_transitions"] = 3

    class ExpectedExit(Exception):
        pass

    for controller in (Qwen3BatchTruncatedSadi(model, span, nouns),
                       Qwen3BatchExpandedNext(model, span, nouns)):
        before = snapshot()
        try:
            with controller.capture():
                raise ExpectedExit()
        except ExpectedExit:
            pass
        assert_restored(before)
    result["exception_exit_restores_patches_and_hooks"] = True
    result["scope"] = ("Two previously observed images, four 24-token runs; exact truncated-SADI no-op parity, "
                       "six source refreshes per step, natural next feedback, single KV/FFN execution, "
                       "raw EOS event checks and patch restoration. Natural gates need not activate every target. "
                       "Not a B32 memory benchmark.")
    del baseline_logits, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result
