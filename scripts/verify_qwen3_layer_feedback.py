"""Small GPU checks on previously observed images, before the new cohort runs."""
from contextlib import ExitStack
import gc

import torch

from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


@torch.inference_mode()
def verify(wrapper, nouns):
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16))
    assert len(group) == 2
    inputs = {key: torch.cat([data[key] for _, data in group]).to(wrapper.device) for key in group[0][1]}
    prompt_len = inputs['input_ids'].shape[1]
    model = wrapper.model
    decoder = model.model.language_model
    args = _generation_kwargs(wrapper, MODEL)
    args['max_new_tokens'] = 24
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def run(mode=None, strength=5.0, wordset=nouns):
        controller = (Qwen3BatchLateAttention(model, span, wordset, 'sadi') if mode is None
                      else Qwen3BatchLayerFeedback(model, span, wordset, mode=mode, strength=strength))
        logits, lengths = [], []
        attn_calls = [0] * len(decoder.layers)
        mlp_calls = [0] * len(decoder.layers)
        raw = {}
        retained_checks = 0
        def output_hook(_module, _args, output):
            logits.append(output.logits[:, -1].detach().cpu())
            sizes = [int(output.past_key_values.get_seq_length(i)) for i in range(len(decoder.layers))]
            assert sizes == [prompt_len + len(logits) - 1] * len(decoder.layers), sizes
            lengths.append(sizes[0])
        def before(layer):
            def hook(_module, _args, output):
                raw[layer] = output[:, :-1].clone()
            return hook
        def after(layer):
            def hook(_module, _args, output):
                nonlocal retained_checks
                assert torch.equal(raw[layer], output[:, :-1]), f'Non-current positions changed in {layer}'
                retained_checks += 1
            return hook
        def count(array, index):
            def hook(*_): array[index] += 1
            return hook
        with ExitStack() as stack:
            handle = model.register_forward_hook(output_hook)
            stack.callback(handle.remove)
            for i, layer in enumerate(decoder.layers):
                for module, counter in ((layer.self_attn, attn_calls), (layer.mlp, mlp_calls)):
                    h = module.register_forward_hook(count(counter, i))
                    stack.callback(h.remove)
            for i in (19, 20):
                h = decoder.layers[i].register_forward_hook(before(i))
                stack.callback(h.remove)
            stack.enter_context(controller.capture())
            for i in (19, 20):
                h = decoder.layers[i].register_forward_hook(after(i))
                stack.callback(h.remove)
            output = model.generate(**inputs, **args)
        tokens = [trim_eos(row, eos) for row in output[:, prompt_len:].tolist()]
        stats = controller.statistics(tokens)
        assert attn_calls == [len(logits)] * len(decoder.layers), 'A layer attention/KV path ran twice'
        if mode != 'same': assert mlp_calls == [len(logits)] * len(decoder.layers)
        else:
            assert all(n == len(logits) for i, n in enumerate(mlp_calls) if i not in (19, 20))
        result = {'steps': len(logits), 'cache_lengths': lengths, 'attention_calls': attn_calls,
                  'mlp_calls_19_20': [mlp_calls[19], mlp_calls[20]],
                  'noncurrent_position_checks': retained_checks, 'statistics': stats}
        return tokens, logits, result

    base_tokens, base_logits, baseline = run()
    results = {'passed': True, 'image_ids': [i for i, _ in group], 'batch_size': len(group),
               'max_new_tokens': 24, 'baseline': baseline, 'controls': {}, 'live': {}}
    for mode in ('next', 'same'):
        for label, strength, wordset in (('zero_strength', 0.0, nouns), ('empty_nouns', 5.0, frozenset())):
            tokens, logits, info = run(mode, strength, wordset)
            assert tokens == base_tokens, (mode, label, 'token mismatch')
            assert len(logits) == len(base_logits) and all(torch.equal(a, b) for a, b in zip(logits, base_logits)), (mode, label, 'logit mismatch')
            info['exact_baseline_logits_and_tokens'] = True
            results['controls'][mode + '_' + label] = info
        tokens, logits, info = run(mode)
        # Normal gates should exercise at least one intervention on these old images.
        calls = sum(item.get('guided_attention_calls', 0) for item in info['statistics'])
        assert calls > 0, f'{mode}: no live intervention exercised'
        if mode == 'same':
            assert any(n > info['steps'] for n in info['mlp_calls_19_20']), 'No same-layer FFN replay'
        info['changed_token_sequences'] = sum(a != b for a, b in zip(tokens, base_tokens))
        results['live'][mode] = info
        del logits
    del base_logits
    gc.collect()
    torch.cuda.empty_cache()
    return results
