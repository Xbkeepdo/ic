"""Exact no-op and rejection checks for layer31 attention-output guidance."""

from contextlib import ExitStack
import torch
from features.qwen3_acg_signal_feedback import Qwen3BatchACGSignal
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper, probe, allowed, parser, threshold):
    (span, group), = same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16)
    inputs = {k: torch.cat([d[k] for _, d in group]).to(wrapper.device) for k in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    shared = {}
    def run(controller):
        logits, kv, encoders = [], [], []
        def observe(_module, _args, out):
            logits.append(out.logits[:, -1].cpu().clone())
            cache = out.past_key_values
            assert all(cache.get_seq_length(i) == prompt+len(logits)-1 for i in range(36))
            kv.append(tuple(t[:, :, -1:].cpu().clone() for layer in cache.layers for t in (layer.keys, layer.values)))
        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            stack.callback(wrapper.model.model.visual.register_forward_pre_hook(lambda *_: encoders.append(1)).remove)
            tokens = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        assert len(encoders) == 1
        return tokens.cpu(), logits, kv, controller.statistics(tokens[:, prompt:].tolist())
    report = dict(passed=True, cases={}, current_KV_layers=list(range(36)), full_batch=2, max_tokens=32)
    with torch.inference_mode():
        baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
        for name, guided, options in [
            ('disabled', True, dict(disable_guidance=True)),
            ('zero_gamma', True, dict(gamma=0.)),
            ('zero_replay', True, dict(diagnostic_zero_replay=True)),
            ('reject_fixed', False, dict(diagnostic_reject_all=True)),
            ('reject_guided', True, dict(diagnostic_reject_all=True)),
            ('live_fixed', False, {}), ('live_guided', True, {})]:
            controller = Qwen3BatchACGSignal(wrapper.model, span, allowed, probe=probe,
                semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer, history_parser=parser,
                real_threshold=threshold, shared_static=shared, signal_guided=guided,
                diagnostic_force_probe_gate=True, **options)
            current = run(controller)
            exact = not name.startswith('live_')
            if exact:
                assert torch.equal(current[0], baseline[0]), name
                assert all(torch.equal(a, b) for a, b in zip(current[1], baseline[1], strict=True)), name
                assert all(torch.equal(a, b) for old, new in zip(current[2], baseline[2], strict=True)
                           for a, b in zip(old, new, strict=True)), name
            stats = current[3]
            proposals = sum(s['guided_attention_calls'] for s in stats)
            accepted = sum(s.get('accepted_guidance_calls', 0) for s in stats)
            if name.startswith(('reject_', 'live_')):
                assert proposals > 0
                assert max(s['maximum_probe_hidden_difference'] for s in stats) == 0
            if name == 'live_guided':
                assert accepted > 0
            report['cases'][name] = dict(exact_native_logits_tokens_all_current_KV=exact,
                proposals=proposals, accepted=accepted, encoder_calls=1,
                gradient_calls=max(s.get('gradient_probe_calls', 0) for s in stats),
                shared_hits=max(s['shared_static_hits'] for s in stats))
            del controller
            __import__('gc').collect(); torch.cuda.empty_cache()
    assert report['cases']['reject_guided']['shared_hits'] > 0
    return report
