"""Check native features, exact no-ops, tail rollback and shared semantics."""

from contextlib import ExitStack
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from features.qwen3_full_risk_f import FullOnlineRiskF
from features.qwen3_native_risk_i_el import NativeRiskIEl
from features.qwen3_semantic_donor_feedback import Qwen3BatchSemanticDonor
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs


def verify(wrapper, probe, allowed, parser, threshold):
    (span, group), = same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16)
    inputs = {k: torch.cat([d[k] for _, d in group]).to(wrapper.device) for k in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    checks = []

    def run(controller, *, features=False):
        logits, kv, counts = [], [], dict(encoder=0)
        reference = FullOnlineRiskF(wrapper.model, span, allowed) if features else None
        reader = NativeRiskIEl(wrapper.model, span, allowed) if features else None
        def observe(_module, _args, out):
            logits.append(out.logits[:, -1].cpu().clone())
            cache = out.past_key_values
            assert all(cache.get_seq_length(i) == prompt+len(logits)-1 for i in range(36))
            kv.append(tuple(t[:, :, -1:].cpu().clone() for layer in cache.layers
                            for t in (layer.keys, layer.values)))
            if features and reader.step == 1:
                targets = [wrapper.tokenizer.encode(' person', add_special_tokens=False)[0],
                           wrapper.tokenizer.encode(' dog', add_special_tokens=False)[0]]
                result = reader.features([0, 1], targets).cpu().numpy()
                previous = reference.full_features([0, 1], targets)
                np.testing.assert_array_equal(result[:, :36], previous['native_risk'])
                np.testing.assert_array_equal(result[:, 36:72], previous['native_I'])
                np.testing.assert_allclose(result[:, 72:], previous['E_l_from_log_s'], rtol=3e-6, atol=1e-9)
                old = probe(torch.from_numpy(np.concatenate([previous['native_risk'],
                    previous['native_I'], previous['E_l_from_log_s']], axis=1)).to(wrapper.device))
                new = probe(torch.from_numpy(result).to(wrapper.device))
                assert (old-new).abs().max() < 1e-5
                checks.append(dict(risk_I_exact=True, E_l_relative_tolerance=3e-6,
                                   maximum_probability_difference=float((old-new).abs().max())))
        def encoded(*_):
            counts['encoder'] += 1
        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            if features:
                stack.enter_context(reference.capture())
                stack.enter_context(reader.capture())
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            stack.callback(wrapper.model.model.visual.register_forward_pre_hook(encoded).remove)
            tokens = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        assert counts['encoder'] == 1
        raw = tokens[:, prompt:].tolist()
        return tokens.cpu(), logits, kv, controller.statistics(raw)

    report = dict(passed=True, cases={}, full_batch=2, tokens=32, current_KV_layers=list(range(36)))
    shared = {}
    with torch.inference_mode():
        baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
        checked = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'), features=True)
        for name, mode, options in [
            ('features_only', None, None),
            ('disabled', 'semantic', dict(disable_guidance=True)),
            ('zero', 'semantic', dict(diagnostic_zero_replay=True)),
            ('reject_history', 'history', dict(diagnostic_reject_all=True)),
            ('reject_semantic', 'semantic', dict(diagnostic_reject_all=True)),
            ('live_history', 'history', {}),
            ('live_semantic', 'semantic', {}),
        ]:
            controller = None if name == 'features_only' else Qwen3BatchSemanticDonor(
                wrapper.model, span, allowed, probe=probe, semantic_tokens=sorted(allowed),
                tokenizer=wrapper.tokenizer, history_parser=parser, real_threshold=threshold,
                transport_mode=mode, shared_static=shared, diagnostic_force_probe_gate=True, **options)
            current = checked if controller is None else run(controller)
            exact = not name.startswith('live_')
            if exact:
                assert torch.equal(current[0], baseline[0]), name
                assert all(torch.equal(a, b) for a, b in zip(current[1], baseline[1], strict=True)), name
                assert all(torch.equal(a, b) for old, new in zip(current[2], baseline[2], strict=True)
                           for a, b in zip(old, new, strict=True)), name
            stats = current[3]
            proposals = sum(s.get('guided_attention_calls', 0) for s in stats)
            if name.startswith(('reject_', 'live_')):
                assert proposals > 0, name
                assert max(s['maximum_probe_hidden_difference'] for s in stats) == 0
            report['cases'][name] = dict(exact_native_logits_tokens_all_current_KV=exact,
                proposals=proposals, accepted=sum(s.get('accepted_guidance_calls', 0) for s in stats),
                encoder_calls=1, shared_hits=max(s.get('shared_static_hits', 0) for s in stats))
        assert report['cases']['reject_semantic']['shared_hits'] > 0
    report['feature_checks'] = checks
    return report
