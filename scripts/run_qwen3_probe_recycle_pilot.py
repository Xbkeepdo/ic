"""Test frozen probe gating with matched uniform and evidence-guided recycling."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_history_route_pilot as driver
from scripts import run_qwen3_signed_feedback as shared

OUT = shared.OUT / (shared.PREFIX + 'signed_feedback_head-gradient-probe-recycle-pilot-v2')
TRACE = shared.OUT / 'qwen3_sadi_pre_attention_i_el_trace'
UNIFORM = 'sadi_probe_uniform_recycle_31'
GUIDED = 'sadi_probe_spatial_recycle_31'
PREPARE = driver.prepare


def prepare():
    from scripts.verify_qwen3_recycle_normalization import verify
    normalization_check = verify()
    assert shared.read(TRACE / 'independent_audit.json')['passed']
    with patch.object(driver, 'OUT', OUT), patch.object(driver, 'FIXED', UNIFORM), \
            patch.object(driver, 'GUIDED', GUIDED):
        plan = PREPARE()
    files = ('features/qwen3_online_i_el.py', 'features/qwen3_pre_sadi_i_el.py',
             'features/qwen3_probe_attention_recycle.py', 'scripts/run_qwen3_probe_recycle_pilot.py',
             'scripts/verify_qwen3_recycle_normalization.py')
    for name in files:
        plan['source_sha256'][str(REPO / name)] = shared.sha(REPO / name)
    trace = shared.read(TRACE / 'protocol.json')
    probe = Path(trace['probe'])
    for path in (TRACE / 'protocol.json', TRACE / 'predictions.npz',
                 probe / 'model.pt', probe / 'input_scaler.npz'):
        plan['source_sha256'][str(path)] = shared.sha(path)
    for path in (TRACE / 'semantics').glob('*.npz'):
        plan['source_sha256'][str(path)] = shared.sha(path)
    plan.update(stage='head-gradient-probe-recycle-pilot-v2', source_trace=str(TRACE), probe=str(probe),
                normalization_preflight=normalization_check,
                variants=['sadi', UNIFORM, GUIDED], candidates=[UNIFORM, GUIDED], copied_controls=['sadi'],
                selection='Frozen prospective 100-image mitigation comparison after probe-transfer diagnosis; same gate and moved mass.')
    plan['recipe'] = dict(
        source_layer=31, intervention_layer=31, target_read_layer=35, base_layers=[5, 18],
        gradient_scope={name: 'source FFN and tail layers 32--35' for name in (UNIFORM, GUIDED)},
        gate='Original I+E_l seed43 p_real<0.5; only training token IDs not generated earlier.',
        readout='I uses local attention before SADI on layers5--18, with actual SADI residuals. E_l reuses frozen image caches.',
        history_source='Recycle positive final-target-effect history entries in heads with negative visual writes.',
        common_fraction='.5 * HellingerSquared(Q_negative, P_ffn)', tau=.03,
        uniform_destination='Uniform distribution across all visual tokens.',
        guided_destination='Per-head negative visual writes times .25+.75 normalized positive(Q_negative-P_ffn), then normalize.',
        conservation='FP32 per-head total attention mass conserved; original prompt entries unchanged.',
        acceptance='Actual final candidate log probability strictly decreases; otherwise restore hidden/current tail KV.',
        diagnostic_force_probe_gate=False,
        normalization='Normalize nonnegative weights by maximum before their sum; skip subnormal-only heads.',
        previous_failed_attempt='head-gradient-probe-recycle-pilot: fixed 1e-12 denominator floor lost attention mass; no completed metrics.',
        scope='I/E_l detector plus visual support and FFN deviation components; not the complete SAM/SAM-V2 formula.')
    shared.write(OUT / 'protocol.json', plan)
    return plan


def preflight(wrapper, probe, allowed):
    import torch
    from features.qwen3_probe_attention_recycle import Qwen3BatchProbeAttentionRecycle
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    span, group = next(same_span_batches(wrapper, [485532, 558955], COCO_ROOT, 16))
    assert len(group) == 2
    inputs = {k: torch.cat([data[k] for _, data in group]).to(wrapper.device) for k in group[0][1]}
    prompt = inputs['input_ids'].shape[1]
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])

    def run(controller):
        logits, tail = [], []

        def observe(_module, _args, output):
            logits.append(output.logits[:, -1].cpu())
            cache = output.past_key_values
            assert [cache.get_seq_length(i) for i in range(36)] == [prompt + len(logits) - 1] * 36
            tail.append(tuple(tensor[:, :, -1:].cpu().clone() for i in range(32, 36)
                              for tensor in (cache.layers[i].keys, cache.layers[i].values)))

        with ExitStack() as stack:
            stack.enter_context(controller.capture())
            handle = wrapper.model.register_forward_hook(observe)
            stack.callback(handle.remove)
            result = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
        tokens = [trim_eos(row, eos) for row in result[:, prompt:].tolist()]
        return tokens, logits, tail, controller.statistics(tokens)

    with torch.inference_mode():
        baseline = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
        report = dict(passed=True, cases={})
        for name, options in (
            ('disabled', dict(disable_guidance=True)),
            ('zero', dict(diagnostic_zero_replay=True)),
            ('reject_uniform', dict(guided=False, diagnostic_reject_all=True)),
            ('reject_guided', dict(guided=True, diagnostic_reject_all=True)),
            ('live_guided', dict(guided=True)),
        ):
            controller = Qwen3BatchProbeAttentionRecycle(wrapper.model, span, allowed, probe=probe,
                semantic_tokens=sorted(allowed), diagnostic_force_probe_gate=True, **options)
            current = run(controller)
            stats = current[3]
            if name != 'live_guided':
                assert current[0] == baseline[0]
                assert len(current[1]) == len(baseline[1])
                assert all(torch.equal(a, b) for a, b in zip(current[1], baseline[1], strict=True))
                assert all(torch.equal(a, b) for old, new in zip(baseline[2], current[2], strict=True)
                           for a, b in zip(old, new, strict=True))
            proposals = sum(s.get('guided_attention_calls', 0) for s in stats)
            if name.startswith('reject') or name == 'live_guided':
                assert proposals > 0
            report['cases'][name] = dict(proposals=proposals,
                accepted=sum(s.get('accepted_guidance_calls', 0) for s in stats),
                maximum_probe_hidden_difference=max(s.get('maximum_probe_hidden_difference', 0) for s in stats),
                exact_native_logits_tokens_and_current_tail_KV=name != 'live_guided',
                diagnostic_force_probe_gate=True)
    return report


def worker(gpu):
    import numpy as np
    import torch
    import models
    from features import qwen3_sar_ems_head_gradient as head
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_probe_attention_recycle import Qwen3BatchProbeAttentionRecycle
    build = models.build_model
    plan = shared.read(OUT / 'protocol.json')
    allowed = frozenset(shared.read(OUT / 'training_vocabulary.json')['token_ids'])
    trace_plan = shared.read(TRACE / 'protocol.json')
    calls, cache, probe = 0, None, None

    def checked_build(*args, **kwargs):
        nonlocal probe
        wrapper = build(*args, **kwargs)
        wrapper.model.eval().requires_grad_(False)
        probe = FullIElProbe(plan['probe']).to(wrapper.device).eval()
        if gpu == 0:
            shared.write(OUT / 'recycle_preflight.json', preflight(wrapper, probe, allowed))
        return wrapper

    def factory(model, span, nouns, *, final_gradient):
        nonlocal calls, cache
        batch = calls // 2
        if calls % 2 == 0:
            with np.load(TRACE / f'semantics/shard{gpu}.batch{batch}.npz', allow_pickle=False) as saved:
                cache = {key: saved[key] for key in saved.files}
            assert cache['image_ids'].tolist() == trace_plan['groups'][gpu][batch]['ids']
        assert cache['span'].tolist() == list(span)
        calls += 1
        return Qwen3BatchProbeAttentionRecycle(model, span, allowed, probe=probe,
            semantic_tokens=cache['token_ids'].tolist(), cached_semantics=torch.from_numpy(cache['E_l']),
            guided=final_gradient)

    with patch.object(shared, 'HEAD_FINAL', GUIDED), patch.object(shared, 'HEAD_SOURCE', UNIFORM), \
            patch.object(head, 'Qwen3BatchHeadGradient', factory), patch.object(models, 'build_model', checked_build):
        shared.worker(OUT, gpu)
    assert calls == 2 * len(trace_plan['groups'][gpu])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), \
                patch.object(driver, 'prepare', prepare):
            driver.run()
    else:
        worker(args.worker)
