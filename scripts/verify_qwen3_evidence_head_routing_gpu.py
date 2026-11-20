"""Check evidence-head routing, exact uniform biases, native KV and actual source refresh."""

from contextlib import ExitStack
import argparse
import gc
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch

from features.qwen3_bounded_visual_tilt import Qwen3BatchBoundedVisualTilt
from features.qwen3_causal_visual_tilt import Qwen3BatchCausalVisualTilt, TARGETS
from features.qwen3_evidence_head_feedback import Qwen3BatchEvidenceHeadFeedback
from features import qwen3_causal_visual_tilt as causal_operator
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches, trim_eos
from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
from scripts.run_qwen3_signed_feedback import read, write, source_groups, sha


def uniform_head_attention(query,key,value,mask,scaling,groups,span,rows,coefficient,gamma):
    key=qwen.repeat_kv(key,groups);value=qwen.repeat_kv(value,groups)
    scores=(query@key.transpose(2,3))*scaling
    if mask is not None:scores=scores+mask[...,:key.shape[-2]]
    start,end=span;visual=scores[rows,:,0,start:end].float()
    bias=.5*visual.abs().mean(-1)*float(gamma)
    shift=bias[:,:,None,None].expand(-1,-1,1,end-start)
    tilted=scores.float().clone();tilted[rows,:,:,start:end]+=shift
    probability=tilted.softmax(-1,dtype=torch.float32).to(query.dtype)
    return (probability@value).transpose(1,2).contiguous(),probability,shift


class UniformHeadReference(Qwen3BatchCausalVisualTilt):
    def _replay(self,rows,native):
        with patch.object(causal_operator,'current_attention',uniform_head_attention):
            return super()._replay(rows,native)


def verify(wrapper, probe, allowed, parser, threshold, batch_label=None):
    first = source_groups(REPO/'outputs/qwen3_block_acg_pilot_20261118/coco100/sadi.shard0.jsonl')[0]
    report = dict(passed=True, batches={})
    for label, ids in [('B2', [485532, 558955]), ('B16', [row['image_id'] for row in first])]:
        if batch_label is not None and label != batch_label:
            continue
        (span, group), = same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
        inputs = {key: torch.cat([item[key] for _, item in group]).to(wrapper.device) for key in group[0][1]}
        prompt = inputs['input_ids'].shape[1]
        eos = wrapper.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        shared_static = {}

        def run(controller):
            logits, current_kv, features, encoders = [], [], [], []
            attention_trace, update_trace = {}, {}
            previous = None
            past_checks = refreshed_checks = 0

            def begin(*_):
                attention_trace.clear()
                update_trace.clear()

            def observe(_module, _args, output):
                nonlocal previous, past_checks
                logits.append(output.logits[:, -1].cpu().clone())
                cache = output.past_key_values
                assert all(cache.get_seq_length(i) == prompt+len(logits)-1 for i in range(36))
                if previous is not None:
                    for layer, (old_keys, old_values) in zip(cache.layers, previous, strict=True):
                        assert torch.equal(layer.keys[:, :, :-1], old_keys)
                        assert torch.equal(layer.values[:, :, :-1], old_values)
                        past_checks += 2
                current_kv.append(tuple(t[:, :, -1:].cpu().clone()
                    for layer in cache.layers for t in (layer.keys, layer.values)))
                # DynamicCache appends into new tensors on the next step;
                # retain references, not additional full-prefix copies.
                previous = [(layer.keys, layer.values) for layer in cache.layers]

            def remember_attention(index):
                def hook(_module, _args, output):
                    if getattr(controller, '_causal_replaying', False):
                        assert output[1] is not None
                        attention_trace[controller.step, index] = output[1][:, :, 0, span[0]:span[1]].detach()
                return hook

            def remember_update(index):
                def hook(_module, _args, output):
                    if getattr(controller, '_causal_replaying', False):
                        update_trace[controller.step, index] = output[:, 0].detach()
                return hook

            def forbid(*_args, **_kwargs):
                raise AssertionError('Causal visual tilts must not backpropagate')

            torch.cuda.reset_peak_memory_stats(wrapper.device)
            with ExitStack() as stack:
                stack.enter_context(controller.capture())
                stack.enter_context(patch.object(torch.autograd, 'grad', forbid))
                stack.callback(wrapper.model.register_forward_pre_hook(begin).remove)
                stack.callback(wrapper.model.register_forward_hook(observe).remove)
                stack.callback(wrapper.model.model.visual.register_forward_pre_hook(lambda *_: encoders.append(1)).remove)
                stack.callback(probe.register_forward_pre_hook(lambda _module, args:
                    features.append((controller.step, args[0].cpu().clone()))).remove)
                if isinstance(controller, Qwen3BatchCausalVisualTilt):
                    for index in TARGETS[:-1]:
                        layer = controller.decoder.layers[index]
                        stack.callback(layer.self_attn.register_forward_hook(remember_attention(index)).remove)
                        stack.callback(layer.mlp.register_forward_hook(remember_update(index)).remove)
                    source_name='_source_distributions' if isinstance(controller,Qwen3BatchEvidenceHeadFeedback) else '_source_coefficients'
                    source_function=getattr(controller,source_name)

                    def check_source(source, rows, attention=None, update=None):
                        nonlocal refreshed_checks
                        if attention is not None:
                            assert source in TARGETS[:-1]
                            key = (controller.step, source)
                            assert key in attention_trace and key in update_trace
                            assert torch.equal(attention, attention_trace[key])
                            assert torch.equal(update, update_trace[key])
                            refreshed_checks += 1
                        return source_function(source, rows, attention, update)

                    stack.enter_context(patch.object(controller, source_name, check_source))
                output = wrapper.model.generate(**inputs,
                    **dict(_generation_kwargs(wrapper, MODEL), max_new_tokens=32))
            tokens = [trim_eos(row, eos) for row in output[:, prompt:].tolist()]
            assert 1 <= len(logits) <= 32 and len(encoders) == 1
            stats = controller.statistics(tokens)
            peak = torch.cuda.max_memory_allocated(wrapper.device)
            previous = None
            return tokens, logits, current_kv, stats, features, peak, past_checks, refreshed_checks

        def exact(current, native):
            assert current[0] == native[0]
            assert all(torch.equal(a, b) for a, b in zip(current[1], native[1], strict=True))
            assert all(torch.equal(a, b) for aa, bb in zip(current[2], native[2], strict=True)
                       for a, b in zip(aa, bb, strict=True))

        def exact_gate(current, reference):
            assert len(current[4]) == len(reference[4])
            for (step, vector), (ref_step, ref_vector) in zip(current[4], reference[4], strict=True):
                assert step == ref_step and vector.shape[-1] == 108 and torch.equal(vector, ref_vector)
            for current_row, ref_row in zip(current[3], reference[3], strict=True):
                assert current_row['full_class_decisions'] == ref_row['full_class_decisions']
                assert current_row['full_gate_events'] == ref_row['full_gate_events']

        args = dict(probe=probe, semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer,
            history_parser=parser, real_threshold=threshold, shared_static=shared_static,
            diagnostic_force_probe_gate=True)
        with torch.inference_mode():
            native = run(Qwen3BatchLateAttention(wrapper.model, span, allowed, 'sadi'))
            if label == 'B16':
                assert native[0] == [row['response_token_ids'][:32] for row in first]
            gc.collect(); torch.cuda.empty_cache()
            reference = run(Qwen3BatchBoundedVisualTilt(wrapper.model, span, allowed,
                diagnostic_reject_all=True, **args))
            exact(reference, native)
            gc.collect(); torch.cuda.empty_cache()
            uniform_reference = run(UniformHeadReference(wrapper.model, span, allowed,
                refresh=False, tilt_mode='fixed', **args))
            gc.collect(); torch.cuda.empty_cache()
            cases=[('zero_replay','evidence',dict(diagnostic_zero_tilt=True)),
                ('forced_reject','evidence',dict(diagnostic_reject_all=True)),
                ('flat_reference','evidence',dict(diagnostic_flat_head_score=True)),
                ('uniform_reference','uniform',{}),('live_guided','evidence',{}),('live_roll','roll',{})]
            if label=='B2':
                cases=[('disabled','evidence',dict(disable_guidance=True)),
                    ('zero_gamma','evidence',dict(gamma=0.)),*cases]
            records={}
            for name,mode,options in cases:
                controller=Qwen3BatchEvidenceHeadFeedback(wrapper.model,span,allowed,
                    refresh=True,head_mode=mode,**args,**options)
                current=run(controller)
                unchanged=name in ('disabled','zero_gamma','zero_replay','forced_reject')
                uniform_exact=name in ('flat_reference','uniform_reference')
                if unchanged:
                    exact(current,native)
                    if name!='disabled':exact_gate(current,reference)
                if uniform_exact:
                    exact(current,uniform_reference);exact_gate(current,uniform_reference)
                stats=current[3];events=[e for row in stats for e in row['full_joint_events']]
                layers=[v for e in events for v in e['layer_details'].values()]
                assert all(row['gradient_probe_calls']==0 and row['evidence_head_routing'] for row in stats)
                assert all(row['intervention_layers']==list(TARGETS) for row in stats)
                assert all(e['source_layer']==19 and e['target_layer']==20 for e in events)
                assert all(x['source_layer']+1==x['target_layer'] for x in layers)
                if name=='zero_replay':
                    assert events and max(row['zero_replay_checks'] for row in stats)>0
                    assert all(x['visual_logit_shift_max']==x['visual_logit_shift_min']==0 for x in layers)
                    assert max(row['maximum_zero_replay_hidden_difference'] for row in stats)==0
                if name not in ('disabled','zero_gamma'):assert events
                if events:assert current[7]==15*len({e['step'] for e in events})
                for x in layers:
                    raw=torch.tensor(x['raw_head_score'],dtype=torch.float64)
                    applied=torch.tensor(x['applied_head_score'],dtype=torch.float64)
                    assert raw.abs().max()<=x['drift']+1e-6
                    expected=torch.zeros_like(raw) if uniform_exact else raw.roll(len(raw)//2) if mode=='roll' else raw
                    assert torch.equal(applied,expected)
                    assert x['fp32_mean_visual_mass_difference']<=32*torch.finfo(torch.float32).eps
                    assert x['bf16_mean_visual_mass_difference']<=2**-7
                    assert x['conditional_visual_difference']<=1e-5
                    if x['target_layer']==20:
                        assert x['source_q_max_difference']==x['source_p_max_difference']==0
                        assert x['native_source_head_score_max_difference']==0
                    if name!='zero_replay':
                        odds=torch.tensor(x['visual_log_odds'],dtype=torch.float64)
                        bias=torch.tensor(x['visual_bias'],dtype=torch.float64)
                        base=torch.tensor(x['reference_bias'],dtype=torch.float64)
                        assert abs(float((odds+bias).sigmoid().mean()-(odds+base).sigmoid().mean()))<=32*2**-23
                        if uniform_exact:assert torch.equal(bias,base)
                if name=='live_guided':
                    assert any(x['source_ffn_max_difference']>0 and x['source_q_max_difference']>0 for x in layers)
                    assert any(x['maximum_individual_head_mass_change']>0 for x in layers)
                assert current[5]<29*2**30
                records[name]=dict(proposals=len(events),kept=sum(e['accepted'] for e in events),
                    exact_native_tokens_logits_current36_KV=unchanged,
                    exact_original108_inputs=unchanged and name!='disabled',
                    exact_uniform_tokens_logits_current36_KV=uniform_exact,
                    exact_uniform_original108_inputs=uniform_exact,
                    all_previous_KV_tensor_checks=current[6],actual_attention_and_ffn_source_checks=current[7],
                    maximum_source_ffn_change=max((x['source_ffn_max_difference'] for x in layers),default=0.),
                    maximum_source_q_change=max((x['source_q_max_difference'] for x in layers),default=0.),
                    maximum_individual_head_mass_change=max((x['maximum_individual_head_mass_change'] for x in layers),default=0.),
                    maximum_fp32_mean_visual_mass_difference=max(row['maximum_fp32_mean_mass_difference'] for row in stats),
                    maximum_bf16_mean_visual_mass_difference=max(row['maximum_bf16_mean_mass_difference'] for row in stats),
                    maximum_conditional_visual_difference=max(row['maximum_conditional_visual_difference'] for row in stats),
                    actual_zero_replay_checks=max(row['zero_replay_checks'] for row in stats),
                    encoder_calls=1,model_calls=len(current[1]),backward_calls=0,peak_allocated=current[5])
                del current,controller
                gc.collect();torch.cuda.empty_cache()
        report['batches'][label] = dict(ids=ids, cases=records)
        del native, reference, uniform_reference, inputs, shared_static
        gc.collect(); torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-label', choices=('B2', 'B16'))
    args = parser.parse_args()
    root = REPO/'outputs/qwen3_evidence_head_routing_preflight_20261119'
    request = read(root/'request.json')
    destination = root/(f'verification.{args.batch_label}.json' if args.batch_label else 'verification.json')
    assert not destination.exists()
    for name, value in request['source_sha256'].items():
        assert sha(Path(name)) == value, name
    assert read(REPO/'outputs/qwen3_evidence_head_routing_cpu_20261119/verification.json')['passed']
    plan = read(REPO/'outputs/qwen3_block_acg_pilot_20261118/protocol.json')
    wrapper = build_wrapper()
    probe = FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    allowed = frozenset(read(REPO/'outputs/qwen3_block_acg_pilot_20261118/object_lexicon.json')['token_ids'])
    result = verify(wrapper, probe, allowed,
        ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'), plan['real_threshold'], args.batch_label)
    for name, value in request['source_sha256'].items():
        assert sha(Path(name)) == value, name
    result['request_sha256'] = sha(root/'request.json')
    result['source_sha256'] = {name: sha(REPO/name) for name in request['new_files']}
    write(destination, result)
    print(__import__('json').dumps(result, indent=2))
