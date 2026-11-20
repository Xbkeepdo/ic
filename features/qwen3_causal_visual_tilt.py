"""Adjacent-layer bounded tilts after full SADI, with native/refreshed controls.

Read layer19 to guide20, then20->21 ...34->35. The original final-token
108-feature gate is unchanged. Both controls evaluate the same current-query
attention operator; only the source of Q/P differs. No backward is needed.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_bounded_visual_tilt import (
    Qwen3BatchBoundedVisualTilt, bounded_coefficients, evidence_preference,
    original_sam_rows, positive_visual_shift,
)
from features.qwen3_online_risk_f import native_write_norms
from features.qwen3_sar_ems_final_target import LastPositionCache

SOURCES = tuple(range(19, 35))
TARGETS = tuple(range(20, 36))


def current_attention(query, key, value, mask, scaling, groups, span,
                      rows, coefficient, gamma):
    """One softmax@V; keep full batch shape and all nonvisual logits intact."""
    assert query.shape[-2] == 1 and len(rows) == len(coefficient)
    key = qwen.repeat_kv(key, groups)
    value = qwen.repeat_kv(value, groups)
    scores = (query @ key.transpose(2, 3)) * scaling
    if mask is not None:
        scores = scores + mask[..., :key.shape[-2]]
    start, end = span
    selected = scores[rows].float()
    shift = positive_visual_shift(selected[..., start:end], coefficient, gamma)
    selected[..., start:end] += shift
    tilted = scores.float().clone()
    tilted[rows] = selected
    probability = tilted.softmax(-1, dtype=torch.float32).to(query.dtype)
    context = (probability @ value).transpose(1, 2).contiguous()
    return context, probability, shift


class Qwen3BatchCausalVisualTilt(Qwen3BatchBoundedVisualTilt):
    def __init__(self, *args, refresh=True, **kwargs):
        assert not kwargs.get('diagnostic_zero_replay', False), 'Use diagnostic_zero_tilt'
        super().__init__(*args, **kwargs)
        self.refresh = bool(refresh)
        self._causal_replaying = False
        self._fresh_attention = {}
        self._fresh_updates = {}
        self.causal_layer_calls = 0
        self.refreshed_norm_calls = 0
        self.zero_replay_checks = 0
        self.maximum_zero_replay_hidden_difference = 0.

    def _remember_causal_update(self, index):
        def remember(_module, _args, output):
            if self._causal_replaying:
                assert output.shape[1] == 1
                self._fresh_updates[index] = output[:, 0].detach()
        return remember

    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            stack.enter_context(super().capture())
            if self.enabled:
                for index in TARGETS[:-1]:
                    handle = self.decoder.layers[index].mlp.register_forward_hook(
                        self._remember_causal_update(index))
                    stack.callback(handle.remove)
            yield self

    def _source_coefficients(self, source, rows, attention=None, update=None):
        if self.tilt_mode == 'fixed':
            coefficient = self.scratch[source]['scores'].new_full(
                (len(rows), self.end-self.start), .5, dtype=torch.float32)
            return coefficient, coefficient.new_ones(len(rows))
        distributions = []
        for row in rows:
            if attention is None:
                norms = self.reader.write_norms[source, row]
                actual_update = self.scratch[source]['update'][row]
            else:
                assert update is not None
                values = self.reader.values[source][row:row+1]
                groups = attention.shape[1]//values.shape[1]
                norms = native_write_norms(qwen.repeat_kv(values, groups)[0],
                    attention[row], self.decoder.layers[source].self_attn.o_proj.weight)
                actual_update = update[row]
                self.refreshed_norm_calls += 1
            static = self.reader._static(source, row)
            log_s = static['log_s'][:, self.reader.columns[int(self.final_tokens[row])]]
            distributions.append(evidence_preference(norms, log_s,
                self.reader.visual[source][row], actual_update))
        q, p = (torch.stack([entry[i] for entry in distributions]) for i in (0, 1))
        return bounded_coefficients(q, p, self.tilt_mode)

    def _budget(self, _source):
        rows, real = original_sam_rows(self)
        if not rows or not self.gamma:
            return [], None, []
        native = {target: self._source_coefficients(target-1, rows) for target in TARGETS}
        maximum = torch.stack([entry[0].amax(-1) for entry in native.values()]).amax(0)
        # If every native budget is zero, causality implies the refreshed path
        # is also unchanged. The diagnostic deliberately executes the zero path.
        selected = list(range(len(rows))) if self.zero_tilt else (maximum > 0).nonzero().flatten().tolist()
        details = [dict(token=int(self.final_tokens[rows[i]]), gain=0., drift=0.,
            budget_max=0., budget_units='visual_attention_logit',
            probe_p_real=float(real[i]), probe_real_threshold=self.real_threshold,
            signal_guided=False, joint_layers=list(TARGETS), layer_details={},
            bounded_visual_tilt=True, causal_visual_tilt=True, tilt_mode=self.tilt_mode,
            refresh_source=self.refresh, same_final_target_throughout_replay=True)
            for i in selected]
        payload = {target: (coefficient[selected], drift[selected])
                   for target, (coefficient, drift) in native.items()}
        return [rows[i] for i in selected], payload, details

    def _replay(self, rows, native):
        # Layer19 is the unchanged anchor; only layers20--35 are recomputed.
        current = self.source_outputs[19].clone()
        old_enabled = self.enabled
        self.enabled = False
        self.replaying_tail = self._replaying = self._causal_replaying = True
        self._fresh_attention.clear()
        self._fresh_updates.clear()
        cache = LastPositionCache(self.tail_kwargs['past_key_values'], rows,
                                  check_identical=self.zero_tilt)
        kwargs = dict(self.tail_kwargs, past_key_values=cache)
        original = qwen.eager_attention_forward
        layer_records = {}
        active = {}

        def inject(module, query, key, value, attention_mask, scaling, dropout=0., **kw):
            index = getattr(module, 'layer_idx', None)
            if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in TARGETS:
                return original(module, query, key, value, attention_mask, scaling, dropout, **kw)
            assert index == active['target'] and not dropout and not module.training
            context, probability, shift = current_attention(query, key, value,
                attention_mask, scaling, module.num_key_value_groups,
                (self.start, self.end), rows, active['coefficient'], self.gamma)
            if index != TARGETS[-1]:
                self._fresh_attention[index] = probability[:, :, 0, self.start:self.end].detach()
            active['shift'] = shift
            self.causal_layer_calls += 1
            return context, probability

        try:
            with patch.object(qwen, 'eager_attention_forward', inject):
                for target in TARGETS:
                    source = target-1
                    base_coefficient, base_drift = native[target]
                    attention_difference = base_drift.new_zeros(len(rows))
                    update_difference = base_drift.new_zeros(len(rows))
                    if source == SOURCES[0]:
                        coefficient, drift = base_coefficient, base_drift
                    else:
                        attention = self._fresh_attention.pop(source)
                        update = self._fresh_updates.pop(source)
                        attention_difference = (attention[rows].float()
                            - self.reader.current[source]['attention'][rows].float()).abs().amax((1, 2))
                        update_difference = (update[rows].float()
                            - self.scratch[source]['update'][rows].float()).abs().amax(-1)
                        if self.refresh:
                            coefficient, drift = self._source_coefficients(source, rows, attention, update)
                        else:
                            coefficient, drift = base_coefficient, base_drift
                    if self.zero_tilt:
                        coefficient = torch.zeros_like(coefficient)
                    active = dict(target=target, coefficient=coefficient)
                    current = self.decoder.layers[target](current, **kwargs)
                    shift = active['shift']
                    layer_records[target] = [dict(source_layer=source, target_layer=target,
                        source_refreshed=self.refresh and source != SOURCES[0],
                        drift=float(drift[i]), coefficient_min=float(coefficient[i].min()),
                        coefficient_max=float(coefficient[i].max()), coefficient_mean=float(coefficient[i].mean()),
                        visual_logit_shift_min=float(shift[i].min()),
                        visual_logit_shift_max=float(shift[i].max()), budget_max=float(shift[i].max()),
                        coefficient_native_max_difference=float((coefficient[i]-base_coefficient[i]).abs().max()),
                        source_attention_max_difference=float(attention_difference[i]),
                        source_ffn_max_difference=float(update_difference[i]),
                        bounded_positive_visual_tilt=True) for i in range(len(rows))]
            assert cache.updates == list(TARGETS)
            assert not self._fresh_attention and not self._fresh_updates
            normalized = self.decoder.norm(current)
            unselected = [row for row in range(len(current)) if row not in rows]
            if unselected:
                assert torch.equal(normalized[unselected], self.reference_hidden[unselected])
            if self.zero_tilt:
                difference = float((normalized-self.reference_hidden).abs().max())
                self.maximum_zero_replay_hidden_difference = max(
                    self.maximum_zero_replay_hidden_difference, difference)
                assert difference == 0
                self.zero_replay_checks += 1
            self._causal_layer_records = layer_records
            return normalized
        finally:
            self.enabled = old_enabled
            self.replaying_tail = self._replaying = self._causal_replaying = False
            self._fresh_attention.clear()
            self._fresh_updates.clear()

    def _record(self, source, target, rows, details):
        assert source == target == 19
        for i, detail in enumerate(details):
            layers = {str(k): self._causal_layer_records[k][i] for k in TARGETS}
            detail['layer_details'] = layers
            detail['budget_max'] = max(x['budget_max'] for x in layers.values())
            detail['drift'] = sum(x['drift'] for x in layers.values())/len(TARGETS)
        super()._record(19, 20, rows, details)
        self._causal_layer_records.clear()

    def statistics(self, responses):
        result = super().statistics(responses)
        for stat in result:
            stat.update(causal_visual_tilt=True, refresh_source=self.refresh,
                intervention_layer='adjacent20--35', intervention_layers=list(TARGETS),
                replayed_tail_layers=list(TARGETS), budget_source_layers=list(SOURCES),
                common_native_point=False, budgets_common_native_point=not self.refresh,
                attention_scores_from_current_replay=True, same_final_target_throughout_replay=True,
                maximum_probe_hidden_difference=None, gradient_hidden_probe_performed=False,
                causal_layer_calls=self.causal_layer_calls, refreshed_norm_calls=self.refreshed_norm_calls,
                zero_replay_checks=self.zero_replay_checks,
                maximum_zero_replay_hidden_difference=self.maximum_zero_replay_hidden_difference,
                transport_mode='adjacent_bounded_positive_visual_logit_tilt',
                acg_scope='Same actual-query attention operator; native versus causally refreshed Q/P.',
                by_layer={str(k): dict(source_layer=k-1, target_layer=k,
                    source_refreshed=self.refresh and k != TARGETS[0]) for k in TARGETS})
        return result
