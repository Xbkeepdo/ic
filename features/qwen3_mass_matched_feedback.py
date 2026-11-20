"""Adjacent Q/P guidance at the same FP32 visual mass as uniform enhancement."""

from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_bounded_visual_tilt import original_sam_rows
from features.qwen3_causal_visual_tilt import Qwen3BatchCausalVisualTilt, SOURCES, TARGETS
from features.qwen3_mass_matched_visual import matched_visual_coefficients
from features.qwen3_sar_ems_final_target import LastPositionCache


class Qwen3BatchMassMatchedFeedback(Qwen3BatchCausalVisualTilt):
    def __init__(self, *args, permutation='none', diagnostic_flat_contrast=False, **kwargs):
        assert kwargs.pop('tilt_mode', 'evidence') == 'evidence'
        super().__init__(*args, tilt_mode='evidence', **kwargs)
        assert self.gamma in (0., 1.) and permutation in ('none', 'half_roll')
        self.permutation = permutation
        self.flat_contrast = bool(diagnostic_flat_contrast)
        self.maximum_fp32_visual_mass_difference = 0.
        self.maximum_bf16_visual_mass_difference = 0.

    def _source_contrast(self, source, rows, attention=None, update=None):
        coefficient, drift = self._source_coefficients(source, rows, attention, update)
        contrast = 2*coefficient-drift[:, None]
        if self.permutation == 'half_roll':
            contrast = contrast.roll(contrast.shape[-1]//2, -1)
        if self.flat_contrast:
            contrast = torch.zeros_like(contrast)
        return contrast, drift

    def _budget(self, _source):
        rows, real = original_sam_rows(self)
        if not rows or not self.gamma:
            return [], None, []
        native = {target: self._source_contrast(target-1, rows) for target in TARGETS}
        # Zero contrast is the uniform reference, rather than no intervention.
        details = [dict(token=int(self.final_tokens[row]), gain=0., drift=0.,
            budget_max=0., budget_units='visual_attention_logit', probe_p_real=float(real[i]),
            probe_real_threshold=self.real_threshold, signal_guided=False, joint_layers=list(TARGETS),
            layer_details={}, bounded_visual_tilt=True, causal_visual_tilt=True,
            mass_matched_visual=True, tilt_mode='evidence', refresh_source=self.refresh,
            contrast_permutation=self.permutation, same_final_target_throughout_replay=True)
            for i, row in enumerate(rows)]
        return rows, native, details

    def _replay(self, rows, native):
        current = self.source_outputs[19].clone()
        old_enabled = self.enabled
        self.enabled = False
        self.replaying_tail = self._replaying = self._causal_replaying = True
        self._fresh_attention.clear(); self._fresh_updates.clear()
        cache = LastPositionCache(self.tail_kwargs['past_key_values'], rows, check_identical=self.zero_tilt)
        kwargs = dict(self.tail_kwargs, past_key_values=cache)
        original = qwen.eager_attention_forward
        active = {}; layer_records = {}

        def inject(module, query, key, value, attention_mask, scaling, dropout=0., **kw):
            index = getattr(module, 'layer_idx', None)
            if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in TARGETS:
                return original(module, query, key, value, attention_mask, scaling, dropout, **kw)
            assert index == active['target'] and not dropout and not module.training
            key = qwen.repeat_kv(key, module.num_key_value_groups)
            value = qwen.repeat_kv(value, module.num_key_value_groups)
            scores = (query @ key.transpose(2, 3))*scaling
            if attention_mask is not None:
                scores = scores+attention_mask[..., :key.shape[-2]]
            visual = scores[rows, :, 0, self.start:self.end].float()
            if self.zero_tilt:
                coefficient = torch.zeros_like(visual)
                residual = visual.new_zeros(visual.shape[:2])
            else:
                coefficient, residual = matched_visual_coefficients(visual, active['contrast'])
            shift = visual.abs()*coefficient
            tilted = scores.float().clone()
            tilted[rows, :, 0, self.start:self.end] += shift
            probability_float = tilted.softmax(-1, dtype=torch.float32)
            probability = probability_float.to(query.dtype)
            # Reference softmax is for measured matching errors only; there is
            # no reference @V, extra decoder pass, or gradient calculation.
            reference_scores = scores[rows].float().clone()
            if not self.zero_tilt:
                reference_scores[:, :, 0, self.start:self.end] += .5*visual.abs()
            reference_float = reference_scores.softmax(-1, dtype=torch.float32)
            fp32_error = (probability_float[rows, :, 0, self.start:self.end].sum(-1)
                         -reference_float[:, :, 0, self.start:self.end].sum(-1)).abs().amax(-1)
            bf16_error = (probability[rows, :, 0, self.start:self.end].float().sum(-1)
                         -reference_float.to(query.dtype)[:, :, 0, self.start:self.end].float().sum(-1)).abs().amax(-1)
            self.maximum_fp32_visual_mass_difference = max(self.maximum_fp32_visual_mass_difference, float(fp32_error.max()))
            self.maximum_bf16_visual_mass_difference = max(self.maximum_bf16_visual_mass_difference, float(bf16_error.max()))
            if index != TARGETS[-1]:
                self._fresh_attention[index] = probability[:, :, 0, self.start:self.end].detach()
            active.update(coefficient=coefficient, shift=shift, residual=residual,
                          fp32_error=fp32_error, bf16_error=bf16_error)
            self.causal_layer_calls += 1
            return (probability @ value).transpose(1, 2).contiguous(), probability

        try:
            with patch.object(qwen, 'eager_attention_forward', inject):
                for target in TARGETS:
                    source = target-1
                    base_contrast, base_drift = native[target]
                    attention_difference = base_drift.new_zeros(len(rows))
                    update_difference = base_drift.new_zeros(len(rows))
                    if source == SOURCES[0]:
                        contrast, drift = base_contrast, base_drift
                    else:
                        attention = self._fresh_attention.pop(source)
                        update = self._fresh_updates.pop(source)
                        attention_difference = (attention[rows].float()
                            -self.reader.current[source]['attention'][rows].float()).abs().amax((1, 2))
                        update_difference = (update[rows].float()
                            -self.scratch[source]['update'][rows].float()).abs().amax(-1)
                        if self.refresh:
                            contrast, drift = self._source_contrast(source, rows, attention, update)
                        else:
                            contrast, drift = base_contrast, base_drift
                    active = dict(target=target, contrast=contrast)
                    current = self.decoder.layers[target](current, **kwargs)
                    coefficient, shift = active['coefficient'], active['shift']
                    layer_records[target] = [dict(source_layer=source, target_layer=target,
                        source_refreshed=self.refresh and source != SOURCES[0], drift=float(drift[i]),
                        coefficient_min=float(coefficient[i].min()), coefficient_max=float(coefficient[i].max()),
                        coefficient_mean=float(coefficient[i].mean()), contrast_min=float(contrast[i].min()),
                        contrast_max=float(contrast[i].max()), contrast_mean=float(contrast[i].mean()),
                        raw_contrast_native_max_difference=float((contrast[i]-base_contrast[i]).abs().max()),
                        visual_logit_shift_min=float(shift[i].min()), visual_logit_shift_max=float(shift[i].max()),
                        budget_max=float(shift[i].max()), source_attention_max_difference=float(attention_difference[i]),
                        source_ffn_max_difference=float(update_difference[i]), bounded_positive_visual_tilt=True,
                        mass_matched_visual=True, fp32_log_partition_residual=float(active['residual'][i].abs().max()),
                        fp32_visual_mass_difference=float(active['fp32_error'][i]),
                        bf16_visual_mass_difference=float(active['bf16_error'][i])) for i in range(len(rows))]
            assert cache.updates == list(TARGETS)
            assert not self._fresh_attention and not self._fresh_updates
            normalized = self.decoder.norm(current)
            unselected = [row for row in range(len(current)) if row not in rows]
            if unselected:
                assert torch.equal(normalized[unselected], self.reference_hidden[unselected])
            if self.zero_tilt:
                difference = float((normalized-self.reference_hidden).abs().max())
                self.maximum_zero_replay_hidden_difference = max(self.maximum_zero_replay_hidden_difference, difference)
                assert difference == 0
                self.zero_replay_checks += 1
            self._causal_layer_records = layer_records
            return normalized
        finally:
            self.enabled = old_enabled
            self.replaying_tail = self._replaying = self._causal_replaying = False
            self._fresh_attention.clear(); self._fresh_updates.clear()

    def statistics(self, responses):
        result = super().statistics(responses)
        for stat in result:
            stat.update(mass_matched_visual=True, contrast_permutation=self.permutation,
                diagnostic_flat_contrast=self.flat_contrast, mass_matching_solver_iterations=32,
                mass_matching_reference_coefficient=0. if self.zero_tilt else .5,
                mass_matching_precision='FP32 per-head visual partition; BF16 rounding error measured separately.',
                maximum_fp32_visual_mass_difference=self.maximum_fp32_visual_mass_difference,
                maximum_bf16_visual_mass_difference=self.maximum_bf16_visual_mass_difference,
                transport_mode='adjacent_mass_matched_visual_logit_tilt',
                acg_scope='Original108 gate plus evidence redistribution at matched uniform-reference visual mass.')
        return result
