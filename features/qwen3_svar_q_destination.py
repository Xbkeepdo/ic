"""Remove only FFN destination weights, preserving its adaptive amount."""

from unittest.mock import patch

import torch

from features import qwen3_probe_attention_recycle as recycle
from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass


def negative_write_destination(budget, normalized_negative_writes, start, end):
    """Keep each head's visual mass and all nonvisual entries unchanged."""
    assert budget.ndim == normalized_negative_writes.ndim == 3
    assert budget[:, :, start:end].shape == normalized_negative_writes.shape
    assert torch.isfinite(normalized_negative_writes).all() and normalized_negative_writes.min() >= 0
    mass = budget[:, :, start:end].sum(-1, keepdim=True)
    assert mass.min() >= 0
    active = mass[..., 0] > 0
    if active.any():
        assert (normalized_negative_writes.sum(-1)[active] - 1).abs().max() < 1e-6
    result = budget.clone()
    result[:, :, start:end] = mass * normalized_negative_writes
    assert (result[:, :, :start] == budget[:, :, :start]).all()
    assert (result[:, :, end:] == budget[:, :, end:]).all()
    assert result[:, :, start:end].min() >= 0 and result.sum(-1).abs().max() < 1e-6
    assert (result.abs().amax((1, 2)) > 0).all()
    return result


class Qwen3BatchSvarQDestination(Qwen3BatchSvarFirstClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.guided, 'The reference must use the evidence-guided signed routes.'
        self.negative_destination_captures = 0

    def _budget(self, source):
        gradient, normalize = self._gradient, recycle.normalize_nonnegative
        gradient_rows, normalized = [], []

        def remember_gradient(index, rows):
            assert index == source
            gradient_rows[:] = rows
            return gradient(index, rows)

        def remember_distribution(values):
            result = normalize(values)
            if values.ndim == 3:
                normalized.append(result.detach())
            return result

        self._gradient = remember_gradient
        try:
            with patch.object(recycle, 'normalize_nonnegative', remember_distribution):
                rows, budget, details = super()._budget(source)
        finally:
            self._gradient = gradient
        self.negative_destination_captures += bool(normalized)
        if not rows:
            return rows, budget, details
        # Parent first normalizes per-head negative writes, then their FFN
        # deficit-weighted destination. Reuse the first result without a new
        # gradient or any model/FFN/LM-head call.
        assert len(normalized) == 2 and len(gradient_rows) == normalized[0].shape[0]
        positions = [gradient_rows.index(row) for row in rows]
        q = normalized[0][positions]
        updated = negative_write_destination(budget, q, self.start, self.end)
        for index, detail in enumerate(details):
            old_prediction = detail.pop('predicted_log_probability_change')
            assert old_prediction <= 1e-6
            detail.update(destination_rule='per_head_negative_writes', ffn_destination_weighting=False,
                ffn_drift_amount_preserved=True, reference_predicted_log_probability_change=old_prediction,
                first_order_log_probability_change_upper_bound=0.,
                first_order_bound_from_negative_visual_and_positive_history=True,
                reference_moved_attention_mass=detail['moved_attention_mass'],
                moved_attention_mass=float(updated[index, :, self.start:self.end].sum()),
                reference_budget_max=detail['budget_max'], budget_max=float(updated[index].abs().max()),
                mass_conservation_error=float(updated[index].sum(-1).abs().max()),
                maximum_nonvisual_budget_difference=float(torch.cat((
                    updated[index, :, :self.start] - budget[index, :, :self.start],
                    updated[index, :, self.end:] - budget[index, :, self.end:]), -1).abs().max()))
        return rows, updated, details

    def statistics(self, responses):
        result = super().statistics(responses)
        for stat in result:
            stat.update(destination_rule='per_head_negative_writes', ffn_destination_weighting=False,
                ffn_drift_amount_preserved=True, negative_destination_captures=self.negative_destination_captures,
                additional_gradient_probes=0, additional_model_forward_calls=0,
                additional_mlp_forward_calls=0, additional_lm_head_calls=0)
        return result
