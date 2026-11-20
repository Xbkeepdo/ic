"""Temper a fixed write/FFN-guided proposal without changing its signed routes."""

import torch

from features.qwen3_probe_attention_recycle import normalize_nonnegative
from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass


def conservative_attention_budget(attention, budget, mode):
    """Return the linear or reverse-KL midpoint on the original changed support.

    System/prompt and all other untouched entries stay fixed. Sign clipping
    only removes floating-point roundoff at the support boundary.
    """
    if mode == 'half':
        return .5 * budget, torch.zeros(len(budget), device=budget.device)
    if mode != 'geometric':
        raise ValueError(mode)
    assert attention.shape == budget.shape and torch.isfinite(attention).all()
    proposed = attention + budget
    assert attention.min() >= 0 and proposed.min() >= -1e-7
    changed = budget != 0
    mass = (attention * changed).sum(-1, keepdim=True)
    # Separate square roots avoid underflow from multiplying small weights.
    weights = attention.sqrt() * proposed.clamp_min(0).sqrt() * changed
    midpoint = mass * normalize_nonnegative(weights)
    raw = (midpoint - attention) * changed
    result = torch.where(budget > 0, raw.clamp_min(0),
                         torch.where(budget < 0, raw.clamp_max(0), torch.zeros_like(raw)))
    correction = (result - raw).abs().amax((1, 2))
    assert correction.max() < 1e-6
    assert result.sum(-1).abs().max() < 1e-6
    assert torch.isfinite(result).all()
    return result, correction


class Qwen3BatchSvarAttentionMidpoint(Qwen3BatchSvarFirstClass):
    def __init__(self, *args, geometry_mode, **kwargs):
        super().__init__(*args, **kwargs)
        assert geometry_mode in ('half', 'geometric')
        assert self.guided, 'The signed-route bound requires the evidence-guided proposal.'
        self.geometry_mode = geometry_mode
        self.geometry_calls = 0

    def _budget(self, source):
        rows, budget, details = super()._budget(source)
        if not rows:
            return rows, budget, details
        self.geometry_calls += 1
        before = None
        if self.geometry_mode == 'geometric':
            state = self.scratch[source]
            before = state['scores'][rows, :, 0].softmax(-1, dtype=torch.float32).to(state['values'].dtype).float()
        updated, correction = conservative_attention_budget(before, budget, self.geometry_mode)
        assert updated.shape == budget.shape
        assert (updated[budget == 0] == 0).all()
        assert (updated[budget > 0] >= 0).all() and (updated[budget < 0] <= 0).all()
        assert updated.sum(-1).abs().max() < 1e-6
        # Every changed visual route opposes the target, and every donor
        # history route supports it. Keeping those signs gives a nonpositive
        # first-order bound without another gradient/projection computation.
        for position, detail in enumerate(details):
            reference_prediction = detail.pop('predicted_log_probability_change')
            assert reference_prediction <= 1e-6
            detail.update(attention_geometry=self.geometry_mode,
                reference_predicted_log_probability_change=reference_prediction,
                first_order_log_probability_change_upper_bound=0.,
                first_order_bound_from_unchanged_signed_support=True,
                reference_moved_attention_mass=detail['moved_attention_mass'],
                reference_budget_max=detail['budget_max'],
                moved_attention_mass=float(updated[position, :, self.start:self.end].sum()),
                budget_max=float(updated[position].abs().max()),
                midpoint_roundoff_correction=float(correction[position]),
                mass_conservation_error=float(updated[position].sum(-1).abs().max()))
        effective = (updated.abs().amax((1, 2)) > 0).nonzero().flatten().tolist()
        if not effective:
            return [], None, []
        if len(effective) != len(rows):
            self.cache_backup = {layer: (keys[effective], values[effective])
                                 for layer, (keys, values) in self.cache_backup.items()}
        return [rows[i] for i in effective], updated[effective], [details[i] for i in effective]

    def statistics(self, responses):
        result = super().statistics(responses)
        for stats in result:
            stats.update(attention_geometry=self.geometry_mode, midpoint_weight=.5,
                geometry_calls=self.geometry_calls, unchanged_signed_support=True,
                prompt_attention_fixed=True, additional_gradient_probes=0,
                additional_model_forward_calls=0, additional_mlp_forward_calls=0,
                additional_lm_head_calls=0)
        return result
