"""Cached SAR patch terms and SADI-inspired attention budgeting."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from features.llava_sadi import head_magnitude_median


def visual_write_gram(values, o_weight) -> torch.Tensor:
    heads, _, head_dim = values.shape
    hidden = o_weight.shape[0]
    projection = o_weight.reshape(hidden, heads, head_dim).permute(1, 2, 0)
    # G[m,h,k] = dot(W_O,h v[h,m], W_O,k v[k,m]).
    head_writes = torch.bmm(values, projection).permute(1, 0, 2).float().contiguous()
    return torch.bmm(head_writes, head_writes.transpose(1, 2))


def visual_write_norms(gram, attention) -> torch.Tensor:
    head_weights = attention.T.float().contiguous()
    gram_times_weights = torch.bmm(gram, head_weights[:, :, None])[:, :, 0]
    return (head_weights * gram_times_weights).sum(dim=-1).clamp_min(0).sqrt()


class SarPatchTerms:
    """Reuse each visual head's output projection and full-vocabulary normalizer."""

    def __init__(self, values, o_weight, visual_hpre, final_norm, lm_head, *, gram=None):
        self.gram = visual_write_gram(values, o_weight) if gram is None else gram
        self.visual_logits = lm_head(final_norm(visual_hpre))
        self.log_normalizer = torch.logsumexp(self.visual_logits.float(), dim=-1)
        self.log_vocab = math.log(self.visual_logits.shape[-1])

    def distribution(self, attention, token: int) -> torch.Tensor:
        log_s = F.softplus(
            self.visual_logits[:, token].float() - self.log_normalizer + self.log_vocab
        )
        return self._weighted_writes(attention, log_s)

    def candidate_mixture(self, attention, tokens, weights) -> torch.Tensor:
        return self._weighted_writes(attention, self.log_semantics(tokens) @ weights)

    def log_semantics(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.softplus(
            self.visual_logits[:, tokens].float() - self.log_normalizer[:, None] + self.log_vocab
        )

    def _weighted_writes(self, attention, log_s) -> torch.Tensor:
        sar_terms = visual_write_norms(self.gram, attention) * (1 + log_s)
        return sar_terms / sar_terms.sum().clamp_min(1e-12)


def rebalance_visual_logits(visual_logits, budget, *, permute=False):
    """Upweight selected patches while preserving each head's visual mass."""
    visual = visual_logits.float()
    disagreement = visual.abs().std(dim=1, correction=0)
    spatial_scale = 0.25 + 0.75 * disagreement / disagreement.amax(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    shift = spatial_scale * budget
    if permute:
        shift = shift.roll(shift.shape[-1] // 3, dims=-1)
    shifted = visual + shift[:, None, :]
    correction = torch.logsumexp(shifted, dim=-1, keepdim=True) - torch.logsumexp(
        visual, dim=-1, keepdim=True
    )
    return (shifted - correction).to(visual_logits.dtype)


def mix_supported_visual_attention(attention, start, end, target, patch_budget):
    """Allocate visual attention to the SAR-supported portion of the EMS deficit."""
    visual = attention[..., start:end]
    visual_mass = visual.sum(dim=-1, keepdim=True)
    visual_distribution = visual / visual_mass.clamp_min(1e-12)
    head_disagreement = 0.5 * (visual_distribution - target).abs().sum(dim=-1, keepdim=True)
    addition = visual_mass * head_disagreement * patch_budget
    return _add_visual_attention(attention, start, end, addition)


def mix_visual_write_prior(attention, start, end, prior, strength):
    """Boost visual mass using current-query write norms as spatial evidence."""
    visual = attention[..., start:end]
    visual_mass = visual.sum(dim=-1, keepdim=True)
    visual_distribution = visual / visual_mass.clamp_min(1e-12)
    disagreement = 0.5 * (visual_distribution - prior).abs().sum(dim=-1, keepdim=True)
    proposal = torch.lerp(visual_distribution, prior, disagreement)
    budget = strength * visual_mass * (1 - visual_mass).clamp_min(0) * prior.sum()
    return _add_visual_attention(attention, start, end, budget * proposal)


def guide_visual_logits_with_writes(visual_logits, visual_attention, prior, strength, cap=None):
    """Increase logits where a head's visual distribution falls short of the writes."""
    conditional = visual_attention / visual_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    shortage = (prior - conditional).clamp_min(0) / prior.clamp_min(1e-12)
    budget = 0.25 + strength * prior / (prior + 1 / prior.numel())
    visual = visual_logits.float()
    scale = head_magnitude_median(visual.abs())
    correction = scale * budget * shortage
    if cap is not None:
        correction = cap * (correction / cap).tanh()
    return (visual + correction).to(visual_logits.dtype)


def _add_visual_attention(attention, start, end, addition):
    normalizer = 1 + addition.sum(dim=-1, keepdim=True)
    updated = attention / normalizer
    updated[..., start:end] = (attention[..., start:end] + addition) / normalizer
    return updated
