"""Match uniform enhancement's visual partition while redistributing by Q/P."""

import torch


def matched_visual_coefficients(visual_logits, contrast, *, iterations=32):
    """Find a head-specific bias without changing the uniform-.5 partition.

    logits: [rows, heads, visual tokens]; contrast: [rows, visual tokens].
    contrast can be D*(2*Q/(Q+P)-1). Only its variation across patches matters.
    Every coefficient stays in [0,1]. Constant contrasts return exactly .5.
    """
    assert visual_logits.ndim == 3 and contrast.ndim == 2
    assert visual_logits.shape[0] == contrast.shape[0] and visual_logits.shape[-1] == contrast.shape[-1]
    assert iterations >= 24
    scores, direction = visual_logits.float(), contrast.float()
    assert torch.isfinite(scores).all() and torch.isfinite(direction).all()
    magnitude = scores.abs()
    target = (scores+.5*magnitude).logsumexp(-1)
    lo = (-direction.amax(-1)-32)[:, None].expand_as(target).clone()
    hi = (-direction.amin(-1)+32)[:, None].expand_as(target).clone()
    for _ in range(iterations):
        bias = (lo+hi)*.5
        coefficient = torch.sigmoid(direction[:, None, :]+bias[..., None])
        value = (scores+magnitude*coefficient).logsumexp(-1)
        below = value < target
        lo = torch.where(below, bias, lo)
        hi = torch.where(below, hi, bias)
    coefficient = torch.sigmoid(direction[:, None, :]+((lo+hi)*.5)[..., None])
    constant = direction.amax(-1) == direction.amin(-1)
    coefficient = torch.where(constant[:, None, None], torch.full_like(coefficient, .5), coefficient)
    # Scores identically zero have no shift at any coefficient. Use the uniform
    # representative rather than assigning significance to an arbitrary bias.
    zero_head = magnitude.amax(-1) == 0
    coefficient = torch.where(zero_head[..., None], torch.full_like(coefficient, .5), coefficient)
    achieved = (scores+magnitude*coefficient).logsumexp(-1)
    return coefficient, achieved-target


def mass_matched_attention(scores, values, visual_span, rows, contrast):
    """One softmax@V after solving the selected heads' visual partitions."""
    assert scores.ndim == 4 and scores.shape[-2] == 1
    assert values.shape[:2] == scores.shape[:2] and values.shape[-2] == scores.shape[-1]
    start, end = visual_span
    assert len(rows) == len(contrast) and len(rows) == len(set(rows))
    selected = scores[rows, :, 0, start:end]
    coefficient, residual = matched_visual_coefficients(selected, contrast)
    shift = selected.float().abs()*coefficient
    tilted = scores.float().clone()
    tilted[rows, :, 0, start:end] += shift
    probability = tilted.softmax(-1, dtype=torch.float32).to(values.dtype)
    context = (probability @ values).transpose(1, 2).contiguous()
    return context, probability, coefficient, residual
