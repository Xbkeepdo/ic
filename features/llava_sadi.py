"""Visual attention logit adjustment from SADI, arXiv:2605.24957v1."""

from __future__ import annotations

import math

import torch


def head_magnitude_median(magnitudes: torch.Tensor) -> torch.Tensor:
    """Use the average of the two central heads when the head count is even."""
    ordered = magnitudes.sort(dim=1).values
    midpoint = magnitudes.shape[1] // 2
    median = ordered[:, midpoint : midpoint + 1]
    if magnitudes.shape[1] % 2 == 0:
        median = (median + ordered[:, midpoint - 1 : midpoint]) * 0.5
    return median


def sadi_visual_logits(
    visual_logits: torch.Tensor,
    *,
    alpha_min: float = 0.25,
    alpha_max: float = 0.80,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply SADI Eq. (1)–(6) to finite last-query visual logits [B, H, M].

    Heads share a patch-wise median and budget. Each head has its own soft mask.
    """
    if visual_logits.ndim != 3 or min(visual_logits.shape) < 1:
        raise ValueError("Expected nonempty visual logits with shape [B, H, M]")
    if not visual_logits.is_floating_point():
        raise TypeError("Visual logits must be floating point")
    if not (0.0 <= alpha_min <= alpha_max < math.inf and 0.0 < eps < math.inf):
        raise ValueError("Expected finite 0 <= alpha_min <= alpha_max and positive eps")
    if alpha_max == 0.0:
        return visual_logits

    # https://arxiv.org/html/2605.24957v1#S3  (layers 5–18 for LLaVA-1.5)
    logits = visual_logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    magnitudes = logits.abs()
    median = head_magnitude_median(magnitudes)
    spread = magnitudes.std(dim=1, correction=0, keepdim=True)
    min_spread = spread.amin(dim=-1, keepdim=True)
    max_spread = spread.amax(dim=-1, keepdim=True)
    relative_spread = (spread - min_spread) / (max_spread - min_spread + eps)
    budget = alpha_min + (alpha_max - alpha_min) * relative_spread
    mask = ((magnitudes - median).abs() / (spread + eps)).clamp_max(1.0)
    correction = budget * median * mask

    return (logits + correction).to(dtype=visual_logits.dtype)
