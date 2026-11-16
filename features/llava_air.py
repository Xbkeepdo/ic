"""AIR* release operators for unpadded LLaVA-1.5-7B cached decoding.

The AIR release also regularizes square attention during prefill. This
implementation shares the unedited prompt cache and edits one decode query
at a time. The default release leaves spectral QK rescaling disabled.
"""

from __future__ import annotations

import torch


AIR_LENS_LAYERS = range(5, 18)
_MODALITY_LAYERS = range(9, 15)
_HALLUCINATION_HEADS = {
    13: (31,),
    14: (16,),
    15: (10, 25, 24),
    16: (29,),
    17: (0, 28),
    18: (9, 26),
    19: (18, 27),
    20: (12,),
    22: (20,),
    26: (9, 28),
    27: (29,),
    30: (9,),
    31: (26, 3),
}
AIR_LAYERS = frozenset(AIR_LENS_LAYERS).union(_HALLUCINATION_HEADS)
AIR_TEXT_LAYERS = frozenset(_HALLUCINATION_HEADS)


def air_projection_gram(o_weight: torch.Tensor, heads: int) -> torch.Tensor:
    """Cache W_O,h^T W_O,h so head writes can be scored in head space."""
    projection = o_weight.detach().reshape(o_weight.shape[0], heads, -1)
    projection = projection.permute(1, 0, 2).float()
    return projection.transpose(1, 2) @ projection


def _head_write_weights(scores, values, visual_start, visual_end, projection_gram):
    text_scores = scores.clone()
    text_scores[..., visual_start:visual_end] = -torch.inf
    full_attention = scores.softmax(dim=-1, dtype=torch.float32)
    text_attention = text_scores.softmax(dim=-1, dtype=torch.float32)
    difference = (full_attention - text_attention).to(values.dtype)
    writes = (difference @ values).float()
    magnitude = ((writes @ projection_gram) * writes).sum(dim=-1, keepdim=True).clamp_min(0)
    total = magnitude.sum(dim=1, keepdim=True)
    safe_total = total.clamp_min(torch.finfo(magnitude.dtype).tiny)
    return torch.where(total > 0, magnitude / safe_total, 1 / scores.shape[1])


def _visual_text_agreement(attention, values, heads, start, end, gram):
    weights = attention[:, heads]
    values = values[:, heads]
    image = (weights[..., start:end] @ values[..., start:end, :]).float()
    text = (weights[..., end:] @ values[..., end:, :]).float()
    metric = gram[list(heads)]
    image_metric = image @ metric
    text_metric = text @ metric
    cosine = (image_metric * text).sum(dim=-1, keepdim=True)
    image_energy = (image_metric * image).sum(dim=-1, keepdim=True).clamp_min(0)
    text_energy = (text_metric * text).sum(dim=-1, keepdim=True).clamp_min(0)
    return (cosine / (image_energy * text_energy).clamp_min(1e-12).sqrt()).clamp(-1, 1)


def air_decode_attention(
    scores: torch.Tensor,
    values: torch.Tensor,
    visual_start: int,
    visual_end: int,
    layer: int,
    *,
    projected_gram: torch.Tensor | None = None,
    agreement_gram: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply AIR* to one cached query with exposed image keys and native head layout.

    Scores are shaped [batch, heads, 1, keys], and values use [batch, heads,
    keys, head_dim]. A projection Gram changes only the cross-head lens.
    The guided visual logits are normalized to the native lens's visual mass.
    """
    if not 0 <= visual_start < visual_end <= scores.shape[-1] or scores.shape[-2] != 1:
        raise ValueError("AIR* requires one query and a nonempty visual key span")

    logits = scores.clone()
    if layer in _MODALITY_LAYERS:
        logits[..., :visual_start] *= 0.97
        logits[..., visual_start:visual_end] *= 1.08

    if layer in AIR_LENS_LAYERS:
        visual = logits[..., visual_start:visual_end]
        native = visual + 0.28 * visual.abs().mean(dim=1, keepdim=True)
        if projected_gram is None:
            visual.copy_(native)
        else:
            weights = _head_write_weights(logits, values, visual_start, visual_end, projected_gram)
            relative_weights = weights - 1 / scores.shape[1]
            boost_change = 0.28 * (relative_weights * visual.float().abs()).sum(dim=1, keepdim=True)
            # Equal head weights preserve the native fp16 lens bit for bit.
            guided = native.float() + boost_change
            adjustment = native.float().logsumexp(dim=-1, keepdim=True) - guided.logsumexp(
                dim=-1, keepdim=True
            )
            logits = logits.float()
            logits[..., visual_start:visual_end] = guided + adjustment

    attention = logits.softmax(dim=-1, dtype=torch.float32).to(scores.dtype)
    heads = _HALLUCINATION_HEADS.get(layer)
    if heads:
        text = attention[:, heads, :, visual_end:]
        threshold = 0.4
        if agreement_gram is not None:
            agreement = _visual_text_agreement(attention, values, heads, visual_start, visual_end,
                                               agreement_gram)
            threshold = threshold + 0.25 * agreement
        over_threshold = text.sum(dim=-1, keepdim=True) >= threshold
        attention[:, heads, :, visual_end:] = text.masked_fill(over_threshold, 0)

    if layer in AIR_LENS_LAYERS:
        attention = attention.float()
        attention = 0.9 * attention + 0.1 * attention.mean(dim=-1, keepdim=True)
    return attention.to(scores.dtype)
