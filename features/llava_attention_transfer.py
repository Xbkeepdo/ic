"""Attention redistribution and visual write reinforcement for cached decoding."""

from __future__ import annotations

import torch


def write_evidence_distribution(write_norms, similarity, previous):
    # Root scaling limits the influence of a single high-magnitude write.
    root = write_norms.sqrt()
    current = root / root.sum().clamp_min(1e-12)
    evidence = current if previous is None else current + (current - previous).clamp_min(0)
    relative_similarity = (similarity - similarity.min()) / (
        similarity.max() - similarity.min()
    ).clamp_min(1e-6)
    evidence = evidence * (1 + relative_similarity)
    return current, evidence / evidence.sum().clamp_min(1e-12)


def transfer_template_attention(attention, visual_start, visual_end, evidence=None,
                                *, template_fraction=0.4, visual_limit=6.0):
    """Keep the instruction and generated-prefix weights fixed."""
    template = attention[..., :visual_start]
    visual = attention[..., visual_start:visual_end]
    template_mass = template.sum(dim=-1, keepdim=True)
    visual_mass = visual.sum(dim=-1, keepdim=True)
    amount = torch.minimum(template_fraction * template_mass, visual_limit * visual_mass)
    recipient = _visual_recipient(visual, visual_mass, evidence)
    updated = attention.clone()
    updated[..., :visual_start] = template * (1 - amount / template_mass.clamp_min(1e-12))
    updated[..., visual_start:visual_end] = visual + amount * recipient
    return updated, amount


def transfer_repeat_attention(attention, visual_start, visual_end, repeated_positions, evidence=None):
    donor = attention.index_select(-1, repeated_positions)
    visual = attention[..., visual_start:visual_end]
    donor_mass = donor.sum(dim=-1, keepdim=True)
    visual_mass = visual.sum(dim=-1, keepdim=True)
    amount = torch.minimum(0.75 * donor_mass, 2.0 * visual_mass)
    recipient = _visual_recipient(visual, visual_mass, evidence)
    updated = attention.clone()
    updated.index_copy_(-1, repeated_positions, donor * (1 - amount / donor_mass.clamp_min(1e-12)))
    updated[..., visual_start:visual_end] = visual + amount * recipient
    return updated, amount


def _visual_recipient(visual, visual_mass, evidence):
    native = visual / visual_mass.clamp_min(1e-12)
    if evidence is None:
        return native
    return (native + evidence) / (1 + evidence.sum())


def projected_visual_head_scores(visual_output, text_output, o_weight):
    """Score the visual write strength with a discount for text-heavy heads."""
    heads, head_dim = visual_output.shape
    projection = o_weight.reshape(o_weight.shape[0], heads, head_dim).permute(1, 2, 0)
    values = torch.stack((visual_output, text_output), dim=1)
    visual_norm, text_norm = torch.bmm(values, projection).float().norm(dim=-1).unbind(dim=1)
    return visual_norm * (visual_norm / (visual_norm + text_norm).clamp_min(1e-12))
