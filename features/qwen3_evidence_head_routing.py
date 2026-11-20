"""Route a fixed mean visual attention mass toward evidence-aligned heads.

This is a project adaptation of head selection, not ACT/AHEAD/CausalLens.
Our source Q/P distributions determine destination-head allocation. Constant
per-head visual logit biases leave within-head spatial preferences intact.
"""

import torch


def evidence_head_scores(visual_logits, evidence, preference):
    assert visual_logits.ndim == 3 and evidence.shape == preference.shape
    assert evidence.shape == (visual_logits.shape[0], visual_logits.shape[-1])
    conditional = visual_logits.float().softmax(-1)
    q, p = evidence.float(), preference.float()
    assert torch.isfinite(q).all() and torch.isfinite(p).all()
    assert q.min() >= 0 and p.min() >= 0
    affinity_q = (conditional.sqrt()*q[:, None].sqrt()).sum(-1)
    affinity_p = (conditional.sqrt()*p[:, None].sqrt()).sum(-1)
    gap = affinity_q-affinity_p
    centered = gap-gap.mean(-1, keepdim=True)
    scale = centered.abs().amax(-1, keepdim=True)
    drift = (1-(q*p).sqrt().sum(-1)).clamp(0, 1)
    score = drift[:, None]*centered/scale.clamp_min(torch.finfo(torch.float32).eps)
    assert torch.isfinite(score).all()
    return score, dict(drift=drift, affinity_evidence=affinity_q,
        affinity_ffn=affinity_p, support_gap=gap, maximum_centered_gap=scale[:, 0])


def matched_head_biases(visual_logits, nonvisual_log_partition, head_score, *, iterations=32):
    """Preserve the current-query uniform-control mean of head visual masses.

    The reference head bias is .5*mean(abs(visual logits)). A row-specific
    scalar b matches mean_h sigmoid(Lvis-Lnonvis+reference+score+b) to the
    uniform-control mean. The solve never couples different images.
    """
    assert visual_logits.ndim == 3
    assert head_score.shape == nonvisual_log_partition.shape == visual_logits.shape[:2]
    assert iterations >= 24
    visual = visual_logits.float(); score = head_score.float()
    reference = .5*visual.abs().mean(-1)
    log_odds = visual.logsumexp(-1)-nonvisual_log_partition.float()
    assert torch.isfinite(log_odds).all() and torch.isfinite(score).all()
    masses = (log_odds+reference).sigmoid()
    target = masses.mean(-1)
    low, high = -score.amax(-1), -score.amin(-1)
    for _ in range(iterations):
        middle = .5*(low+high)
        achieved = (log_odds+reference+score+middle[:, None]).sigmoid().mean(-1)
        below = achieved < target
        low = torch.where(below, middle, low)
        high = torch.where(below, high, middle)
    bias = .5*(low+high)
    shift = reference+score+bias[:, None]
    constant = score.amax(-1) == score.amin(-1)
    saturated = (masses*(1-masses)).sum(-1) == 0
    shift = torch.where((constant | saturated)[:, None], reference, shift)
    residual = (log_odds+shift).sigmoid().mean(-1)-target
    return shift, dict(reference_bias=reference, offset=bias,
        reference_visual_mass=masses, solved_visual_mass=(log_odds+shift).sigmoid(),
        mean_visual_mass_residual=residual)


def head_routing_attention(scores, values, span, rows, evidence, preference, *,
                           mode='evidence', zero_tilt=False, flat_score=False):
    assert mode in ('uniform', 'evidence', 'roll')
    assert scores.ndim == values.ndim == 4 and scores.shape[-2] == 1
    assert len(rows) == len(set(rows)) == len(evidence)
    start, end = span
    visual = scores[rows, :, 0, start:end].float()
    nonvisual = torch.cat((scores[rows, :, 0, :start], scores[rows, :, 0, end:]), -1)
    raw, diagnostics = evidence_head_scores(visual, evidence, preference)
    actual = raw.roll(raw.shape[-1]//2, -1) if mode == 'roll' else raw
    if mode == 'uniform' or flat_score:
        actual = torch.zeros_like(actual)
    shifts, solver = matched_head_biases(visual, nonvisual.float().logsumexp(-1), actual)
    if zero_tilt:
        shifts = torch.zeros_like(shifts)
    tilted = scores.float().clone()
    tilted[rows, :, 0, start:end] += shifts[..., None]
    probability_float = tilted.softmax(-1, dtype=torch.float32)
    probability = probability_float.to(values.dtype)
    context = (probability@values).transpose(1, 2).contiguous()
    reference_scores = scores[rows].float().clone()
    if not zero_tilt:
        reference_scores[:, :, 0, start:end] += solver['reference_bias'][..., None]
    reference = reference_scores.softmax(-1, dtype=torch.float32)
    mass = probability_float[rows, :, 0, start:end].sum(-1)
    reference_mass = reference[:, :, 0, start:end].sum(-1)
    fp32_difference = (mass.mean(-1)-reference_mass.mean(-1)).abs()
    bf16_difference = (probability[rows, :, 0, start:end].float().sum(-1).mean(-1)
        -reference.to(values.dtype)[:, :, 0, start:end].float().sum(-1).mean(-1)).abs()
    conditional_difference = ((visual+shifts[..., None]).softmax(-1)
        -visual.softmax(-1)).abs().amax((1, 2))
    return context, probability, dict(diagnostics, **solver, raw_head_score=raw,
        applied_head_score=actual, visual_bias=shifts,
        fp32_mean_visual_mass_difference=fp32_difference,
        bf16_mean_visual_mass_difference=bf16_difference,
        conditional_visual_difference=conditional_difference,
        maximum_individual_head_mass_change=(mass-reference_mass).abs().amax(-1))
