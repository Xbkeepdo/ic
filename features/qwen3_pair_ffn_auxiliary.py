"""Remove only an FFN component that favors a target over visual counterevidence."""

import torch


def pair_ffn_anchor(anchor, residual, update, row, direction, amount, policy):
    assert policy in ['baseline', 'pair', 'rotated', 'uniform', 'opposite']
    assert 0 <= amount <= .5 and direction.shape == update.shape[1:]
    assert torch.equal(residual+update[:, None], anchor)
    unit = direction.float()/direction.float().norm().clamp_min(1e-30)
    current = update[row].float()
    positive = float((current@unit).clamp_min(0))
    length = amount*positive
    if policy == 'baseline' or length == 0:
        return anchor, dict(positive_projection=positive, intended_removed_norm=0., actual_update_difference_norm=0.)
    if policy == 'pair': vector = unit
    elif policy == 'rotated': vector = unit.roll(len(unit)//2)
    elif policy == 'uniform': vector = current/current.norm().clamp_min(1e-30)
    else: vector = -unit
    modified = (current-length*vector).to(update.dtype)
    result = anchor.clone()
    result[row, 0] = residual[row, 0]+modified
    return result, dict(positive_projection=positive, intended_removed_norm=length,
        actual_update_difference_norm=float((current-modified.float()).norm()),
        original_update_norm=float(current.norm()), original_pair_projection=float(current@unit),
        changed_pair_projection=float(modified.float()@unit))
