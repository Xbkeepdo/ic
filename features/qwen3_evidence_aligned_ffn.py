"""Amplify an FFN component already aligned with a visually supported rival."""

import torch


def support_ffn_anchor(anchor, residual, update, row, direction, amount, policy):
    assert policy in ['baseline', 'aligned', 'rotated', 'uniform', 'opposite']
    assert 0 <= amount <= .5 and direction.shape == update.shape[1:]
    assert torch.equal(residual+update[:, None], anchor)
    unit = direction.float()/direction.float().norm().clamp_min(1e-30)
    current = update[row].float()
    projection = float(current@unit)
    length = amount*max(0., projection)
    details = dict(original_support_projection=projection, intended_added_norm=0.,
                   actual_update_difference_norm=0.)
    if policy == 'baseline' or length == 0: return anchor, details
    if policy == 'aligned': vector = unit
    elif policy == 'rotated': vector = unit.roll(len(unit)//2)
    elif policy == 'uniform': vector = current/current.norm().clamp_min(1e-30)
    else: vector = -unit
    modified = (current+length*vector).to(update.dtype)
    result = anchor.clone()
    result[row, 0] = residual[row, 0]+modified
    details.update(intended_added_norm=length,
        actual_update_difference_norm=float((current-modified.float()).norm()),
        original_update_norm=float(current.norm()), changed_support_projection=float(modified.float()@unit))
    return result, details
