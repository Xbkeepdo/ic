"""Local counterfactual logit lens for the actual visual contribution to attention."""

import torch
from torch.nn import functional as F


def remove_visual_write(controller, layer, rows):
    state = controller.scratch[layer]
    weights = state['attention'][rows].float()
    values = controller.visual_values[layer][rows].float()
    context = torch.einsum('bhm,bhmd->bhd', weights, values).flatten(1)
    projection = controller.decoder.layers[layer].self_attn.o_proj
    if layer not in controller.projections:
        controller.projections[layer] = projection.weight.float()
    visual = F.linear(context, controller.projections[layer])
    residual = state['ffn_residual'][rows].float()
    update = state['update'][rows].float()
    return residual, update, visual


def readout_at(controller, hidden, weights):
    """Apply the exact RMSNorm formula in FP32, then read selected token logits."""
    norm = controller.decoder.norm
    hidden = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    return torch.einsum('bkd,bd->bk', weights, hidden * norm.weight.float())


def readout_contrast(controller, layer, rows, tokens):
    residual, update, visual = remove_visual_write(controller, layer, rows)
    weights = controller.head.weight[tokens].float()
    before = readout_at(controller, residual, weights)
    after = readout_at(controller, residual + update, weights)
    removed = readout_at(controller, residual - visual, weights)
    removed_after = readout_at(controller, residual + update - visual, weights)
    return before - removed, after - before, after - removed_after, after
