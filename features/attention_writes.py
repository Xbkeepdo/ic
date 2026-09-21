"""Per-visual-token attention writes in the decoder residual space."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from models.dgst_capture import attention_row_from_capture


@torch.inference_mode()
def visual_attention_writes(
    layer: Any,
    capture: dict[str, Any],
    positions: Sequence[int],
    visual_start: int,
    visual_end: int,
) -> torch.Tensor:
    """Return ``[target, visual, hidden]`` attention contributions ``a_m``.

    The output-projection bias belongs to the complete attention update and
    is excluded from each individual token's write.
    """
    if hasattr(layer, "self_attn"):
        attention = layer.self_attn
        normalized = layer.input_layernorm(capture["h_prev"][0])
        value_projection = attention.v_proj
        output_projection = attention.o_proj
        packed = False
    elif hasattr(layer, "attention"):
        attention = layer.attention
        normalized = layer.attention_norm(capture["h_prev"][0])
        value_projection = attention.wqkv
        output_projection = attention.wo
        packed = True
    else:
        raise TypeError(f"Unsupported decoder layer: {type(layer).__name__}")

    config = getattr(attention, "config", None)
    heads = int(getattr(attention, "num_heads", getattr(config, "num_attention_heads", 0)))
    kv_heads = int(getattr(attention, "num_key_value_heads", getattr(config, "num_key_value_heads", 0)))
    head_dim = int(getattr(attention, "head_dim", normalized.shape[-1] // heads))
    groups = heads // kv_heads
    length = normalized.shape[0]
    projected = value_projection(normalized)
    if packed:
        values = projected.view(length, kv_heads, 2 + groups, head_dim)[:, :, -1, :]
    else:
        values = projected.view(length, kv_heads, head_dim)
    values = values.permute(1, 0, 2).repeat_interleave(groups, dim=0)
    rows = torch.stack(
        [attention_row_from_capture(capture, int(position)) for position in positions]
    ).to(device=values.device, dtype=values.dtype)
    weights = rows[:, :, visual_start:visual_end]
    visual_values = values[:, visual_start:visual_end, :]
    head_writes = (
        weights.permute(0, 2, 1).unsqueeze(-1)
        * visual_values.permute(1, 0, 2).unsqueeze(0)
    ).reshape(len(positions), visual_end - visual_start, heads * head_dim)
    return F.linear(head_writes, output_projection.weight, bias=None).float()


def attention_write_source_distribution(
    ffn_outputs: torch.Tensor,
    writes: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Project FFN outputs onto each write, or use temperature-scaled cosine."""
    if mode not in {"pam", "cam"}:
        raise ValueError("attention-write source mode must be pam or cam")
    ffn_outputs = ffn_outputs.float()
    writes = writes.float()
    dot = (writes * ffn_outputs[:, None, :]).sum(dim=-1)
    write_norm = writes.norm(dim=-1).clamp_min(1e-12)
    if mode == "pam":
        scores = dot / write_norm
    else:
        ffn_norm = ffn_outputs.norm(dim=-1)[:, None].clamp_min(1e-12)
        scores = dot / (write_norm * ffn_norm * 0.07)
    return torch.softmax(scores, dim=-1)
