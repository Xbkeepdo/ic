"""Target-conditioned visual erasure for a read-only contrastive branch.

This is an ASCD-inspired project adaptation, not an ASCD reproduction.
The ordinary full-SADI path owns history; this branch never commits KV.
"""

from contextlib import contextmanager
import math
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_region_memory import LAYERS


def select_positions(attention, evidence, preference, policy, fraction=.1):
    assert policy in ('attention', 'sar', 'sar_ems', 'shifted')
    assert attention.ndim == 2 and evidence.ndim == preference.ndim == 1
    assert attention.shape[-1] == evidence.numel() == preference.numel()
    assert 0 < fraction < 1
    importance = {'attention': attention.float().sum(0), 'sar': evidence,
                  'sar_ems': (evidence-preference).clamp_min(0),
                  'shifted': (evidence-preference).clamp_min(0)}[policy]
    if importance.max().item() == 0:
        importance = evidence
    # Stable ties are resolved by visual position, not device topk tie order.
    chosen = torch.argsort(importance, descending=True, stable=True)[:math.ceil(fraction*len(evidence))]
    if policy == 'shifted':
        chosen = (chosen+len(evidence)//2) % len(evidence)
    return chosen


def contrast_logits(native, negative, alpha=1., beta=.1):
    assert native.shape == negative.shape and 0 <= alpha and 0 < beta <= 1
    assert torch.isfinite(native).all() and torch.isfinite(negative).all()
    if alpha == 0:
        return native
    scores = (1+alpha)*native-alpha*negative
    keep = native >= native.amax(dim=-1, keepdim=True)+math.log(beta)
    return scores.masked_fill(~keep, -torch.inf)


@contextmanager
def erase_visual(span, row, selected):
    """Erase selected visual logits in layers20--35 for one current query."""
    original = qwen.eager_attention_forward
    calls = []
    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        index = getattr(module, 'layer_idx', None)
        if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in LAYERS or not len(selected):
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        assert not dropout and not module.training and query.shape[-2] == 1
        key = qwen.repeat_kv(key, module.num_key_value_groups)
        value = qwen.repeat_kv(value, module.num_key_value_groups)
        scores = (query @ key.transpose(2, 3))*scaling
        if attention_mask is not None:
            scores = scores+attention_mask[..., :key.shape[-2]]
        start, end = span
        assert start < end <= scores.shape[-1] and selected.min() >= 0 and selected.max() < end-start
        scores[row, :, :, start+selected] = -torch.inf
        probability = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        assert not probability[row, :, :, start+selected].any()
        assert torch.isfinite(probability).all()
        calls.append(index)
        return (probability @ value).transpose(1, 2).contiguous(), probability
    with patch.object(qwen, 'eager_attention_forward', attention):
        yield calls
    assert calls == (list(LAYERS) if len(selected) else [])
