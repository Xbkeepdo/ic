"""Contrast target-conditioned visual evidence and actual FFN preference."""

from contextlib import contextmanager
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_region_memory import LAYERS


def mix_visual(probability, span, row, prior, amount):
    assert 0 <= amount <= .5 and torch.isfinite(prior).all() and prior.min() >= 0
    start, end = span
    assert prior.shape == (end-start,) and prior.sum() > 0
    if amount == 0:
        return probability
    result = probability.clone(); visual = probability[row, :, :, start:end]
    mass = visual.sum(-1, keepdim=True); normalized = prior/prior.sum()
    result[row, :, :, start:end] = (1-amount)*visual+amount*mass*normalized
    assert torch.allclose(result[row, :, :, start:end].sum(-1, keepdim=True), mass, rtol=1e-5, atol=1e-8)
    return result


@contextmanager
def visual_preference(span, row, prior, amount):
    original = qwen.eager_attention_forward; calls = []
    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        index = getattr(module, 'layer_idx', None)
        if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in LAYERS or amount == 0:
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        assert not dropout and not module.training and query.shape[-2] == 1
        key = qwen.repeat_kv(key, module.num_key_value_groups)
        value = qwen.repeat_kv(value, module.num_key_value_groups)
        scores = (query @ key.transpose(2, 3))*scaling
        if attention_mask is not None: scores = scores+attention_mask[..., :key.shape[-2]]
        probability = scores.softmax(-1, dtype=torch.float32)
        changed = mix_visual(probability, span, row, prior, amount).to(query.dtype)
        assert torch.isfinite(changed).all(); calls.append(index)
        return (changed @ value).transpose(1, 2).contiguous(), changed
    with patch.object(qwen, 'eager_attention_forward', attention): yield calls
    assert calls == (list(LAYERS) if amount else [])
