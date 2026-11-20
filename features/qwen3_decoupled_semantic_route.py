"""Move visual attention using separate evidence destinations and FFN donors."""

from contextlib import contextmanager
from unittest.mock import patch
import math
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen


def movement_fraction(drift, gain, strength=5.):
    assert 0 <= drift <= 1 and strength >= 0
    return min(.5, strength*drift*math.tanh(gain/2)) if gain > 0 else 0.


def semantic_destination(mass, prior, target, rival, attention=None):
    assert mass.ndim == 2 and prior.shape == mass.shape[1:]
    relative = mass/prior.clamp_min(1e-30)
    difference = (relative[:, rival].log1p()-relative[:, target].log1p()).clamp_min(0)
    if attention is not None:
        assert attention.shape == difference.shape
        difference = difference*attention
    return difference/difference.sum().clamp_min(1e-30)


def move_visual_probability(probability, span, row, destination, donor_preference,
                            donor_reference, amount, uniform_donor=False):
    """Keep removed mass fixed when changing destination, or uniformizing donors."""
    assert probability.ndim == 4 and probability.shape[2] == 1
    assert 0 <= amount <= .5
    length = span[1]-span[0]
    assert destination.shape == donor_preference.shape == donor_reference.shape == (length,)
    if amount == 0: return probability, []
    for vector in [destination, donor_preference, donor_reference]:
        assert torch.isfinite(vector).all() and (vector >= 0).all()
        assert abs(float(vector.sum())-1) < 1e-5
    old = probability[row, :, 0, span[0]:span[1]]
    donor_rate = donor_preference/(donor_preference+donor_reference).clamp_min(1e-30)
    removed = old*(amount*donor_rate)[None]
    mass = removed.sum(-1, keepdim=True)
    if uniform_donor:
        removed = old*(mass/old.sum(-1, keepdim=True).clamp_min(1e-30))
    changed = old-removed+mass*destination[None]
    result = probability.clone()
    result[row, :, 0, span[0]:span[1]] = changed
    return result, mass[:, 0].tolist()


@contextmanager
def semantic_route(span, row, destination, preference, donor_reference, amount,
                   layers, uniform_donor=False):
    original = qwen.eager_attention_forward
    records = []
    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        index = getattr(module, 'layer_idx', None)
        if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in layers or amount == 0:
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        assert not dropout and not module.training and query.shape[-2] == 1
        key = qwen.repeat_kv(key, module.num_key_value_groups)
        value = qwen.repeat_kv(value, module.num_key_value_groups)
        scores = (query@key.transpose(2, 3))*scaling
        if attention_mask is not None: scores = scores+attention_mask[..., :key.shape[-2]]
        native = scores.softmax(-1, dtype=torch.float32)
        probability, moved = move_visual_probability(native, span, row, destination,
            preference, donor_reference, amount, uniform_donor)
        before = native[row, :, 0, span[0]:span[1]].sum(-1)
        after = probability[row, :, 0, span[0]:span[1]].sum(-1)
        assert torch.allclose(before, after, rtol=1e-6, atol=1e-8)
        cast = probability.to(query.dtype)
        native_cast = native.to(query.dtype)
        records.append(dict(layer=index, removed_mass_per_head=moved,
            fp32_mass_error=float((after-before).abs().max()),
            bf16_mass_error=float((cast[row, :, 0, span[0]:span[1]].float().sum(-1)-
                native_cast[row, :, 0, span[0]:span[1]].float().sum(-1)).abs().max())))
        return (cast@value).transpose(1, 2).contiguous(), cast
    with patch.object(qwen, 'eager_attention_forward', attention): yield records
    assert [r['layer'] for r in records] == (list(layers) if amount else [])
