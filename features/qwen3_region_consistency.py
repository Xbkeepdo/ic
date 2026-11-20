"""Verify a visual source with a finer crop, then attenuate its attention.

This is a project adaptation, not an implementation of a paper baseline.
The old region-memory reconstruction is reused only as a read-only visual
inspection. Generation uses native values and a visual-mass-preserving mask.
"""

from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_region_memory import RegionMemory, LAYERS
from models.dgst_capture import apply_decoder_final_norm


def category_probabilities(reader, row, aliases):
    cached = reader._static(19, row)['log_s']
    columns = [reader.columns[token] for token in aliases]
    return cached[:, columns].expm1().sum(-1) / reader.head.weight.shape[0]


def mask_attention(module, query, key, value, attention_mask, scaling, span, row, rect, grid, strength):
    key = qwen.repeat_kv(key, module.num_key_value_groups)
    value = qwen.repeat_kv(value, module.num_key_value_groups)
    scores = (query @ key.transpose(2, 3))*scaling
    if attention_mask is not None: scores = scores+attention_mask[..., :key.shape[-2]]
    start, end = span; height, width = grid; top, left, bottom, right = rect
    assert end-start == height*width and query.shape[-2] == 1 and strength > 0
    old = scores[row:row+1, :, :, start:end].float()
    penalty = torch.zeros((height, width), device=query.device, dtype=torch.float32)
    penalty[top:bottom, left:right] = float(strength)
    changed = old-penalty.flatten()
    changed += old.logsumexp(-1, keepdim=True)-changed.logsumexp(-1, keepdim=True)
    full = scores.float().clone(); full[row:row+1, :, :, start:end] = changed
    original_p = scores.float().softmax(-1)
    probability = full.softmax(-1)
    before = original_p[row, :, :, start:end].sum(-1)
    after = probability[row, :, :, start:end].sum(-1)
    error = float(((before-after).abs()/before.clamp_min(1e-12)).max())
    assert error < 1e-5
    probability = probability.to(query.dtype)
    return (probability @ value).transpose(1, 2).contiguous(), probability, error


class RegionConsistency(RegionMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verified_probabilities = {}; self.verification_cache_hits = 0

    def inspect(self, row, native_cache, reader, crop, rect, aliases, amount=1.):
        assert amount in (0., 1.)
        key = (row, tuple(rect), amount)
        height, width = self.grid(row); top, left, bottom, right = rect
        coarse = category_probabilities(reader, row, aliases).reshape(height, width)[top:bottom, left:right].mean()
        if key in self.verified_probabilities:
            self.verification_cache_hits += 1
        else:
            captured = []
            def hook(_module, args, kwargs):
                h = args[0] if args else kwargs['hidden_states']
                assert h.shape[1] == self.prompt
                captured.append(h[row, self.start:self.end].detach().clone())
            handle = reader.decoder.layers[19].register_forward_pre_hook(hook, with_kwargs=True)
            unused = None
            try: unused = self.keys(row, native_cache, reader, crop=crop, rect=rect, amount=amount)
            finally: handle.remove()
            del unused
            assert len(captured) == 1
            hidden = apply_decoder_final_norm(reader.decoder.norm, captured[0])
            # Match the native reader's whole-M/64-chunk head geometry and
            # log1p/expm1 arithmetic; an ROI-only head can change BF16 results.
            columns = torch.tensor(reader.tokens, device=hidden.device)
            log_s = []
            for offset in range(0, len(hidden), 64):
                logits = reader.head(hidden[offset:offset+64]).float()
                selected = logits[:, columns]-logits.logsumexp(-1, keepdim=True)
                log_s.append(torch.log1p(logits.shape[-1]*selected.clamp_max(0).exp()))
            self.verified_probabilities[key] = torch.cat(log_s)
        columns = [reader.columns[token] for token in aliases]
        fine = self.verified_probabilities[key][:, columns].expm1().sum(-1)/reader.head.weight.shape[0]
        fine = fine.reshape(height, width)[top:bottom, left:right].mean()
        if amount == 0: assert torch.equal(coarse, fine)
        discrepancy = float(((coarse-fine)/(coarse+fine).clamp_min(1e-30)).clamp(0, 1))
        return dict(coarse_category_probability=float(coarse), fine_category_probability=float(fine),
                    reliability_deficit=discrepancy, aliases=list(aliases))

    def attenuate(self, row, native_cache, reader, rect, strength, reference, zero=False):
        original = qwen.eager_attention_forward
        calls, errors = [], []
        def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            index = getattr(module, 'layer_idx', None)
            if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in LAYERS or not strength:
                return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            assert not dropout and not module.training
            context, probability, error = mask_attention(module, query, key, value, attention_mask,
                scaling, (self.start, self.end), row, rect, self.grid(row), strength)
            calls.append(index); errors.append(error)
            return context, probability
        with patch.object(qwen, 'eager_attention_forward', attention):
            normalized = self.replay(row, native_cache, reader, {}, reference, zero=zero)
        assert calls == (list(LAYERS) if strength else [])
        return normalized, max(errors, default=0.)
