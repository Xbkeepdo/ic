"""Read I before SADI's local score edit, keeping the actual SADI residuals.

This is a local counterfactual readout, not a full no-SADI forward pass.
The model's attention outputs and generation are returned unchanged.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.llava_decoding_evidence import ATTENTION_LAYERS
from features.qwen3_online_i_el import OnlineIElCapture


class PreSadiIElCapture(OnlineIElCapture):
    def __init__(self, *args, cached_semantics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_semantics = cached_semantics

    def _input(self, index):
        if self.cached_semantics is None:
            return super()._input(index)

        def hook(_module, args, kwargs):
            if self.step == 0:
                hidden = args[0] if args else kwargs['hidden_states']
                cached = self.cached_semantics[index]
                assert cached.shape == (len(hidden), len(self.tokens))
                self.semantics[index] = cached.to(hidden.device)
        return hook

    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            stack.enter_context(super().capture())
            original = qwen.eager_attention_forward

            def read(module, query, key, value, *args, **kwargs):
                result = original(module, query, key, value, *args, **kwargs)
                if isinstance(module, qwen.Qwen3VLTextAttention) and module.layer_idx in ATTENTION_LAYERS:
                    scaling = kwargs.get('scaling', args[1] if len(args) > 1 else module.scaling)
                    mask = kwargs.get('attention_mask', args[0] if args else None)
                    keys = qwen.repeat_kv(key, module.num_key_value_groups)
                    scores = (query[:, :, -1:] @ keys.transpose(2, 3)) * scaling
                    if mask is not None:
                        scores = scores + mask[..., -1:, :key.shape[-2]]
                    weights = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
                    self.attention[module.layer_idx] = weights[:, :, 0, self.start:self.end].contiguous()
                return result

            stack.enter_context(patch.object(qwen, 'eager_attention_forward', read))
            yield self
