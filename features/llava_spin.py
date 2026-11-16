"""SPIN head suppression and an output-projected visual-write ablation."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers.models.llama import modeling_llama as llama


class SpinAttention:
    # LLaVA-1.5-7B CHAIR recipe from YUECHE77/SPIN.
    retained_fraction = 0.95
    suppression_scale = 0.08

    def __init__(self):
        self.projection_grams: dict[int, torch.Tensor] = {}

    def projected_scores(self, module, visual_output):
        heads, width = visual_output.shape[-2:]
        layer = int(module.layer_idx)
        if layer not in self.projection_grams:
            weight = module.o_proj.weight.reshape(-1, heads, width).permute(1, 0, 2)
            weight = weight.float().contiguous()
            self.projection_grams[layer] = weight.transpose(1, 2) @ weight
        writes = visual_output.float().transpose(0, 1)
        gram = self.projection_grams[layer]
        return ((writes @ gram) * writes).sum(dim=-1).transpose(0, 1)

    @contextmanager
    def capture(self, start: int, end: int, *, projected_write: bool, layers=range(32)):
        original = llama.eager_attention_forward

        def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if int(module.layer_idx) not in layers or key.shape[-2] < end:
                return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            key = llama.repeat_kv(key, module.num_key_value_groups)
            value = llama.repeat_kv(value, module.num_key_value_groups)
            logits = (query @ key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                logits = logits + attention_mask[..., :key.shape[-2]]
                logits = logits.clamp_min(torch.finfo(logits.dtype).min)
            attention = logits.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            output = attention @ value

            if projected_write:
                visual_output = attention[:, :, -1:, start:end] @ value[:, :, start:end, :]
                scores = self.projected_scores(module, visual_output.squeeze(2))
            else:
                # The public implementation ranks the final query's image logits.
                # Its per-layer rescaling and softmax are monotonic in this sum.
                scores = logits[:, :, -1, start:end].sum(dim=-1, dtype=torch.float32)
            retained = int(output.shape[1] * self.retained_fraction)
            heads = scores.topk(retained, dim=1).indices
            gains = torch.full_like(scores, self.suppression_scale, dtype=output.dtype)
            gains.scatter_(dim=1, index=heads, value=1.0)
            output[:, :, -1, :] *= gains.unsqueeze(-1)
            return output.transpose(1, 2).contiguous(), attention

        with patch.object(llama, "eager_attention_forward", forward):
            yield
