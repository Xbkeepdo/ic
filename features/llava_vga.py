"""LLaVA Vision-Guided Attention and a projected-write coverage ablation.

VGA is from Zhao et al., CVPR 2026 (https://arxiv.org/abs/2511.20032).
The baseline follows beta-nlp/VGA's LLaVA attention and greedy sampler.
https://github.com/beta-nlp/VGA/blob/main/llava/model/language_model/modeling_llama.py
https://github.com/beta-nlp/VGA/blob/main/vcd_utils/greedy_sample.py
"""

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.llama import modeling_llama as llama

from features.llava_sar_ems_attention import visual_write_gram, visual_write_norms


class VgaAttention:
    """Keep separately reset generation guidance over one shared image prefill."""

    layers = range(2, 16)
    write_layer = 18
    gain = 0.2
    coverage_step = 0.02

    def __init__(self, model, tokenizer, start: int, end: int):
        if not 0 <= start < end:
            raise ValueError("Expected a nonempty visual token span")
        self.norm = model.language_model.norm
        self.head = model.get_output_embeddings()
        self.tokenizer = tokenizer
        self.start = start
        self.end = end
        self.visual_probabilities = None
        self.initial_guidance = None
        self.write_gram = None
        self.visual_attention = None
        self.projected_write = False
        self.guidance = None
        self.coefficient = None

    @contextmanager
    def capture_prefill(self):
        """Read the final-normalized visual rows from the unchanged prefix pass."""

        def capture(_norm, _args, output):
            visual = output[0, self.start:self.end]
            probabilities = self.head(visual).float().softmax(dim=-1)
            top_probabilities = probabilities.topk(10, dim=-1).values.float()
            salience = -(top_probabilities * (top_probabilities + 1e-8).log()).sum(dim=-1)
            salience = salience / math.log(10)
            self.visual_probabilities = probabilities
            self.initial_guidance = (salience / salience.sum()).to(probabilities.dtype)

        hook = self.norm.register_forward_hook(capture)
        try:
            yield
        finally:
            hook.remove()

    def _set_guidance(self, guidance):
        self.guidance = guidance
        self.coefficient = self.gain * (guidance > 1e-8).sum() / guidance.numel()

    def prepare(self, response: list[int]):
        """Update coverage using the emitted token and the query which predicted it."""
        if not response:
            return
        token = response[-1]
        if not self.tokenizer.convert_ids_to_tokens(token).startswith("▁"):
            return
        token_map = self.visual_probabilities[:, token]
        if self.projected_write:
            writes = visual_write_norms(self.write_gram, self.visual_attention)
            token_map = token_map.float() * writes
        guidance = self.guidance
        mass = token_map.sum()
        token_map = (token_map / torch.where(mass > 0, mass, 1.0)).to(guidance.dtype)
        updated = ((1 + self.coverage_step) * guidance - self.coverage_step * token_map).relu()
        remaining_mass = updated.sum()
        updated = updated / torch.where(remaining_mass > 0, remaining_mass, 1.0)
        # The released sampler has no fallback when the selected token receives zero probability.
        updated = torch.where((mass > 0) & (remaining_mass > 0), updated, guidance)
        self._set_guidance(updated)

    @contextmanager
    def capture(self, *, projected_write: bool = False):
        """Apply the official additive value correction during single-query decoding."""
        if self.initial_guidance is None:
            raise RuntimeError("Capture the image prefill before starting VGA")
        self.projected_write = projected_write
        self.visual_attention = None
        self._set_guidance(self.initial_guidance)
        original = llama.eager_attention_forward

        def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            output, attention = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            layer = int(module.layer_idx)
            if projected_write and layer == self.write_layer:
                if self.write_gram is None:
                    values = self._visual_values(module, value)
                    self.write_gram = visual_write_gram(values[0], module.o_proj.weight)
                self.visual_attention = attention[0, :, -1, self.start:self.end].contiguous()
            if layer in self.layers:
                values = self._visual_values(module, value)
                guidance = self.guidance.to(values.dtype)[None, None, None, :]
                patch_update = (guidance @ values).transpose(1, 2)
                similarity = F.cosine_similarity(output, patch_update, dim=-1, eps=1e-6)
                similarity = (1 + similarity) / 2
                head_total = similarity.sum(dim=-1, keepdim=True)
                head_weights = similarity / torch.where(head_total > 0, head_total, 1.0)
                head_weights = (2 - head_weights * similarity.shape[-1]).relu()
                coefficient = torch.where(self.coefficient > 1e-6, self.coefficient, 0.0)
                output = output + (coefficient * head_weights).unsqueeze(-1) * patch_update
            return output, attention

        with patch.object(llama, "eager_attention_forward", forward):
            yield

    def _visual_values(self, module, values):
        visual = values[:, :, self.start:self.end, :]
        return llama.repeat_kv(visual, module.num_key_value_groups)
