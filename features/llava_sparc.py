"""SPARC's LLaVA COCO decoding recipe with a projected-write selection ablation."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers.models.llama import modeling_llama as llama

from features.llava_sar_ems_attention import visual_write_gram, visual_write_norms


# https://github.com/mingi000508/SPARC/blob/main/scripts/llava/captioning_coco.sh
SPARC_ALPHA = 1.1
SPARC_BETA = 0.1
SPARC_THRESHOLD = 1.5
SPARC_SELECTION_LAYER = 20
SPARC_SCALE_LAYERS = range(32)


class _SparcState:
    def __init__(self, start: int, end: int, projected_write: bool):
        self.start = start
        self.end = end
        self.projected_write = projected_write
        self.selected = None
        self.previous = None
        self.attention_ema = None
        self.write_ema = None
        self.write_gram = None
        self.selection_steps = 0
        self.selected_patches = 0

    def scale_values(self, layer: int, value: torch.Tensor) -> None:
        if layer == 0:
            self.previous = self.selected
        if layer in SPARC_SCALE_LAYERS and self.previous is not None and self.previous.numel():
            # DynamicCache passes the appended value tensor by reference. Like
            # the author implementation, this is a persistent FP16 cache update.
            value[:, :, self.start + self.previous, :] *= SPARC_ALPHA

    def observe(self, module, attention: torch.Tensor, value: torch.Tensor) -> None:
        if attention.shape[0] != 1 or attention.shape[-1] < self.end:
            raise ValueError("SPARC expects one prompt with a complete visual key span")
        visual_attention = attention[0, :, -1, self.start:self.end]
        current = visual_attention.mean(dim=0)
        selected = None
        if self.attention_ema is not None:
            increase = (current - self.attention_ema) / self.attention_ema
            selected = (increase >= SPARC_THRESHOLD).nonzero(as_tuple=True)[0]
            self.selection_steps += 1
            self.selected_patches += selected.numel()
        self.attention_ema = (current if self.attention_ema is None
                              else (1 - SPARC_BETA) * current + SPARC_BETA * self.attention_ema)

        if self.projected_write:
            if self.write_gram is None:
                # The first prediction has no selected history. Reuse its
                # unscaled V so previous boosts cannot create their own signal.
                visual_values = llama.repeat_kv(value[:, :, self.start:self.end, :],
                                                module.num_key_value_groups)[0].contiguous()
                self.write_gram = visual_write_gram(visual_values, module.o_proj.weight)
            writes = visual_write_norms(self.write_gram, visual_attention)
            if self.write_ema is not None:
                if selected.numel():
                    relative_write = (writes - self.write_ema) / self.write_ema.clamp_min(1e-12)
                    selected = relative_write.topk(selected.numel(), sorted=False).indices
                self.write_ema = (1 - SPARC_BETA) * writes + SPARC_BETA * self.write_ema
            else:
                self.write_ema = writes

        self.selected = selected

    def statistics(self) -> dict[str, int]:
        return {"selection_steps": self.selection_steps, "selected_patches": self.selected_patches}


@contextmanager
def sparc_attention(start: int, end: int, *, projected_write: bool = False):
    """Apply SPARC to a single LLaVA trajectory with its own appendable KV cache."""
    if not 0 <= start < end:
        raise ValueError("Expected a nonempty visual key span")
    state = _SparcState(start, end, projected_write)
    original = llama.eager_attention_forward

    def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        layer = int(module.layer_idx)
        state.scale_values(layer, value)
        output, weights = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        if layer == SPARC_SELECTION_LAYER:
            state.observe(module, weights, value)
        return output, weights

    with patch.object(llama, "eager_attention_forward", forward):
        yield state
