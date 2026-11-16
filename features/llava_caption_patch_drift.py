"""Measure target-conditioned visual writes and later FFN preference during replay."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.llama import modeling_llama as llama

from features.llava_sar_ems_attention import SarPatchTerms


WRITE_LAYER = 18
FFN_LAYER = 20
TAU = 0.03


class CaptionPatchDrift:
    def __init__(self, decoder, lm_head):
        self.decoder = decoder
        self.lm_head = lm_head
        self.visual_hpre: dict[int, torch.Tensor] = {}
        self.attention: torch.Tensor | None = None
        self.ffn: torch.Tensor | None = None

    @contextmanager
    def capture_prompt(self, start: int, end: int):
        self.visual_hpre.clear()

        def capture(layer_index):
            def remember(_layer, args, kwargs):
                states = args[0] if args else kwargs["hidden_states"]
                self.visual_hpre[layer_index] = states[0, start:end].contiguous()

            return remember

        with ExitStack() as stack:
            for layer_index in (WRITE_LAYER, FFN_LAYER):
                layer = self.decoder.layers[layer_index]
                hook = layer.register_forward_pre_hook(capture(layer_index), with_kwargs=True)
                stack.callback(hook.remove)
            yield

    @contextmanager
    def capture_tail(self, start: int, end: int, batch_rows, query_rows):
        self.attention = None
        self.ffn = None
        original = llama.eager_attention_forward

        def eager_forward(module, *args, **kwargs):
            output, weights = original(module, *args, **kwargs)
            if module.layer_idx == WRITE_LAYER:
                self.attention = weights.transpose(1, 2)[batch_rows, query_rows, :, start:end].contiguous()
            return output, weights

        def remember_ffn(_module, _args, output):
            self.ffn = output[batch_rows, query_rows]

        with ExitStack() as stack:
            stack.enter_context(patch.object(llama, "eager_attention_forward", eager_forward))
            hook = self.decoder.layers[FFN_LAYER].mlp.register_forward_hook(remember_ffn)
            stack.callback(hook.remove)
            yield

    def score(self, prefix_cache, start: int, end: int, targets) -> list[dict]:
        if self.attention is None or self.ffn is None or len(self.visual_hpre) != 2:
            raise RuntimeError("Could not capture LLaVA image states, layer 18 eager attention, or layer 20 FFN")
        module = self.decoder.layers[WRITE_LAYER].self_attn
        visual_values = prefix_cache.layers[WRITE_LAYER].values[:, :, start:end, :]
        visual_values = llama.repeat_kv(visual_values, module.num_key_value_groups)[0].contiguous()
        terms = SarPatchTerms(visual_values, module.o_proj.weight, self.visual_hpre[WRITE_LAYER],
                              self.decoder.norm, self.lm_head)

        # Each patch caches dot products between the heads' projected writes.
        # A^T G A recovers the squared norm of their sum for each target query.
        heads = self.attention.permute(2, 1, 0).float().contiguous()
        writes = (heads * torch.bmm(terms.gram, heads)).sum(dim=1).T.clamp_min(0).sqrt()
        log_s = terms.log_semantics(targets).T
        write_total = writes.sum(dim=1)
        weighted = writes * (1 + log_s)
        f_log = weighted.sum(dim=1)
        support = (writes * log_s).sum(dim=1) / write_total.clamp_min(1e-12)
        target = weighted / f_log[:, None].clamp_min(1e-12)

        ffn_directions = F.normalize(self.ffn.float(), dim=-1, eps=1e-8)
        visual_directions = F.normalize(self.visual_hpre[FFN_LAYER].float(), dim=-1, eps=1e-8)
        preference = ((ffn_directions @ visual_directions.T) / TAU).softmax(dim=-1)
        hellinger2 = (1 - (target * preference).sqrt().sum(dim=1)).clamp(0, 1)
        return [
            {"write_norm": total, "f_log": score,
             "write_semantic_support": semantic if total > 1e-12 else None,
             "ffn_hellinger2": distance if total > 1e-12 else None}
            for total, score, semantic, distance in torch.stack((write_total, f_log, support, hellinger2), dim=1).tolist()
        ]
