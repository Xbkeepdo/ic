"""Port the LLaVA SADI and late SAR/EMS operators to Qwen3-VL text attention."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.llava_decoding_evidence import (
    ATTENTION_LAYERS, EARLY_WRITE_LAYER, FFN_EDIT_LAYER, SAR_EMS_ATTENTION_LAYERS, TAU,
)
from features.llava_sadi import sadi_visual_logits
from features.llava_sar_ems_attention import SarPatchTerms, rebalance_visual_logits


VQA_OBJECT_VARIANT = "sadi_sar_ems_vqa_object"
LATE_VARIANTS = ("sadi_sar_ems_attention", "svar_sar_ems_attention")
VARIANTS = ("greedy", "svar", "sadi", *LATE_VARIANTS)


class Qwen3LateAttention:
    def __init__(self, model, span: tuple[int, int], nouns: frozenset[int], variant: str, *,
                 object_token: int | None = None, yes_no_ids: tuple[int, int] | None = None):
        self.decoder = model.model.language_model
        self.head = model.get_output_embeddings()
        self.start, self.end = span
        self.nouns = nouns
        self.patches = None
        self.visual_hpre = None
        self.ffn_patches = None
        self.visual_values = None
        self.yes_no_readout = None
        if yes_no_ids is not None:
            yes, no = map(int, yes_no_ids)
            vocab_size = self.head.weight.shape[0]
            if yes == no or not (0 <= yes < vocab_size and 0 <= no < vocab_size):
                raise ValueError("Expected distinct Yes and No token IDs in that order")
            weight = self.head.weight.detach()
            self.yes_no_readout = weight[yes].float() - weight[no].float()
        self.reset(variant, object_token=object_token)

    def reset(self, variant: str, object_token: int | None = None) -> None:
        """Begin a question or variant while keeping the current image's visual terms."""
        if variant not in VARIANTS and variant != VQA_OBJECT_VARIANT:
            raise ValueError(f"Unknown Qwen3 attention variant {variant!r}")
        if variant == VQA_OBJECT_VARIANT and object_token is not None:
            if self.yes_no_readout is None:
                raise ValueError("The object-conditioned variant requires Yes and No token IDs")
            if not 0 <= int(object_token) < self.head.weight.shape[0]:
                raise ValueError("Question object token ID is outside the model vocabulary")
        self.variant = variant
        self.object_token = None if object_token is None else int(object_token)
        self.targets: list[tuple[int, int]] = []
        self.steps = 0
        self.attention = None
        self.ffn_residual = None
        self.budget = None
        self.guided_attention_calls = 0
        self.draft_yes_no_margin = None
        self.ffn_yes_no_gain = None

    def _remember_visual_states(self, layer_index: int):
        def remember(_module, args, kwargs):
            states = args[0] if args else kwargs["hidden_states"]
            if states.shape[1] < self.end:
                return
            visual = states[0, self.start:self.end].contiguous()
            if layer_index == EARLY_WRITE_LAYER:
                self.visual_hpre = visual
            else:
                self.ffn_patches = F.normalize(visual.float(), dim=-1)

        return remember

    def _register_visual_hooks(self, stack: ExitStack) -> None:
        for layer_index in (EARLY_WRITE_LAYER, FFN_EDIT_LAYER):
            layer = self.decoder.layers[layer_index]
            handle = layer.register_forward_pre_hook(self._remember_visual_states(layer_index), with_kwargs=True)
            stack.callback(handle.remove)

    @contextmanager
    def capture_visual_prefix(self):
        """Capture visual hpre during a neutral multimodal prefill for the current image."""
        self.visual_hpre = None
        self.ffn_patches = None
        self.visual_values = None
        self.patches = None
        with ExitStack() as stack:
            self._register_visual_hooks(stack)
            yield

    def use_visual_prefix_cache(self, past_key_values) -> None:
        """Save the current image's layer-18 visual values in query-head order."""
        if self.visual_hpre is None or self.ffn_patches is None:
            raise RuntimeError("The visual prefix did not capture both Qwen3 visual hidden states")
        module = self.decoder.layers[EARLY_WRITE_LAYER].self_attn
        values = past_key_values.layers[EARLY_WRITE_LAYER].values
        if values is None or values.ndim != 4 or values.shape[0] != 1 or values.shape[-2] < self.end:
            raise ValueError("The Qwen3 prefix cache does not contain the complete visual span")
        visual = values[:, :, self.start:self.end, :].clone(memory_format=torch.contiguous_format)
        self.visual_values = qwen.repeat_kv(visual, module.num_key_value_groups)[0].contiguous()
        self.patches = None

    def _remember_ffn_input(self, _module, args):
        self.ffn_residual = args[0][0, -1]

    def _plan_guidance(self, _module, _args, output):
        self.budget = None
        step = self.steps
        self.steps += 1
        vqa = self.variant == VQA_OBJECT_VARIANT
        if vqa and (step != 0 or self.object_token is None):
            return
        update = output[0, -1]
        norm = self.decoder.norm
        post = norm(self.ffn_residual + update)
        if vqa:
            token = self.object_token
            post_margin = self.yes_no_readout @ post.float()
            gain = post_margin - self.yes_no_readout @ norm(self.ffn_residual).float()
            self.draft_yes_no_margin, self.ffn_yes_no_gain = torch.stack((post_margin, gain)).tolist()
            if not (self.draft_yes_no_margin > 0.0 and self.ffn_yes_no_gain > 0.0):
                return
        else:
            token = int(self.head(post).argmax())
            if token not in self.nouns:
                return
            gain = self.head.weight[token].float() @ (post.float() - norm(self.ffn_residual).float())
            if gain.item() <= 0:
                return
        if self.patches is None:
            layer = self.decoder.layers[EARLY_WRITE_LAYER]
            self.patches = SarPatchTerms(self.visual_values, layer.self_attn.o_proj.weight,
                                        self.visual_hpre, norm, self.head)
        target = self.patches.distribution(self.attention, token)
        preference = (self.ffn_patches @ F.normalize(update.float(), dim=-1) / TAU).softmax(dim=0)
        drift = (1 - (target * preference).sqrt().sum()).clamp(0, 1)
        deficit = (target - preference).clamp_min(0)
        self.budget = 5.0 * drift * (gain / 2.0).tanh() * deficit / deficit.amax().clamp_min(1e-6)
        self.targets.append((step, token))

    def statistics(self, tokens: list[int]) -> dict:
        if self.steps != len(tokens) or self.guided_attention_calls != 2 * len(self.targets):
            raise RuntimeError("Qwen3 attention hooks and generated token positions do not align")
        if self.variant == VQA_OBJECT_VARIANT:
            return {
                "guided_steps": len(self.targets),
                "guided_attention_calls": self.guided_attention_calls,
                "object_token_id": self.object_token,
                "draft_yes_no_margin": self.draft_yes_no_margin,
                "ffn_yes_no_gain": self.ffn_yes_no_gain,
            }
        return {
            "guided_steps": len(self.targets),
            "target_matches_generated": sum(tokens[step] == token for step, token in self.targets),
            "guided_attention_calls": self.guided_attention_calls,
        }

    @contextmanager
    def capture(self):
        if self.variant == "greedy":
            yield
            return
        original = qwen.eager_attention_forward
        sar_ems = self.variant in LATE_VARIANTS or self.variant == VQA_OBJECT_VARIANT

        def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if not isinstance(module, qwen.Qwen3VLTextAttention):
                return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            layer = module.layer_idx
            early = layer in ATTENTION_LAYERS
            late = sar_ems and layer in SAR_EMS_ATTENTION_LAYERS and self.budget is not None
            if not (early or late):
                return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            key = qwen.repeat_kv(key, module.num_key_value_groups)
            value = qwen.repeat_kv(value, module.num_key_value_groups)
            scores = (query @ key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., :key.shape[-2]]
            visual = scores[:, :, -1, self.start:self.end]
            if visual.shape[-1] != self.end - self.start:
                raise RuntimeError("Incomplete Qwen3 visual KV span")
            if early:
                if self.variant in {"svar", "svar_sar_ems_attention"}:
                    visual.add_(0.5 * visual.abs().mean(dim=1, keepdim=True))
                else:
                    visual.copy_(sadi_visual_logits(visual))
            if late:
                visual.copy_(rebalance_visual_logits(visual, self.budget[None, :]))
                self.guided_attention_calls += 1
            attention = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            attention = F.dropout(attention, p=dropout, training=module.training)
            if sar_ems and layer == EARLY_WRITE_LAYER:
                if self.visual_values is None:
                    self.visual_values = value[0, :, self.start:self.end, :].contiguous()
                self.attention = attention[0, :, -1, self.start:self.end].contiguous()
            return (attention @ value).transpose(1, 2).contiguous(), attention

        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, "eager_attention_forward", forward))
            if sar_ems:
                if self.visual_hpre is None or self.ffn_patches is None:
                    self._register_visual_hooks(stack)
                layer = self.decoder.layers[FFN_EDIT_LAYER]
                handle = layer.post_attention_layernorm.register_forward_pre_hook(self._remember_ffn_input)
                stack.callback(handle.remove)
                handle = layer.mlp.register_forward_hook(self._plan_guidance)
                stack.callback(handle.remove)
            yield
