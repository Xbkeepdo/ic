"""Batch the Qwen3 caption interventions for unpadded prompts sharing a visual span."""

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
from features.qwen3_sar_ems_attention import LATE_VARIANTS, VARIANTS


class Qwen3BatchLateAttention:
    def __init__(self, model, span: tuple[int, int], nouns: frozenset[int], variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Qwen3 caption variant {variant!r}")
        if not 0 <= span[0] < span[1]:
            raise ValueError("Expected a nonempty shared visual span")
        self.model = model
        self.decoder = model.model.language_model
        self.head = model.get_output_embeddings()
        self.start, self.end = span
        self.nouns = nouns
        self.variant = variant
        eos = getattr(model.generation_config, "eos_token_id", None)
        self.eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        self.eos_tokens = None
        self.active = None
        self.visual_hpre = None
        self.ffn_patches = None
        self.visual_values = None
        self.attention = None
        self.ffn_residual = None
        self.patches: dict[int, SarPatchTerms] = {}
        self.targets: dict[int, list[tuple[int, int]]] = {}
        self.guidance_calls: dict[int, list[int]] = {}
        self.budget_rows: list[int] = []
        self.budget = None
        self.steps = 0

    def _remember_active_rows(self, _module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            return
        if self.active is None:
            self.active = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            self.eos_tokens = ids.new_tensor(self.eos_ids)
        elif self.steps and self.eos_ids:
            finished = (ids[:, -1, None] == self.eos_tokens).any(dim=-1)
            self.active &= ~finished

    def _remember_visual_states(self, layer_index: int):
        def remember(_module, args, kwargs):
            states = args[0] if args else kwargs["hidden_states"]
            if states.shape[1] < self.end:
                return
            visual = states[:, self.start:self.end].contiguous()
            if layer_index == EARLY_WRITE_LAYER:
                self.visual_hpre = visual
            else:
                self.ffn_patches = F.normalize(visual.float(), dim=-1)

        return remember

    def _remember_ffn_input(self, _module, args):
        self.ffn_residual = args[0][:, -1]

    def _plan_guidance(self, _module, _args, output):
        step = self.steps
        self.steps += 1
        self.budget = None
        self.budget_rows.clear()
        updates = output[:, -1]
        norm = self.decoder.norm
        post = norm(self.ffn_residual + updates)
        tokens = self.head(post).argmax(dim=-1)
        if self.active is not None:
            tokens = tokens.masked_fill(~self.active, -1)

        budgets = []
        layer = self.decoder.layers[EARLY_WRITE_LAYER]
        for row, token in enumerate(tokens.tolist()):
            if token not in self.nouns:
                continue
            gain = self.head.weight[token].float() @ (post[row].float() - norm(self.ffn_residual[row]).float())
            if gain.item() <= 0:
                continue
            patches = self.patches.get(row)
            if patches is None:
                patches = SarPatchTerms(self.visual_values[row], layer.self_attn.o_proj.weight,
                                        self.visual_hpre[row], norm, self.head)
                self.patches[row] = patches
            target = patches.distribution(self.attention[row], token)
            preference = (self.ffn_patches[row] @ F.normalize(updates[row].float(), dim=-1) / TAU).softmax(dim=0)
            drift = (1 - (target * preference).sqrt().sum()).clamp(0, 1)
            deficit = (target - preference).clamp_min(0)
            budget = 5.0 * drift * (gain / 2.0).tanh() * deficit / deficit.amax().clamp_min(1e-6)
            budgets.append(budget)
            self.budget_rows.append(row)
            self.targets.setdefault(row, []).append((step, token))
        if budgets:
            self.budget = torch.stack(budgets)

    def statistics(self, trimmed_responses: list[list[int]]) -> list[dict]:
        """Report events up to each row's actual generated length, including its EOS if retained."""
        if self.variant not in LATE_VARIANTS:
            return [{} for _ in trimmed_responses]
        if (self.attention is None or len(trimmed_responses) != self.attention.shape[0]
                or any(len(tokens) > self.steps for tokens in trimmed_responses)):
            raise RuntimeError("Qwen3 batch attention hooks and generated positions do not align")
        summaries = []
        for row, tokens in enumerate(trimmed_responses):
            targets = [(step, token) for step, token in self.targets.get(row, ()) if step < len(tokens)]
            calls = sum(step < len(tokens) for step in self.guidance_calls.get(row, ()))
            if calls != 2 * len(targets):
                raise RuntimeError(f"Qwen3 batch row {row} has unmatched late attention calls")
            summaries.append({
                "guided_steps": len(targets),
                "target_matches_generated": sum(tokens[step] == token for step, token in targets),
                "guided_attention_calls": calls,
            })
        return summaries

    @contextmanager
    def capture(self):
        if self.variant == "greedy":
            yield
            return
        original = qwen.eager_attention_forward
        sar_ems = self.variant in LATE_VARIANTS

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
                raise RuntimeError("Incomplete Qwen3 batched visual KV span")
            if early:
                if self.variant in {"svar", "svar_sar_ems_attention"}:
                    visual.add_(0.5 * visual.abs().mean(dim=1, keepdim=True))
                else:
                    visual.copy_(sadi_visual_logits(visual))
            if late:
                rows = self.budget_rows
                visual[rows] = rebalance_visual_logits(visual[rows], self.budget)
                for row in rows:
                    self.guidance_calls.setdefault(row, []).append(self.steps - 1)
            attention = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            attention = F.dropout(attention, p=dropout, training=module.training)
            if sar_ems and layer == EARLY_WRITE_LAYER:
                if self.visual_values is None:
                    self.visual_values = value[:, :, self.start:self.end, :].contiguous()
                self.attention = attention[:, :, -1, self.start:self.end].contiguous()
            return (attention @ value).transpose(1, 2).contiguous(), attention

        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, "eager_attention_forward", forward))
            if sar_ems:
                handle = self.model.register_forward_pre_hook(self._remember_active_rows, with_kwargs=True)
                stack.callback(handle.remove)
                for layer_index in (EARLY_WRITE_LAYER, FFN_EDIT_LAYER):
                    layer = self.decoder.layers[layer_index]
                    handle = layer.register_forward_pre_hook(
                        self._remember_visual_states(layer_index), with_kwargs=True,
                    )
                    stack.callback(handle.remove)
                layer = self.decoder.layers[FFN_EDIT_LAYER]
                handle = layer.post_attention_layernorm.register_forward_pre_hook(self._remember_ffn_input)
                stack.callback(handle.remove)
                handle = layer.mlp.register_forward_hook(self._plan_guidance)
                stack.callback(handle.remove)
            yield
