"""Layer-local SAR/EMS feedback on a fixed SADI base for Qwen3-VL.

Layer indices are zero based. ``next`` uses 18 -> 19 and 19 -> 20.
``same`` probes layer 19/20, then recomputes that layer's last attention
query and FFN once. It never invokes self-attention twice or mutates KV.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.llava_decoding_evidence import ATTENTION_LAYERS, TAU
from features.llava_sadi import sadi_visual_logits
from features.llava_sar_ems_attention import SarPatchTerms, rebalance_visual_logits


class Qwen3BatchLayerFeedback:
    """Use one independent budget per target layer, during the same token step.

    ``disable_guidance`` and zero ``strength`` are exact no-op controls relative
    to the existing SADI batch controller. ``force_guidance`` is diagnostic
    only: it bypasses noun/positive-gain gates and is labelled in statistics.
    """

    def __init__(self, model, span, nouns, mode="next", strength=5.0, *,
                 disable_guidance=False, force_guidance=False):
        if mode not in ("next", "same"):
            raise ValueError(f"Unknown layer feedback mode {mode!r}")
        if not 0 <= span[0] < span[1]:
            raise ValueError("Expected a nonempty shared visual span")
        if not 0 <= strength < float("inf"):
            raise ValueError("strength must be finite and nonnegative")
        self.model = model
        self.decoder = model.model.language_model
        self.head = model.get_output_embeddings()
        self.start, self.end = span
        self.nouns = nouns
        self.mode = mode
        self.strength = float(strength)
        self.force_guidance = bool(force_guidance)
        self.enabled = bool(strength and not disable_guidance and (nouns or force_guidance))
        self.targets = (19, 20)
        self.sources = (18, 19) if mode == "next" else (19, 20)
        eos = getattr(model.generation_config, "eos_token_id", None)
        self.eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        self.active = None
        self.eos_tokens = None
        self.steps = 0
        self.step = -1
        self.visual_hpre = {}
        self.visual_normalized = {}
        self.visual_values = {}
        self.patch_terms = {}
        self.scratch = {}
        self.pending = {}
        self.events = {}
        self._replaying = False

    def _begin_step(self, _module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise RuntimeError("Layer feedback expects input_ids on every generation forward")
        if self.active is None:
            self.active = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            self.eos_tokens = ids.new_tensor(self.eos_ids)
        elif self.steps and self.eos_ids:
            self.active &= ~(ids[:, -1, None] == self.eos_tokens).any(dim=-1)
        if self.pending:
            raise RuntimeError("A previous generation step left an unapplied budget")
        self.scratch.clear()
        self.step = self.steps
        self.steps += 1

    def _remember_input(self, index):
        def remember(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self.scratch[index] = {"residual": hidden[:, -1:].contiguous()}
            if hidden.shape[1] >= self.end:
                # Only the current final text query is modified. The image
                # prefix is causal and may be cached throughout generation.
                visual = hidden[:, self.start:self.end].contiguous()
                self.visual_hpre[index] = visual
                self.visual_normalized[index] = F.normalize(visual.float(), dim=-1)
        return remember

    def _remember_ffn_input(self, index):
        def remember(_module, args):
            if not self._replaying:
                self.scratch[index]["ffn_residual"] = args[0][:, -1].contiguous()
        return remember

    def _remember_ffn_output(self, index):
        def remember(_module, _args, output):
            if not self._replaying:
                self.scratch[index]["update"] = output[:, -1].contiguous()
        return remember

    def _budget(self, source):
        state = self.scratch[source]
        updates, residual = state["update"], state["ffn_residual"]
        norm = self.decoder.norm
        post = norm(residual + updates)
        tokens = self.head(post).argmax(dim=-1).masked_fill(~self.active, -1)
        before = norm(residual)
        rows, budgets, details = [], [], []
        layer = self.decoder.layers[source]
        for row, token in enumerate(tokens.tolist()):
            if token < 0 or (token not in self.nouns and not self.force_guidance):
                continue
            gain = self.head.weight[token].float() @ (post[row].float() - before[row].float())
            if self.force_guidance:
                gain = gain.clamp_min(1.0)
            elif gain.item() <= 0:
                continue
            cache_key = (source, row)
            terms = self.patch_terms.get(cache_key)
            if terms is None:
                terms = SarPatchTerms(
                    self.visual_values[source][row], layer.self_attn.o_proj.weight,
                    self.visual_hpre[source][row], norm, self.head,
                )
                self.patch_terms[cache_key] = terms
            target = terms.distribution(state["attention"][row], token)
            preference = (
                self.visual_normalized[source][row]
                @ F.normalize(updates[row].float(), dim=-1) / TAU
            ).softmax(dim=0)
            drift = (1 - (target * preference).sqrt().sum()).clamp(0, 1)
            deficit = (target - preference).clamp_min(0)
            budget = self.strength * drift * (gain / 2.0).tanh() * deficit / deficit.amax().clamp_min(1e-6)
            maximum = budget.max().item()
            if maximum <= 0:
                continue
            if not torch.isfinite(budget).all().item():
                raise RuntimeError(f"Non-finite feedback budget at layer {source}")
            rows.append(row)
            budgets.append(budget)
            details.append({"token": token, "gain": gain.item(), "drift": drift.item(),
                            "budget_max": maximum})
        return rows, torch.stack(budgets) if budgets else None, details

    def _record(self, source, target, rows, details):
        for row, detail in zip(rows, details, strict=True):
            self.events.setdefault(row, []).append({
                "step": self.step, "source_layer": source, "target_layer": target, **detail,
            })

    def _recompute(self, index, original_output, rows, budget):
        """Recompute one query for the complete batch; replace only guided rows.

        Query/key/value projections and rotary/KV update are already complete.
        Their inputs do not depend on the current layer's attention output.
        Reusing those exact tensors is therefore causal and cache-safe.
        """
        state = self.scratch[index]
        layer = self.decoder.layers[index]
        scores = state["scores"].clone()
        visual = scores[:, :, 0, self.start:self.end]
        visual[rows] = rebalance_visual_logits(visual[rows], budget)
        attention = scores.softmax(dim=-1, dtype=torch.float32).to(state["values"].dtype)
        context = (attention @ state["values"]).transpose(1, 2).contiguous()
        context = context.reshape(context.shape[0], 1, -1)
        # Preserve batch size during both projections. Calling only selected
        # rows could add an unrelated batch-shape numerical change.
        attended = state["residual"] + layer.self_attn.o_proj(context)
        self._replaying = True
        try:
            corrected = attended + layer.mlp(layer.post_attention_layernorm(attended))
        finally:
            self._replaying = False
        output = original_output.clone()
        output[rows, -1] = corrected[rows, 0]
        return output

    def _after_layer(self, index):
        def finish(_module, _args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise RuntimeError("Expected the installed Qwen3 decoder layer tensor output")
            rows, budget, details = self._budget(index)
            if not rows:
                self.scratch.pop(index)
                return output
            target = index + 1 if self.mode == "next" else index
            if self.mode == "next":
                self.pending[target] = (index, rows, budget, details)
            else:
                output = self._recompute(index, output, rows, budget)
                self._record(index, target, rows, details)
            self.scratch.pop(index)
            return output
        return finish

    def statistics(self, trimmed_responses):
        if self.active is None or len(trimmed_responses) != self.active.shape[0]:
            raise RuntimeError("Generation rows do not match controller rows")
        if self.pending or any(len(tokens) > self.steps for tokens in trimmed_responses):
            raise RuntimeError("Generation steps and guidance events do not align")
        results = []
        for row, tokens in enumerate(trimmed_responses):
            events = [event for event in self.events.get(row, ()) if event["step"] < len(tokens)]
            guided_steps = {event["step"] for event in events}
            matches = [event for event in events if tokens[event["step"]] == event["token"]]
            by_layer = {}
            for source, target in zip(self.sources, self.targets, strict=True):
                selected = [event for event in events if event["target_layer"] == target]
                count = len(selected)
                by_layer[str(target)] = {
                    "source_layer": source, "target_layer": target, "guided_steps": count,
                    "target_matches_generated": sum(tokens[e["step"]] == e["token"] for e in selected),
                    **{f"mean_{key}": sum(e[key] for e in selected) / count if count else 0.0
                       for key in ("budget_max", "drift", "gain")},
                }
            results.append({
                "guided_steps": len(guided_steps), "guided_attention_calls": len(events),
                "target_matches_generated": len({event["step"] for event in matches}),
                "target_match_events": len(matches), "by_layer": by_layer,
                "local_recompute_events": len(events) if self.mode == "same" else 0,
                "diagnostic_force_guidance": self.force_guidance,
            })
        return results

    @contextmanager
    def capture(self):
        if self.model.training:
            raise RuntimeError("Layer feedback is an inference-only controller")
        for index in {*ATTENTION_LAYERS, *self.sources, *self.targets}:
            if self.decoder.layers[index].self_attn.config._attn_implementation != "eager":
                raise RuntimeError("Layer feedback requires eager Qwen3 attention")
        original = qwen.eager_attention_forward

        def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if not isinstance(module, qwen.Qwen3VLTextAttention):
                return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            index = module.layer_idx
            early = index in ATTENTION_LAYERS
            source = self.enabled and index in self.sources
            guided = self.enabled and index in self.pending
            if not (early or source or guided):
                return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if dropout or module.training:
                raise RuntimeError("Layer feedback requires deterministic inference attention")
            key = qwen.repeat_kv(key, module.num_key_value_groups)
            value = qwen.repeat_kv(value, module.num_key_value_groups)
            scores = (query @ key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., :key.shape[-2]]
            visual = scores[:, :, -1, self.start:self.end]
            if visual.shape[-1] != self.end - self.start:
                raise RuntimeError("Incomplete visual KV span in layer feedback")
            if early:
                visual.copy_(sadi_visual_logits(visual))
            if guided:
                origin, rows, budget, details = self.pending.pop(index)
                visual[rows] = rebalance_visual_logits(visual[rows], budget)
                self._record(origin, index, rows, details)
            attention = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
            if source:
                if index not in self.visual_values:
                    self.visual_values[index] = value[:, :, self.start:self.end].contiguous()
                self.scratch[index]["attention"] = attention[:, :, -1, self.start:self.end].contiguous()
                if self.mode == "same":
                    self.scratch[index]["scores"] = scores[:, :, -1:].clone()
                    self.scratch[index]["values"] = value
            return (attention @ value).transpose(1, 2).contiguous(), attention

        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, "eager_attention_forward", forward))
            handle = self.model.register_forward_pre_hook(self._begin_step, with_kwargs=True)
            stack.callback(handle.remove)
            if self.enabled:
                for index in self.sources:
                    layer = self.decoder.layers[index]
                    handle = layer.register_forward_pre_hook(self._remember_input(index), with_kwargs=True)
                    stack.callback(handle.remove)
                    handle = layer.post_attention_layernorm.register_forward_pre_hook(self._remember_ffn_input(index))
                    stack.callback(handle.remove)
                    handle = layer.mlp.register_forward_hook(self._remember_ffn_output(index))
                    stack.callback(handle.remove)
                    handle = layer.register_forward_hook(self._after_layer(index))
                    stack.callback(handle.remove)
            yield
