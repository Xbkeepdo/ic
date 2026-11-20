"""Adapt text-route suppression using visual support and FFN spatial drift.

CRG (arXiv:2605.24024) motivates signed visual/text route selection.
AttnReal (arXiv:2503.08342) motivates acting on generated-history attention.
This is a local single-layer adaptation, not either paper's full method.
"""

import torch
from torch.nn import functional as F

from features.llava_decoding_evidence import TAU
from features.qwen3_sar_ems_head_gradient import Qwen3BatchHeadGradient
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchHistoryRoute(Qwen3BatchHeadGradient):
    def __init__(self, model, span, nouns, *, use_ems=True, **kwargs):
        super().__init__(model, span, nouns, final_gradient=True, **kwargs)
        self.use_ems = use_ems
        self.prompt_length = None

    def _begin_step(self, module, args, kwargs):
        super()._begin_step(module, args, kwargs)
        if self.step == 0:
            ids = kwargs.get('input_ids', args[0] if args else None)
            self.prompt_length = ids.shape[-1]

    def _budget(self, source):
        state = self.scratch[source]
        rows = [i for i, (token, active) in enumerate(
            zip(self.final_tokens.tolist(), self.active.tolist(), strict=True))
            if active and token in self.nouns]
        if not rows or state['values'].shape[-2] <= self.prompt_length:
            return [], None, []
        visual, gain, _, _ = readout_contrast(self, source, rows, self.final_tokens[rows, None])
        weak = (visual[:, 0] < .35).nonzero().flatten().tolist()
        if not weak:
            return [], None, []
        rows = [rows[i] for i in weak]
        gradient = self._gradient(source, rows).float()
        values = self.visual_values[source][rows].float()
        projected = (gradient @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        attention = state['scores'].softmax(-1, dtype=torch.float32).to(state['values'].dtype)
        visual_attention = state['attention'][rows].float()
        effect = torch.einsum('bhmd,bhd->bhm', values, projected)
        visual_effect = (visual_attention * effect).sum(-1)
        history_context = (attention[:, :, :, self.prompt_length:]
                           @ state['values'][:, :, self.prompt_length:])[:, :, 0]
        history_effect = (history_context[rows].float() * projected).sum(-1)
        conflict = (visual_effect < 0) & (history_effect > 0)
        reliance = visual_effect.abs() / (visual_effect.abs() + history_effect.abs()).clamp_min(1e-12)
        ranks = reliance.masked_fill(~conflict, float('inf')).argsort(-1).argsort(-1)
        selected = conflict & (ranks < 4)
        negative = (-visual_attention * effect).clamp_min(0).sum(1)
        evidence = negative / negative.sum(-1, keepdim=True).clamp_min(1e-12)
        similarity = (self.visual_normalized[source][rows]
                      @ F.normalize(state['update'][rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
        # Matched comparison: only the EMS-dependent part differs.
        amount = .5 + .5 * drift if self.use_ems else drift.new_full(drift.shape, .5)
        budget = -selected.float() * amount[:, None]
        predicted = (budget * history_effect).sum(-1)
        assert torch.isfinite(budget).all() and (predicted <= 0).all()
        effective = selected.any(-1).nonzero().flatten().tolist()
        if not effective:
            return [], None, []
        details = [dict(token=int(self.final_tokens[rows[i]]), gain=float(gain[weak[i], 0]),
                        drift=float(drift[i]), budget_max=float(amount[i]),
                        selected_history_heads=int(selected[i].sum()),
                        predicted_log_probability_change=float(predicted[i]),
                        previously_generated_token=int(self.final_tokens[rows[i]]) in self.seen[rows[i]])
                   for i in effective]
        cache = self.tail_kwargs['past_key_values']
        self.cache_backup = {
            layer: (cache.layers[layer].keys[rows, :, -1:].clone(),
                    cache.layers[layer].values[rows, :, -1:].clone())
            for layer in range(32, len(self.decoder.layers))}
        return [rows[i] for i in effective], budget[effective, :, None], details

    def _recompute(self, index, original_output, rows, budget):
        if budget.ndim == 2:
            return super()._recompute(index, original_output, rows, budget)
        state, layer = self.scratch[index], self.decoder.layers[index]
        attention = state['scores'].softmax(-1, dtype=torch.float32).to(state['values'].dtype)
        # Post-softmax route gating: visual and prompt components stay fixed.
        context = (attention @ state['values'])[:, :, 0]
        history = (attention[:, :, :, self.prompt_length:]
                   @ state['values'][:, :, self.prompt_length:])[:, :, 0]
        context[rows] = (context[rows].float() + budget * history[rows].float()).to(context.dtype)
        context = context[:, :, None].transpose(1, 2).contiguous().flatten(2)
        attended = state['residual'] + layer.self_attn.o_proj(context)
        self._replaying = True
        try:
            corrected = attended + layer.mlp(layer.post_attention_layernorm(attended))
        finally:
            self._replaying = False
        result = original_output.clone()
        result[rows, -1] = corrected[rows, 0]
        return result

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [e for e in self.events.get(row, ()) if e['step'] < len(tokens)]
            repeated = [e for e in events if e.get('previously_generated_token', False)]
            stats.update(history_route_gating=True, use_ems=self.use_ems,
                         prompt_length=self.prompt_length,
                         selected_history_head_events=sum(e.get('selected_history_heads', 0) for e in events),
                         repeated_object_proposals=len(repeated),
                         repeated_object_accepted=sum(e.get('accepted', False) for e in repeated),
                         repeated_object_token_changes=sum(tokens[e['step']] != e['token'] for e in repeated))
        return result
