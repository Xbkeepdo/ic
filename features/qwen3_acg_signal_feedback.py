"""Late ACG adaptation after complete SADI, with optional SAR/FFN guidance.

ACG equations: https://arxiv.org/html/2601.13707v2, Algorithm1.
Scope, original SAM-V2 gate, acceptance and signal weights are adaptations.
"""

import torch
from torch.nn import functional as F

from features.qwen3_semantic_donor_feedback import Qwen3BatchSemanticDonor
from features.qwen3_online_risk_f import native_write_norms
from features.qwen3_probe_attention_recycle import normalize_nonnegative


def acg_contexts(scores, values, start, end):
    """One query, original batch shapes; mask visual keys without new Q/K/V."""
    assert scores.ndim == values.ndim == 4 and scores.shape[-2] == 1
    attention = scores.softmax(-1, dtype=torch.float32).to(values.dtype)
    conditional = (attention@values)[:, :, 0]
    masked = scores.clone()
    masked[:, :, :, start:end] = -torch.inf
    assert torch.isfinite(masked).any(-1).all()
    textual = (masked.softmax(-1, dtype=torch.float32).to(values.dtype)@values)[:, :, 0].float()
    difference = conditional.float()-textual
    unit = textual/textual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    perpendicular = difference-(difference*unit).sum(-1, keepdim=True)*unit
    assert torch.isfinite(perpendicular).all()
    return conditional, perpendicular, textual


def sar_ffn_drift(write_norms, log_s, visual_hpre, ffn_update):
    evidence = normalize_nonnegative(write_norms*(1+log_s))
    preference = F.cosine_similarity(visual_hpre.float(), ffn_update.float()[None], dim=-1).div(.03).softmax(-1)
    drift = (1-(evidence*preference).sqrt().sum()).clamp(0, 1)
    return drift


def head_scales(perpendicular, projected_gradient, drift, gamma):
    effect = (perpendicular*projected_gradient).sum(-1)
    negative = (-effect).clamp_min(0)
    beta = negative/negative.amax(-1, keepdim=True).clamp_min(torch.finfo(negative.dtype).tiny)
    scales = float(gamma)*drift[:, None]*beta
    assert torch.isfinite(scales).all() and scales.min() >= 0 and scales.max() <= gamma+1e-6
    predicted = (scales*effect).sum(-1)
    assert (predicted <= 1e-6).all()
    return scales, predicted, effect


class Qwen3BatchACGSignal(Qwen3BatchSemanticDonor):
    def __init__(self, *args, signal_guided=True, gamma=1.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources = self.targets = (31,)
        self.signal_guided, self.gamma = bool(signal_guided), float(gamma)
        assert 0 <= self.gamma < float('inf')

    def _budget(self, source):
        rows = []
        for row, (token, active) in enumerate(zip(self.final_tokens.tolist(), self.active.tolist(), strict=True)):
            if not active or token not in self.nouns or token in self.seen[row]:
                continue
            prefix = self.tokenizer.decode(self.output_history[row], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            assert len(self.output_history[row]) == self.step
            prior, canonical = self.history_parser.classes(prefix), self.token_classes[token]
            blocked = canonical in prior
            self.class_decisions.setdefault(row, []).append(dict(step=self.step, token=token,
                canonical=canonical, prior_classes=sorted(prior), blocked=blocked))
            if not blocked or self.force_probe_gate:
                rows.append(row)
        if not rows:
            return [], None, []
        real = self.probe(self.reader.features(rows, self.final_tokens[rows].tolist()))
        for row, probability in zip(rows, real.tolist(), strict=True):
            self.probe_scores.setdefault(row, []).append(probability)
            self.gate_events.setdefault(row, []).append(dict(step=self.step, token=int(self.final_tokens[row]),
                p_real=probability, threshold=self.real_threshold, positive=probability < self.real_threshold))
        chosen = list(range(len(rows))) if self.force_probe_gate else (real < self.real_threshold).nonzero().flatten().tolist()
        rows, real = [rows[i] for i in chosen], real[chosen]
        if not rows or not self.gamma:
            return [], None, []
        state = self.scratch[source]
        conditional, perpendicular, _ = acg_contexts(state['scores'], state['values'], self.start, self.end)
        self.acg_context = (self.step, conditional)
        perpendicular = perpendicular[rows]
        if self.signal_guided:
            gradient = self._gradient(source, rows).float()
            if source not in self.projections:
                self.projections[source] = self.decoder.layers[source].self_attn.o_proj.weight.float()
            projected = (gradient@self.projections[source]).reshape(len(rows), perpendicular.shape[1], -1)
            drift = []
            for row in rows:
                attention = self.reader.current[source]['attention'][row]
                values = self.visual_values[source][row]
                norms = native_write_norms(values, attention, self.decoder.layers[source].self_attn.o_proj.weight)
                log_s = self.reader._static(source, row)['log_s'][:, self.reader.columns[int(self.final_tokens[row])]]
                drift.append(sar_ffn_drift(norms, log_s, self.reader.visual[source][row], state['update'][row]))
            drift = torch.stack(drift)
            scales, predicted, effects = head_scales(perpendicular, projected, drift, self.gamma)
        else:
            drift = perpendicular.new_zeros(len(rows))
            scales = perpendicular.new_full(perpendicular.shape[:2], self.gamma)
            predicted = effects = None
        budget = scales[:, :, None]*perpendicular
        effective = (budget.abs().amax((1, 2)) > 0).nonzero().flatten().tolist()
        if not effective:
            return [], None, []
        details = []
        for i in effective:
            d = dict(token=int(self.final_tokens[rows[i]]), gain=0., gain_used=False,
                drift=float(drift[i]), budget_max=float(budget[i].abs().max()),
                probe_p_real=float(real[i]), probe_real_threshold=self.real_threshold,
                signal_guided=self.signal_guided, acg_gamma=self.gamma,
                nonzero_heads=int((scales[i]>0).sum()), maximum_head_scale=float(scales[i].max()),
                head_scales=scales[i].tolist())
            if predicted is not None:
                d.update(predicted_log_probability_change=float(predicted[i]), head_effects=effects[i].tolist())
            details.append(d)
        final_rows = [rows[i] for i in effective]
        cache = self.tail_kwargs['past_key_values']
        self.cache_backup = {layer: (cache.layers[layer].keys[final_rows, :, -1:].clone(),
                                    cache.layers[layer].values[final_rows, :, -1:].clone())
                             for layer in range(source+1, len(self.decoder.layers))}
        return final_rows, budget[effective], details

    def _recompute(self, index, original_output, rows, budget):
        if self.diagnostic_zero_replay or budget.ndim == 2:
            return super()._recompute(index, original_output, rows, budget)
        state, layer = self.scratch[index], self.decoder.layers[index]
        assert self.acg_context[0] == self.step
        conditional = self.acg_context[1]
        context = conditional.clone()
        context[rows] = (conditional[rows].float()+budget).to(context.dtype)
        context = context[:, :, None].transpose(1, 2).contiguous().flatten(2)
        attended = state['residual']+layer.self_attn.o_proj(context)
        self._replaying = True
        try:
            corrected = attended+layer.mlp(layer.post_attention_layernorm(attended))
        finally:
            self._replaying = False
        result = original_output.clone()
        result[rows, -1] = corrected[rows, 0]
        return result

    def statistics(self, responses):
        result = super().statistics(responses)
        for stat in result:
            stat.update(intervention_layer=31, replayed_tail_layers=list(range(32, 36)),
                transport_mode='orthogonal_attention_output_contrast', signal_guided=self.signal_guided,
                acg_gamma=self.gamma, acg_scope='Gated layer31 adaptation, not original all-layer ACG.',
                risk_order='risk0..35,I0..35,E_l0..35')
        return result
