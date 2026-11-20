"""Gate history-to-image attention transfer with a frozen I/E_l detector."""

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from features.llava_decoding_evidence import TAU
from features.qwen3_pre_sadi_i_el import PreSadiIElCapture
from features.qwen3_sar_ems_head_gradient import Qwen3BatchHeadGradient
from features.qwen3_visual_write_contrast import readout_contrast


def normalize_nonnegative(values):
    """Normalize without discarding small but nonzero gradient magnitudes."""
    tiny = torch.finfo(values.dtype).tiny
    scaled = values / values.amax(-1, keepdim=True).clamp_min(tiny)
    return scaled / scaled.sum(-1, keepdim=True).clamp_min(tiny)


class Qwen3BatchProbeAttentionRecycle(Qwen3BatchHeadGradient):
    def __init__(self, model, span, allowed_tokens, *, probe, semantic_tokens,
                 cached_semantics=None, guided=True, diagnostic_force_probe_gate=False, **kwargs):
        super().__init__(model, span, allowed_tokens, final_gradient=True, **kwargs)
        self.probe = probe
        self.guided = guided
        self.force_probe_gate = diagnostic_force_probe_gate
        self.reader = PreSadiIElCapture(model, span, semantic_tokens, cached_semantics=cached_semantics)
        self.prompt_length = None
        self.probe_scores = {}

    def _begin_step(self, module, args, kwargs):
        super()._begin_step(module, args, kwargs)
        if self.step == 0:
            self.prompt_length = kwargs.get('input_ids', args[0] if args else None).shape[1]

    def _budget(self, source):
        state = self.scratch[source]
        rows = [i for i, (token, active) in enumerate(
            zip(self.final_tokens.tolist(), self.active.tolist(), strict=True))
            if active and token in self.nouns and token not in self.seen[i]]
        if not rows:
            return [], None, []
        targets = self.final_tokens[rows].tolist()
        real = self.probe(self.reader.features(rows, targets))
        for row, probability in zip(rows, real.tolist(), strict=True):
            self.probe_scores.setdefault(row, []).append(probability)
        selected = list(range(len(rows))) if self.force_probe_gate else (real < .5).nonzero().flatten().tolist()
        if not selected:
            return [], None, []
        rows = [rows[i] for i in selected]
        real = real[selected]
        visual_support, gain, _, _ = readout_contrast(self, source, rows, self.final_tokens[rows, None])
        gradient = self._gradient(source, rows).float()
        values = state['values'][rows].float()
        projected = (gradient @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        effect = torch.einsum('bhkd,bhd->bhk', values, projected)
        actual = state['scores'][rows, :, 0].softmax(-1, dtype=torch.float32).to(state['values'].dtype).float()
        visual = actual[:, :, self.start:self.end]
        negative = (-visual * effect[:, :, self.start:self.end]).clamp_min(0)
        valid_head = negative.amax(-1, keepdim=True) > torch.finfo(negative.dtype).tiny
        evidence = normalize_nonnegative(negative.sum(1))
        similarity = (self.visual_normalized[source][rows]
                      @ F.normalize(state['update'][rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
        amount = .5 * drift
        history = actual[:, :, self.prompt_length:]
        positive = effect[:, :, self.prompt_length:] > 0
        movable = positive & valid_head
        removed = amount[:, None, None] * history * movable
        mass = removed.sum(-1, keepdim=True)
        if self.guided:
            deficit = (evidence - preference).clamp_min(0)
            spatial = .25 + .75 * deficit / deficit.amax(-1, keepdim=True).clamp_min(1e-12)
            destination = normalize_nonnegative(normalize_nonnegative(negative) * spatial[:, None])
        else:
            destination = torch.full_like(negative, 1 / (self.end - self.start))
        delta = torch.zeros_like(actual)
        delta[:, :, self.start:self.end] = mass * destination
        delta[:, :, self.prompt_length:] -= removed
        predicted = (delta * effect).sum((1, 2))
        assert torch.isfinite(delta).all()
        assert delta.sum(-1).abs().max() < 1e-6
        if self.guided:
            assert (predicted <= 1e-6).all()
        effective = (delta.abs().amax((1, 2)) > 0).nonzero().flatten().tolist()
        if not effective:
            return [], None, []
        details = [dict(token=int(self.final_tokens[rows[i]]), gain=float(gain[i, 0]),
            drift=float(drift[i]), budget_max=float(delta[i].abs().max()),
            probe_p_real=float(real[i]), visual_support=float(visual_support[i, 0]),
            moved_attention_mass=float(mass[i].sum()), predicted_log_probability_change=float(predicted[i]))
            for i in effective]
        cache = self.tail_kwargs['past_key_values']
        self.cache_backup = {
            layer: (cache.layers[layer].keys[rows, :, -1:].clone(),
                    cache.layers[layer].values[rows, :, -1:].clone())
            for layer in range(32, len(self.decoder.layers))}
        return [rows[i] for i in effective], delta[effective], details

    def _recompute(self, index, original_output, rows, budget):
        if budget.ndim == 2:
            return super()._recompute(index, original_output, rows, budget)
        state, layer = self.scratch[index], self.decoder.layers[index]
        attention = state['scores'].softmax(-1, dtype=torch.float32).to(state['values'].dtype)
        updated = attention[rows, :, 0].float() + budget
        assert updated.min() >= -1e-7
        attention[rows, :, 0] = updated.clamp_min(0).to(attention.dtype)
        context = (attention @ state['values']).transpose(1, 2).contiguous().flatten(2)
        attended = state['residual'] + layer.self_attn.o_proj(context)
        self._replaying = True
        try:
            corrected = attended + layer.mlp(layer.post_attention_layernorm(attended))
        finally:
            self._replaying = False
        result = original_output.clone()
        result[rows, -1] = corrected[rows, 0]
        return result

    @contextmanager
    def capture(self):
        with super().capture():
            if self.enabled:
                with self.reader.capture():
                    yield self
            else:
                yield self

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, stats in enumerate(result):
            probabilities = self.probe_scores.get(row, ())
            stats.update(probe_gated_recycle=True, spatial_guidance=self.guided,
                         diagnostic_force_probe_gate=self.force_probe_gate,
                         probe_calls=len(probabilities), probe_positive_calls=sum(p < .5 for p in probabilities),
                         first_token_only=True, probe_real_threshold=.5)
        return result
