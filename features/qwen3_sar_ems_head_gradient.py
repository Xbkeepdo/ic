"""Guide visual attention per head using target effects and FFN spatial drift."""

import torch
from torch.nn import functional as F

from features.llava_decoding_evidence import TAU
from features.qwen3_sar_ems_final_target import Qwen3BatchFinalTargetReplay
from features.qwen3_visual_write_contrast import readout_contrast


class ReadOnlyTailCache:
    """Build differentiable current keys/values against a frozen cached prefix."""

    def __init__(self, cache):
        self.cache = cache

    def update(self, keys, values, layer_idx, cache_kwargs):
        old = self.cache.layers[layer_idx]
        position = cache_kwargs['cache_position']
        assert position.numel() == 1 and keys.shape[-2] == 1
        assert old.keys.shape[-2] == int(position.item()) + 1
        return (torch.cat((old.keys[:, :, :-1].detach().clone(), keys), dim=-2),
                torch.cat((old.values[:, :, :-1].detach().clone(), values), dim=-2))


class Qwen3BatchHeadGradient(Qwen3BatchFinalTargetReplay):
    def __init__(self, model, span, nouns, *, final_gradient=True,
                 disable_guidance=False, diagnostic_zero_replay=False,
                 diagnostic_reject_all=False):
        super().__init__(model, span, nouns, disable_guidance=disable_guidance,
                         diagnostic_zero_replay=diagnostic_zero_replay)
        self.final_gradient = final_gradient
        self.diagnostic_reject_all = diagnostic_reject_all
        self.reference_hidden = None
        self.probe_rows = []
        self.probe_before = None
        self.gradient_probes = 0
        self.maximum_probe_hidden_difference = 0.
        self.cache_backup = None

    def _gradient(self, source, rows):
        self.replaying_tail = self._replaying = True
        try:
            with torch.inference_mode(False), torch.enable_grad():
                residual = self.scratch[source]['ffn_residual'].detach().clone().requires_grad_(True)
                layer = self.decoder.layers[source]
                current = residual[:, None] + layer.mlp(layer.post_attention_layernorm(residual[:, None]))
                if self.final_gradient:
                    kwargs = dict(self.tail_kwargs)
                    kwargs['past_key_values'] = ReadOnlyTailCache(kwargs['past_key_values'])
                    kwargs['position_embeddings'] = tuple(x.detach().clone() for x in kwargs['position_embeddings'])
                    kwargs['cache_position'] = kwargs['cache_position'].detach().clone()
                    for tail in self.decoder.layers[32:]:
                        current = tail(current, **kwargs)
                normalized = self.decoder.norm(current)
                logits = self.head(normalized).float()[:, 0]
                target = self.final_tokens[rows].detach().clone()
                log_probability = logits.log_softmax(-1)[rows, target]
                gradient, = torch.autograd.grad(log_probability.sum(), residual)
                gradient = gradient[rows].detach()
                if self.final_gradient:
                    difference = (normalized.detach() - self.reference_hidden).abs().max().item()
                    self.maximum_probe_hidden_difference = max(self.maximum_probe_hidden_difference, difference)
                reference = self.head(self.reference_hidden).float()[:, 0].log_softmax(-1)
                self.probe_before = reference[rows, target].detach()
                self.probe_rows = rows
                self.gradient_probes += 1
        finally:
            self.replaying_tail = self._replaying = False
        if not torch.isfinite(gradient).all().item():
            raise RuntimeError('Non-finite target gradient')
        return gradient

    def _budget(self, source):
        state = self.scratch[source]
        top = self.final_tokens.tolist()
        rows = [i for i, (token, active) in enumerate(zip(top, self.active.tolist(), strict=True))
                if active and token in self.nouns and token not in self.seen[i]]
        if not rows:
            return [], None, []
        targets = self.final_tokens[rows, None]
        visual, gain, _, _ = readout_contrast(self, source, rows, targets)
        weak = (visual[:, 0] < .35).nonzero().flatten().tolist()
        if not weak:
            return [], None, []
        rows = [rows[i] for i in weak]
        support, gain = visual[weak, 0], gain[weak, 0]
        gradient = self._gradient(source, rows).float()
        values = self.visual_values[source][rows].float()
        projected = (gradient @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        effect = torch.einsum('bhmd,bhd->bhm', values, projected)
        attention = state['attention'][rows].float()
        conditional = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        centered = effect - (conditional * effect).sum(-1, keepdim=True)
        derivative = attention * centered
        negative = (-attention * effect).clamp_min(0).sum(1)
        total = negative.sum(-1, keepdim=True)
        evidence = negative / total.clamp_min(1e-12)
        similarity = (self.visual_normalized[source][rows]
                      @ F.normalize(state['update'][rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
        deficit = (evidence - preference).clamp_min(0)
        spatial = .25 + .75 * deficit / deficit.amax(-1, keepdim=True).clamp_min(1e-6)
        negative_share = total[:, 0] / (attention * effect).abs().sum((1, 2)).clamp_min(1e-12)
        scale = 3 * drift * negative_share.clamp(0, 1) * (.35 - support).tanh()
        direction = derivative / derivative.abs().amax((1, 2), keepdim=True).clamp_min(1e-12)
        budget = -scale[:, None, None] * direction * spatial[:, None]
        predicted = (derivative * budget).sum((1, 2))
        assert torch.isfinite(budget).all() and (predicted <= 1e-8).all()
        maxima = budget.abs().amax((1, 2))
        effective = (maxima > 0).nonzero().flatten().tolist()
        details = [dict(token=int(self.final_tokens[rows[i]]), gain=float(gain[i]),
                        drift=float(drift[i]), budget_max=float(maxima[i]),
                        predicted_log_probability_change=float(predicted[i])) for i in effective]
        cache = self.tail_kwargs['past_key_values']
        self.cache_backup = {
            layer: (cache.layers[layer].keys[rows, :, -1:].clone(),
                    cache.layers[layer].values[rows, :, -1:].clone())
            for layer in range(32, len(self.decoder.layers))}
        return [rows[i] for i in effective], budget[effective], details

    def _recompute(self, index, original_output, rows, budget):
        if budget.ndim == 2:
            return super()._recompute(index, original_output, rows, budget)
        state, layer = self.scratch[index], self.decoder.layers[index]
        scores = state['scores'].clone()
        visual = scores[:, :, 0, self.start:self.end]
        original = visual[rows].float()
        shifted = original + budget
        mass_correction = shifted.logsumexp(-1, keepdim=True) - original.logsumexp(-1, keepdim=True)
        visual[rows] = (shifted - mass_correction).to(visual.dtype)
        attention = scores.softmax(-1, dtype=torch.float32).to(state['values'].dtype)
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

    def _finish_decoder(self, module, args, output):
        self.reference_hidden = output.last_hidden_state[:, -1:]
        self.probe_rows = []
        self.cache_backup = None
        previous = self.replay_steps
        output = super()._finish_decoder(module, args, output)
        if self.replay_steps != previous and self.probe_rows:
            log_probability = self.head(output.last_hidden_state[:, -1:]).float()[:, 0].log_softmax(-1)
            rejected, indices = [], []
            for i, row in enumerate(self.probe_rows):
                history = self.events.get(row, ())
                if history and history[-1]['step'] == self.step:
                    event = history[-1]
                    event['actual_log_probability_change'] = float(
                        log_probability[row, event['token']] - self.probe_before[i])
                    accepted = event['actual_log_probability_change'] < 0 and not self.diagnostic_reject_all
                    event['accepted'] = accepted
                    if not accepted:
                        rejected.append(row)
                        indices.append(i)
            if rejected:
                output.last_hidden_state[rejected, -1] = self.reference_hidden[rejected, 0]
                cache = self.tail_kwargs['past_key_values']
                for layer, (keys, values) in self.cache_backup.items():
                    cache.layers[layer].keys[rejected, :, -1:] = keys[indices]
                    cache.layers[layer].values[rejected, :, -1:] = values[indices]
        return output

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [e for e in self.events.get(row, ()) if e['step'] < len(tokens)]
            changes = [e['actual_log_probability_change'] for e in events if 'actual_log_probability_change' in e]
            accepted = [e for e in events if e.get('accepted', False)]
            stats.update(head_specific=True, gradient_scope='final' if self.final_gradient else 'source',
                         gradient_probe_calls=self.gradient_probes,
                         maximum_probe_hidden_difference=self.maximum_probe_hidden_difference,
                         measured_log_probability_changes=len(changes),
                         log_probability_decreases=sum(x < 0 for x in changes),
                         accepted_guidance_calls=len(accepted),
                         diagnostic_reject_all=self.diagnostic_reject_all,
                         mean_log_probability_change=sum(changes) / len(changes) if changes else 0.)
        return result
