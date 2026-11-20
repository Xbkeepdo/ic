"""Use a frozen detector and visual evidence to guide an explicit rival.

Adapt the existing attention recycling operation to a final-logit margin.
The prefix stays fixed. Three bounded attention proposals share one gradient;
only a proposal whose final argmax equals the selected rival is retained.
"""

import torch

from features.qwen3_probe_attention_recycle import Qwen3BatchProbeAttentionRecycle
from features.qwen3_sar_ems_final_target import LastPositionCache, Qwen3BatchFinalTargetReplay
from features.qwen3_sar_ems_head_gradient import ReadOnlyTailCache
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchProbeGroundedRival(Qwen3BatchProbeAttentionRecycle):
    def __init__(self, *args, diagnostic_force_rival=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_rival = diagnostic_force_rival
        self.rivals = {}
        self.rival_choices = {}
        self.current_logit = None
        self.trial_calls = 0

    def _gradient(self, source, rows):
        self.replaying_tail = self._replaying = True
        try:
            with torch.inference_mode(False), torch.enable_grad():
                residual = self.scratch[source]['ffn_residual'].detach().clone().requires_grad_(True)
                layer = self.decoder.layers[source]
                current = residual[:, None] + layer.mlp(layer.post_attention_layernorm(residual[:, None]))
                kwargs = dict(self.tail_kwargs)
                kwargs['past_key_values'] = ReadOnlyTailCache(kwargs['past_key_values'])
                kwargs['position_embeddings'] = tuple(x.detach().clone() for x in kwargs['position_embeddings'])
                kwargs['cache_position'] = kwargs['cache_position'].detach().clone()
                for tail in self.decoder.layers[32:]:
                    current = tail(current, **kwargs)
                normalized = self.decoder.norm(current)
                logits = self.head(normalized).float()[:, 0]
                targets = self.final_tokens[rows].detach().clone()
                rivals = targets.new_tensor([self.rivals[row]['token'] for row in rows])
                margin = logits[rows, targets] - logits[rows, rivals]
                gradient, = torch.autograd.grad(margin.sum(), residual)
                gradient = gradient[rows].detach()
                difference = (normalized.detach() - self.reference_hidden).abs().max().item()
                self.maximum_probe_hidden_difference = max(self.maximum_probe_hidden_difference, difference)
                self.probe_rows = rows
                self.gradient_probes += 1
        finally:
            self.replaying_tail = self._replaying = False
        assert torch.isfinite(gradient).all()
        return gradient

    def _budget(self, source):
        rows = [i for i, (token, active) in enumerate(
            zip(self.final_tokens.tolist(), self.active.tolist(), strict=True))
            if active and token in self.nouns and token not in self.seen[i]]
        if not rows:
            return [], None, []
        original_probe = self.probe
        self.rivals = {}

        def select_rival(features):
            real = original_probe(features)
            permitted = torch.zeros_like(real, dtype=torch.bool)
            for slot, row in enumerate(rows):
                target = int(self.final_tokens[row])
                decision = dict(step=self.step, target_token=target, probe_p_real=float(real[slot]),
                                selected=False, reason='target_probe_not_flagged')
                if float(real[slot]) < .5 or self.force_rival:
                    ranked = self.current_logit[row].topk(16).indices.tolist()
                    candidates = [t for t in ranked if t != target and t in self.nouns
                                  and float(self.current_logit[row, target] - self.current_logit[row, t]) <= 2]
                    if self.force_rival:
                        candidates = [t for t in ranked if t != target][:1]
                    decision['reason'] = 'no_supported_plausible_rival'
                    if candidates:
                        intensity = features[slot:slot + 1, :36].expand(len(candidates), -1)
                        if self.force_rival:
                            candidate_real = torch.ones(len(candidates), device=real.device)
                        else:
                            semantics = torch.stack([self.reader.semantics[i][row,
                                [self.reader.columns[t] for t in candidates]] for i in range(36)], 1)
                            candidate_real = original_probe(torch.cat((intensity, semantics), 1))
                        tokens = self.final_tokens.new_tensor([[target, *candidates]])
                        support, _, _, _ = readout_contrast(self, source, [row], tokens)
                        for j, candidate in enumerate(candidates):
                            if self.force_rival or (float(candidate_real[j]) >= .5
                                    and float(support[0, j + 1]) > 0
                                    and float(support[0, j + 1]) > float(support[0, 0])):
                                self.rivals[row] = dict(token=candidate, p_real=float(candidate_real[j]),
                                    visual_support=float(support[0, j + 1]),
                                    logit_gap=float(self.current_logit[row, target] - self.current_logit[row, candidate]))
                                permitted[slot] = True
                                decision.update(selected=True, reason='supported_rival',
                                    rival_token=candidate, rival_p_real=float(candidate_real[j]),
                                    rival_visual_support=float(support[0, j + 1]),
                                    target_visual_support=float(support[0, 0]),
                                    native_logit_gap=self.rivals[row]['logit_gap'])
                                break
                self.rival_choices.setdefault(row, []).append(decision)
            return torch.where(permitted, torch.zeros_like(real) if self.force_rival else real,
                               torch.ones_like(real))

        self.probe = select_rival
        try:
            effective, budget, details = super()._budget(source)
        finally:
            self.probe = original_probe
        if not effective:
            return effective, budget, details
        for row, detail in zip(effective, details, strict=True):
            rival = self.rivals[row]
            detail.update(rival_token=rival['token'], rival_probe_p_real=rival['p_real'],
                          rival_visual_support=rival['visual_support'], native_logit_gap=rival['logit_gap'],
                          predicted_margin_change=detail.pop('predicted_log_probability_change'))
        cache = self.tail_kwargs['past_key_values']
        self.cache_backup = {layer: (cache.layers[layer].keys[effective, :, -1:].clone(),
                                    cache.layers[layer].values[effective, :, -1:].clone())
                             for layer in range(32, len(self.decoder.layers))}
        self.probe_rows = effective
        return effective, budget, details

    def _finish_decoder(self, module, args, output):
        if self.step == 0 or self.diagnostic_zero_replay:
            return Qwen3BatchFinalTargetReplay._finish_decoder(self, module, args, output)
        self.reference_hidden = output.last_hidden_state[:, -1:]
        self.probe_rows = []
        self.cache_backup = None
        self.current_logit = self.head(self.reference_hidden).float()[:, 0]
        self.final_tokens = self.current_logit.argmax(-1)
        rows, budget, details = self._budget(31)
        if not rows:
            self.scratch.clear()
            return output
        native_logp = self.current_logit.log_softmax(-1)
        hidden = output.last_hidden_state.clone()
        pending = list(range(len(rows)))
        for detail in details:
            detail.update(accepted=False, trials=[])
        for multiplier in (1., 2., 4.):
            if not pending:
                break
            trial_rows = [rows[i] for i in pending]
            # The base operation moves .5*D of eligible history mass.
            # Scale up to D and min(1,2D), never beyond available mass.
            scales = budget.new_tensor([min(1., .5 * details[i]['drift'] * multiplier)
                                        / (.5 * details[i]['drift']) for i in pending])
            proposal = budget[pending] * scales[:, None, None]
            cache = LastPositionCache(self.tail_kwargs['past_key_values'], trial_rows)
            kwargs = dict(self.tail_kwargs, past_key_values=cache)
            current = self._recompute(31, self.source_output, trial_rows, proposal)
            self.replaying_tail = True
            try:
                for layer in self.decoder.layers[32:]:
                    current = layer(current, **kwargs)
            finally:
                self.replaying_tail = False
            assert cache.updates == list(range(32, len(self.decoder.layers)))
            corrected = self.decoder.norm(current)
            logits = self.head(corrected).float()[:, 0]
            logp = logits.log_softmax(-1)
            next_pending = []
            for i in pending:
                row, detail = rows[i], details[i]
                target, rival = detail['token'], detail['rival_token']
                margin = float(logits[row, target] - logits[row, rival])
                top = int(logits[row].argmax())
                change = float(logp[row, target] - native_logp[row, target])
                keep = top == rival and margin < detail['native_logit_gap'] and not self.diagnostic_reject_all
                detail['trials'].append(dict(multiplier=multiplier, margin=margin,
                    predicted_token=top, actual_log_probability_change=change, accepted=keep))
                detail.update(actual_log_probability_change=change,
                              actual_margin_change=margin - detail['native_logit_gap'],
                              accepted=keep)
                if keep:
                    hidden[row, -1] = corrected[row, 0]
                else:
                    next_pending.append(i)
            pending = next_pending
            self.trial_calls += 1
        if pending:
            rejected = [rows[i] for i in pending]
            cache = self.tail_kwargs['past_key_values']
            for layer, (keys, values) in self.cache_backup.items():
                cache.layers[layer].keys[rejected, :, -1:] = keys[pending]
                cache.layers[layer].values[rejected, :, -1:] = values[pending]
        output.last_hidden_state = hidden
        self._record(31, 31, rows, details)
        self.replay_steps += 1
        self.scratch.clear()
        return output

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [e for e in self.events.get(row, ()) if e['step'] < len(tokens)]
            stats.update(grounded_rival=True, diagnostic_force_rival=self.force_rival,
                acceptance='Final argmax must equal the supported rival and its margin over the original must improve.',
                attention_trial_forward_calls=self.trial_calls,
                rival_choices=[e for e in self.rival_choices.get(row, ()) if e['step'] < len(tokens)],
                recycle_events=[dict(e, generated_token=tokens[e['step']]) for e in events if 'accepted' in e])
        return result
