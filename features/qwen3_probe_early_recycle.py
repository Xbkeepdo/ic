"""Move the frozen probe-guided attention recycle operation to layer 19."""

import torch

from features.qwen3_probe_attention_recycle import Qwen3BatchProbeAttentionRecycle
from features.qwen3_sar_ems_final_target import LastPositionCache
from features.qwen3_sar_ems_head_gradient import ReadOnlyTailCache


class Qwen3BatchProbeEarlyRecycle(Qwen3BatchProbeAttentionRecycle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources = self.targets = (19,)

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
                for tail in self.decoder.layers[source + 1:]:
                    current = tail(current, **kwargs)
                normalized = self.decoder.norm(current)
                logits = self.head(normalized).float()[:, 0]
                targets = self.final_tokens[rows].detach().clone()
                objective = logits.log_softmax(-1)[rows, targets]
                gradient, = torch.autograd.grad(objective.sum(), residual)
                gradient = gradient[rows].detach()
                difference = (normalized.detach() - self.reference_hidden).abs().max().item()
                self.maximum_probe_hidden_difference = max(self.maximum_probe_hidden_difference, difference)
                self.gradient_probes += 1
        finally:
            self.replaying_tail = self._replaying = False
        assert torch.isfinite(gradient).all()
        return gradient

    def _budget(self, source):
        rows, budget, details = super()._budget(source)
        if rows:
            cache = self.tail_kwargs['past_key_values']
            self.cache_backup = {layer: (cache.layers[layer].keys[rows, :, -1:].clone(),
                                        cache.layers[layer].values[rows, :, -1:].clone())
                                 for layer in range(source + 1, len(self.decoder.layers))}
        return rows, budget, details

    def _finish_decoder(self, module, args, output):
        if self.step == 0:
            self.scratch.clear()
            return output
        source, = self.sources
        self.reference_hidden = output.last_hidden_state[:, -1:]
        native = self.head(self.reference_hidden).float()[:, 0]
        self.final_tokens = native.argmax(-1)
        if self.diagnostic_zero_replay:
            rows = self.active.nonzero().flatten().tolist()
            budget = torch.zeros_like(self.scratch[source]['scores'][rows, :, 0], dtype=torch.float32)
            details = [dict(token=int(self.final_tokens[row]), gain=0., drift=0., budget_max=0.) for row in rows]
        else:
            rows, budget, details = self._budget(source)
        if not rows:
            self.scratch.clear()
            return output
        # The remembered kwargs at layer32 share the same mask, rotary data,
        # cache object and current position as layers20--35.
        cache = LastPositionCache(self.tail_kwargs['past_key_values'], rows,
                                  check_identical=self.diagnostic_zero_replay)
        kwargs = dict(self.tail_kwargs, past_key_values=cache)
        current = self._recompute(source, self.source_output, rows, budget)
        self.replaying_tail = True
        try:
            for tail in self.decoder.layers[source + 1:]:
                current = tail(current, **kwargs)
        finally:
            self.replaying_tail = False
        assert cache.updates == list(range(source + 1, len(self.decoder.layers)))
        corrected = self.decoder.norm(current)
        if self.diagnostic_zero_replay:
            assert torch.equal(corrected[rows, 0], self.reference_hidden[rows, 0])
        hidden = output.last_hidden_state.clone()
        hidden[rows, -1] = corrected[rows, 0]
        if not self.diagnostic_zero_replay:
            before, after = native.log_softmax(-1), self.head(corrected).float()[:, 0].log_softmax(-1)
            rejected = []
            for i, (row, detail) in enumerate(zip(rows, details, strict=True)):
                change = float(after[row, detail['token']] - before[row, detail['token']])
                accepted = change < 0 and not self.diagnostic_reject_all
                detail.update(actual_log_probability_change=change, accepted=accepted)
                if not accepted:
                    hidden[row, -1] = self.reference_hidden[row, 0]
                    rejected.append(i)
            if rejected:
                indices = [rows[i] for i in rejected]
                native_cache = self.tail_kwargs['past_key_values']
                for layer, (keys, values) in self.cache_backup.items():
                    native_cache.layers[layer].keys[indices, :, -1:] = keys[rejected]
                    native_cache.layers[layer].values[indices, :, -1:] = values[rejected]
        output.last_hidden_state = hidden
        self._record(source, source, rows, details)
        self.replay_steps += 1
        self.scratch.clear()
        return output

    def statistics(self, responses):
        result = super().statistics(responses)
        for tokens, stats, row in zip(responses, result, range(len(responses)), strict=True):
            stats.update(replayed_tail_layers=list(range(20, 36)), intervention_layer=19,
                         recycle_events=[dict(e, generated_token=tokens[e['step']])
                             for e in self.events.get(row, ()) if e['step'] < len(tokens) and 'accepted' in e])
        return result
