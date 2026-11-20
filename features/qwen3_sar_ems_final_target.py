"""Read the final candidate, then guide its source-layer visual attention."""

from contextlib import ExitStack, contextmanager

import torch
from torch.nn import functional as F

from features.llava_decoding_evidence import TAU
from features.qwen3_sar_ems_removal_gate import Qwen3BatchWriteRemovalGate
from features.qwen3_visual_write_contrast import readout_contrast


class LastPositionCache:
    """Replace the current tail position for selected rows in an existing cache."""

    def __init__(self, cache, rows, *, check_identical=False):
        self.cache = cache
        self.rows = rows
        self.check_identical = check_identical
        self.updates = []

    def update(self, keys, values, layer_idx, cache_kwargs):
        position = cache_kwargs['cache_position']
        layer = self.cache.layers[layer_idx]
        if position.numel() != 1 or keys.shape[-2] != 1:
            raise RuntimeError('Tail replay requires a single current token')
        offset = int(position.item())
        if layer.keys.shape[-2] != offset + 1:
            raise RuntimeError('Tail replay position differs from the existing cache end')
        if self.check_identical:
            assert torch.equal(layer.keys[self.rows, :, offset:], keys[self.rows])
            assert torch.equal(layer.values[self.rows, :, offset:], values[self.rows])
        layer.keys[self.rows, :, offset:] = keys[self.rows]
        layer.values[self.rows, :, offset:] = values[self.rows]
        self.updates.append(layer_idx)
        return layer.keys, layer.values


class Qwen3BatchFinalTargetReplay(Qwen3BatchWriteRemovalGate):
    """Use final-token identity with layer 31 evidence and FFN preference.

    Only cached single-token decoding is replayed. Layer 31 reuses its Q/K/V;
    layers 32--35 recompute the current token and replace its existing cache
    entries. Previous positions and rows outside the gate are retained.
    """

    def __init__(self, model, span, nouns, *, disable_guidance=False,
                 diagnostic_zero_replay=False):
        super().__init__(model, span, nouns)
        self.mode = 'same'
        self.sources = self.targets = (31,)
        self.enabled = not disable_guidance
        self.diagnostic_zero_replay = diagnostic_zero_replay
        self.replaying_tail = False
        self.tail_kwargs = None
        self.source_output = None
        self.final_tokens = None
        self.replay_steps = 0

    def _begin_step(self, module, args, kwargs):
        super()._begin_step(module, args, kwargs)
        self.tail_kwargs = None
        self.source_output = None

    def _after_layer(self, index):
        def retain(_module, _args, output):
            self.source_output = output[:, -1:].clone()
            return output
        return retain

    def _remember_tail(self, _module, _args, kwargs):
        if not self.replaying_tail:
            self.tail_kwargs = dict(kwargs)

    def _budget(self, source):
        state = self.scratch[source]
        residual, update = state['ffn_residual'], state['update']
        top = self.final_tokens.tolist()
        rows = [i for i, (token, active) in enumerate(zip(top, self.active.tolist(), strict=True))
                if active and token in self.nouns and token not in self.seen[i]]
        if not rows:
            return [], None, []
        targets = [top[i] for i in rows]
        tokens = self.final_tokens.new_tensor(targets)[:, None]
        visual, gain, _, _ = readout_contrast(self, source, rows, tokens)
        weak = (visual[:, 0] < 0.35).nonzero().flatten().tolist()
        if not weak:
            return [], None, []
        rows = [rows[i] for i in weak]
        targets = [targets[i] for i in weak]
        support, gain = visual[weak, 0], gain[weak, 0]
        r = residual[rows].float()
        norm = self.decoder.norm
        direction = self.head.weight[targets].float() * norm.weight.float()
        direction = direction - r * (direction * r).mean(-1, keepdim=True) / (
            r.square().mean(-1, keepdim=True) + norm.variance_epsilon)
        values = self.visual_values[source][rows].float()
        projected = (direction @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        writes = torch.einsum('bhm,bhmd,bhd->bm', state['attention'][rows].float(), values, projected)
        negative = (-writes).clamp_min(0)
        total = negative.sum(-1, keepdim=True)
        evidence = negative / total.clamp_min(1e-12)
        similarity = (self.visual_normalized[source][rows]
                      @ F.normalize(update[rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
        deficit = (evidence - preference).clamp_min(0)
        negative_share = (total[:, 0] / writes.abs().sum(-1).clamp_min(1e-12)).clamp(0, 1)
        scale = 3 * drift * negative_share * (0.35 - support).tanh()
        budget = scale[:, None] * deficit / deficit.amax(-1, keepdim=True).clamp_min(1e-6)
        if not torch.isfinite(budget).all().item():
            raise RuntimeError('Non-finite final-target evidence budget')
        effective = (budget.amax(-1) > 0).nonzero().flatten().tolist()
        details = [dict(token=targets[i], gain=float(gain[i]), drift=float(drift[i]),
                        budget_max=float(budget[i].max())) for i in effective]
        return [rows[i] for i in effective], budget[effective], details

    def _finish_decoder(self, _module, _args, output):
        # Prefill has different matrix shapes from a one-query replay.
        if self.step == 0:
            self.scratch.clear()
            return output
        if self.tail_kwargs is None or self.source_output is None:
            raise RuntimeError('Missing source or tail state for final-target feedback')
        self.final_tokens = self.head(output.last_hidden_state[:, -1:]).argmax(-1)[:, 0]
        if self.diagnostic_zero_replay:
            rows = self.active.nonzero().flatten().tolist()
            budget = self.source_output.new_zeros((len(rows), self.end - self.start))
            details = [dict(token=int(self.final_tokens[i]), gain=0., drift=0., budget_max=0.) for i in rows]
        else:
            rows, budget, details = self._budget(31)
        if not rows:
            self.scratch.clear()
            return output
        cache = LastPositionCache(self.tail_kwargs['past_key_values'], rows,
                                  check_identical=self.diagnostic_zero_replay)
        kwargs = dict(self.tail_kwargs, past_key_values=cache)
        current = self._recompute(31, self.source_output, rows, budget)
        self.replaying_tail = True
        try:
            for layer in self.decoder.layers[32:]:
                current = layer(current, **kwargs)
        finally:
            self.replaying_tail = False
        assert cache.updates == list(range(32, len(self.decoder.layers)))
        corrected = self.decoder.norm(current)
        if self.diagnostic_zero_replay:
            assert torch.equal(corrected[rows, 0], output.last_hidden_state[rows, -1])
        hidden = output.last_hidden_state.clone()
        hidden[rows, -1] = corrected[rows, 0]
        output.last_hidden_state = hidden
        self.replay_steps += 1
        self._record(31, 31, rows, details)
        self.scratch.clear()
        return output

    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            stack.enter_context(super().capture())
            if self.enabled:
                handle = self.decoder.layers[32].register_forward_pre_hook(
                    self._remember_tail, with_kwargs=True)
                stack.callback(handle.remove)
                handle = self.decoder.register_forward_hook(self._finish_decoder)
                stack.callback(handle.remove)
            yield

    def statistics(self, responses):
        result = super().statistics(responses)
        for row in result:
            row.update(final_target_read_layer=35, replayed_tail_layers=[32, 33, 34, 35],
                       diagnostic_zero_replay=self.diagnostic_zero_replay)
        return result
