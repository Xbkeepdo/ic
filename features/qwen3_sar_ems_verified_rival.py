"""Auxiliary final margin correction for visually checked near-ties."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers import LogitsProcessor

from features.qwen3_sar_ems_removal_gate import Qwen3BatchWriteRemovalGate
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchVerifiedRival(Qwen3BatchWriteRemovalGate):
    """Keep the spatial adjustment and recheck its two tokens before decoding."""

    def __init__(self, model, span, nouns):
        super().__init__(model, span, nouns)
        self.sources = (31, 35)
        self.targets = (32, 35)
        self.proposals = {}
        self.confirmed = {}
        self.corrections = {}

    def _begin_step(self, module, args, kwargs):
        super()._begin_step(module, args, kwargs)
        self.proposals.clear()
        self.confirmed.clear()

    def _budget(self, source):
        if source == 31:
            rows, budget, details = super()._budget(source)
            if rows:
                state = self.scratch[source]
                norm = self.decoder.norm(state['ffn_residual'] + state['update'])
                primary = self.head(norm).argmax(-1).tolist()
                self.proposals.update((row, (primary[row], detail['token'])) for row, detail in zip(rows, details, strict=True))
            return rows, budget, details

        # Use the final layer's actual visual write to confirm that the rival
        # retains support after the spatial edit and remaining decoder layers.
        if self.proposals:
            rows = list(self.proposals)
            tokens = torch.tensor(list(self.proposals.values()), device=self.active.device, dtype=torch.long)
            visual, ffn, _, _ = readout_contrast(self, source, rows, tokens)
            for row, v, g in zip(rows, visual.tolist(), ffn.tolist(), strict=True):
                advantage = v[1] - v[0]
                if v[1] > 0 and advantage > 0 and g[0] > g[1]:
                    primary, rival = self.proposals[row]
                    self.confirmed[row] = (primary, rival, min(0.5, advantage, g[0] - g[1]))
        return [], None, []

    def process_scores(self, scores):
        if not self.confirmed:
            return scores
        winners = scores.argmax(-1).tolist()
        for row, (primary, rival, penalty) in self.confirmed.items():
            if winners[row] != primary:
                continue
            gap = float(scores[row, primary] - scores[row, rival])
            if gap < penalty:
                scores[row, primary] -= penalty
                self.corrections[row] = self.corrections.get(row, 0) + 1
        return scores

    @contextmanager
    def install(self):
        class CheckRival(LogitsProcessor):
            def __call__(_, input_ids, scores):
                return self.process_scores(scores)

        # Keep original generation settings and apply the check after the
        # actual decoder forward. This controller is installed per batch.
        original = self.model.generate

        def generate(*args, **kwargs):
            processors = list(kwargs.get('logits_processor') or ())
            kwargs['logits_processor'] = [*processors, CheckRival()]
            return original(*args, **kwargs)

        with patch.object(self.model, 'generate', generate):
            with super().capture():
                yield

    def capture(self):
        return self.install()

    def statistics(self, trimmed_responses):
        rows = super().statistics(trimmed_responses)
        for i, row in enumerate(rows):
            row['verified_rival_penalties'] = self.corrections.get(i, 0)
        return rows
