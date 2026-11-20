"""Allow a bounded increase in visual attention when the current writes support the target."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from features import qwen3_sar_ems_layer_feedback as feedback
from features.qwen3_sar_ems_signed import Qwen3BatchSignedLayerFeedback


class Qwen3BatchSupportedVisualMass(Qwen3BatchSignedLayerFeedback):
    def __init__(self, model, span, nouns, strength=5.0):
        super().__init__(model, span, nouns, strength=strength, directional_evidence=True)
        self.net_support = {}

    def _evidence(self, source, row, token, attention, residual):
        layer = self.decoder.layers[source]
        norm = self.decoder.norm
        values = self.visual_values[source][row]
        readout = self.head.weight[token].float() * norm.weight.float()
        r = residual.float()
        readout = readout - r * ((r * readout).mean() / (r.square().mean() + norm.variance_epsilon))
        if source not in self.readout_projections:
            self.readout_projections[source] = layer.self_attn.o_proj.weight.float()
        direction = (readout @ self.readout_projections[source]).reshape(values.shape[0], -1)
        contribution = (attention.float() * torch.einsum('hmd,hd->hm', values.float(), direction)).sum(0)
        support = contribution.clamp_min(0)
        total = support.sum()
        if total.item() <= 1e-12:
            return None
        self.net_support[row] = (contribution.sum() / total).clamp(0, 1)
        return support / total

    def _budget(self, source):
        self.net_support.clear()
        rows, budget, details = super()._budget(source)
        if not rows:
            return rows, budget, details
        drift = budget.new_tensor([detail['drift'] for detail in details])
        gain = budget.new_tensor([detail['gain'] for detail in details])
        # A shift between 0 and 0.5 changes the visual-vs-text attention
        # odds by at most exp(0.5), without changing the revised visual ratios.
        mass = 0.5 * drift * (gain / 2).tanh() * torch.stack([self.net_support[row] for row in rows])
        return rows, torch.cat((budget, mass[:, None]), dim=1), details

    @contextmanager
    def capture(self):
        original = feedback.rebalance_visual_logits

        def rebalance(logits, budget):
            directional = original(logits, budget[:, :-1])
            return (directional.float() + budget[:, -1, None, None]).to(directional.dtype)

        with patch.object(feedback, 'rebalance_visual_logits', rebalance):
            with super().capture():
                yield
