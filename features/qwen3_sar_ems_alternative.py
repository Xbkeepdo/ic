"""Guide attention toward a close alternative only when visual writes favor it."""

import torch
from torch.nn import functional as F

from features.qwen3_sar_ems_layer_feedback import TAU
from features.qwen3_sar_ems_signed import Qwen3BatchSignedLayerFeedback


class Qwen3BatchSupportedAlternative(Qwen3BatchSignedLayerFeedback):
    def __init__(self, model, span, nouns, strength=5.0):
        super().__init__(model, span, nouns, strength=strength, directional_evidence=True)
        self.sources = (24, 25)
        self.targets = (25, 26)

    def _budget(self, source):
        state = self.scratch[source]
        residual, update = state['ffn_residual'], state['update']
        norm = self.decoder.norm
        before, after = norm(residual), norm(residual + update)
        logits = self.head(after).float()
        top_scores, top_ids = logits.topk(8, dim=-1)
        active = self.active.tolist()
        best_ids = logits.argmax(dim=-1).tolist()
        rows, budgets, details = [], [], []
        for row, primary in enumerate(best_ids):
            if not active[row] or primary not in self.nouns:
                continue
            primary_gain = self.head.weight[primary].float() @ (after[row].float() - before[row].float())
            if primary_gain.item() <= 0:
                continue
            ranked = zip(top_ids[row].tolist(), top_scores[row].tolist(), strict=True)
            peak = logits[row, primary].item()
            alternatives = [token for token, score in ranked
                            if token != primary and token in self.nouns and peak - score <= 2.0]
            if not alternatives:
                continue

            candidates = [primary, *alternatives]
            r = residual[row].float()
            readout = self.head.weight[candidates].float() * norm.weight.float()
            denominator = r.square().mean() + norm.variance_epsilon
            readout = readout - (readout * r).mean(dim=-1, keepdim=True) * r / denominator
            layer = self.decoder.layers[source]
            if source not in self.readout_projections:
                self.readout_projections[source] = layer.self_attn.o_proj.weight.float()
            values = self.visual_values[source][row].float()
            readout = (readout @ self.readout_projections[source]).reshape(len(candidates), values.shape[0], -1)
            contributions = torch.einsum('hm,hmd,khd->km', state['attention'][row].float(), values, readout)
            positive = contributions.clamp_min(0)
            total_positive = positive.sum(dim=-1)
            support = contributions.sum(dim=-1)
            distributions = positive / total_positive.clamp_min(1e-12)[:, None]
            preference = (self.visual_normalized[source][row] @ F.normalize(update[row].float(), dim=-1)
                          / TAU).softmax(dim=0)
            drift = (1 - (distributions * preference).sqrt().sum(dim=-1)).clamp(0, 1)
            eligible = ((support[1:] > support[0]) & (support[1:] > 0) & (drift[1:] < drift[0]))
            if not eligible.any().item():
                continue
            chosen = int(support[1:].masked_fill(~eligible, -float('inf')).argmax().item()) + 1
            advantage = (support[chosen] - support[0]) / (support[chosen] + support[0].abs()).clamp_min(1e-12)
            deficit = (distributions[chosen] - preference).clamp_min(0)
            budget = self.strength * drift[0] * (primary_gain / 2).tanh() * advantage * deficit / deficit.max().clamp_min(1e-6)
            budget_max = budget.max().item()
            if budget_max <= 0:
                continue
            if not torch.isfinite(budget).all().item():
                raise RuntimeError('Non-finite alternative guidance budget')
            rows.append(row)
            budgets.append(budget)
            details.append({'token': candidates[chosen], 'gain': primary_gain.item(), 'drift': drift[0].item(),
                            'budget_max': budget_max})
        return rows, torch.stack(budgets) if budgets else None, details
