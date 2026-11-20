"""Use visual writes and FFN drift in the margin between plausible entity tokens."""

import torch
from torch.nn import functional as F

from features.qwen3_sar_ems_alternative import Qwen3BatchSupportedAlternative
from features.qwen3_sar_ems_layer_feedback import TAU


class Qwen3BatchMarginFeedback(Qwen3BatchSupportedAlternative):
    """Guide the next attention layer toward patches that favor a rival entity."""

    def _budget(self, source):
        state = self.scratch[source]
        residual, update = state['ffn_residual'], state['update']
        norm = self.decoder.norm
        before, after = norm(residual), norm(residual + update)
        logits = self.head(after).float()
        top_scores, top_ids = logits.topk(8, dim=-1)
        rows, budgets, details = [], [], []

        for row, (active, primary) in enumerate(zip(self.active.tolist(), logits.argmax(-1).tolist(), strict=True)):
            if not active or primary not in self.nouns:
                continue
            peak = logits[row, primary].item()
            alternatives = [token for token, score in zip(top_ids[row].tolist(), top_scores[row].tolist(), strict=True)
                            if token != primary and token in self.nouns and peak - score <= 2.0]
            if not alternatives:
                continue
            candidates = [primary, *alternatives]
            weights = self.head.weight[candidates].float()
            gains = weights @ (after[row].float() - before[row].float())
            margin_gains = gains[0] - gains[1:]
            if gains[0].item() <= 0 or not (margin_gains > 0).any().item():
                continue

            r = residual[row].float()
            readout = weights * norm.weight.float()
            readout = readout - (readout * r).mean(-1, keepdim=True) * r / (r.square().mean() + norm.variance_epsilon)
            if source not in self.readout_projections:
                self.readout_projections[source] = self.decoder.layers[source].self_attn.o_proj.weight.float()
            values = self.visual_values[source][row].float()
            projected = (readout @ self.readout_projections[source]).reshape(len(candidates), values.shape[0], -1)
            writes = torch.einsum('hm,hmd,khd->km', state['attention'][row].float(), values, projected)

            # A relevant patch supports the rival more than the primary and
            # supplies positive evidence for the rival itself.
            margin = writes[1:] - writes[0]
            supporting = margin.clamp_min(0) * (writes[1:] > 0)
            totals = supporting.sum(-1)
            distributions = supporting / totals.clamp_min(1e-12)[:, None]
            preference = (self.visual_normalized[source][row] @ F.normalize(update[row].float(), dim=-1)
                          / TAU).softmax(dim=0)
            drift = (1 - (distributions * preference).sqrt().sum(-1)).clamp(0, 1)

            # A net visual advantage and a contrary FFN margin change are
            # required. Use the bounded signed margin to weight each rival.
            net_margin = margin.sum(-1)
            eligible = (net_margin > 0) & (writes[1:].sum(-1) > 0) & (totals > 0) & (margin_gains > 0)
            advantage = (net_margin / margin.abs().sum(-1).clamp_min(1e-12)).clamp(0, 1)
            confidence = drift * (margin_gains / 2).tanh() * advantage
            confidence = confidence.masked_fill(~eligible, 0)
            best = int(confidence.argmax().item())
            if confidence[best].item() <= 0:
                continue
            deficit = (distributions[best] - preference).clamp_min(0)
            budget = self.strength * confidence[best] * deficit / deficit.max().clamp_min(1e-6)
            maximum = budget.max().item()
            if not torch.isfinite(budget).all().item():
                raise RuntimeError('Non-finite margin feedback budget')
            if maximum > 0:
                rows.append(row)
                budgets.append(budget)
                details.append({'token': alternatives[best], 'gain': margin_gains[best].item(),
                                'drift': drift[best].item(), 'budget_max': maximum})
        return rows, torch.stack(budgets) if budgets else None, details
