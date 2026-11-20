"""Use both directions of the SAR and FFN preference difference in next-layer guidance."""

import torch
from torch.nn import functional as F

from features.llava_sar_ems_attention import SarPatchTerms
from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback, TAU


class Qwen3BatchSignedLayerFeedback(Qwen3BatchLayerFeedback):
    def __init__(self, model, span, nouns, strength=5.0, *, directional_evidence=False):
        super().__init__(model, span, nouns, mode="next", strength=strength)
        self.directional_evidence = directional_evidence
        self.readout_projections = {}

    def _evidence(self, source, row, token, attention, residual):
        layer = self.decoder.layers[source]
        values = self.visual_values[source][row]
        output_projection = layer.self_attn.o_proj.weight
        norm = self.decoder.norm
        if self.directional_evidence:
            # Local derivative of the target's final-norm logit at pre-FFN r.
            # Omit the common positive 1/RMS(r), which cancels when normalizing.
            readout = self.head.weight[token].float() * norm.weight.float()
            r = residual.float()
            readout = readout - r * ((r * readout).mean() / (r.square().mean() + norm.variance_epsilon))
            if source not in self.readout_projections:
                self.readout_projections[source] = output_projection.float()
            direction = (readout @ self.readout_projections[source]).reshape(values.shape[0], -1)
            contribution = (attention.float() * torch.einsum('hmd,hd->hm', values.float(), direction)).sum(0)
            support = contribution.clamp_min(0)
            if support.sum().item() <= 1e-12:
                return None
            return support / support.sum()

        cache_key = (source, row)
        terms = self.patch_terms.get(cache_key)
        if terms is None:
            terms = SarPatchTerms(values, output_projection, self.visual_hpre[source][row], norm, self.head)
            self.patch_terms[cache_key] = terms
        return terms.distribution(attention, token)

    def _budget(self, source):
        state = self.scratch[source]
        updates, residual = state["update"], state["ffn_residual"]
        norm = self.decoder.norm
        post, before = norm(residual + updates), norm(residual)
        tokens = self.head(post).argmax(dim=-1).masked_fill(~self.active, -1)
        rows, budgets, details = [], [], []
        for row, token in enumerate(tokens.tolist()):
            if token < 0 or token not in self.nouns:
                continue
            gain = self.head.weight[token].float() @ (post[row].float() - before[row].float())
            if gain.item() <= 0:
                continue
            evidence = self._evidence(source, row, token, state["attention"][row], residual[row])
            if evidence is None:
                continue
            preference = (
                self.visual_normalized[source][row]
                @ F.normalize(updates[row].float(), dim=-1) / TAU
            ).softmax(dim=0)
            drift = (1 - (evidence * preference).sqrt().sum()).clamp(0, 1)
            difference = evidence - preference
            if self.directional_evidence:
                # Preserve the original positive-only correction for this ablation.
                difference = difference.clamp_min(0)
                span = difference.max()
            else:
                span = difference.max() - difference.min()
            # The signed version keeps the original bound on peak-to-peak correction.
            budget = (self.strength * drift * (gain / 2).tanh()) * difference / span.clamp_min(1e-6)
            maximum = budget.max().item()
            if maximum <= 0:
                continue
            if not torch.isfinite(budget).all().item():
                raise RuntimeError(f"Non-finite signed budget at layer {source}")
            rows.append(row)
            budgets.append(budget)
            details.append({"token": token, "gain": gain.item(), "drift": drift.item(),
                            "budget_max": maximum})
        return rows, torch.stack(budgets) if budgets else None, details
