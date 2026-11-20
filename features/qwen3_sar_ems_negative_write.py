"""Use negative visual contributions to weakly supported entity tokens."""

import torch
from torch.nn import functional as F

from features.llava_decoding_evidence import TAU
from features.qwen3_sar_ems_removal_gate import Qwen3BatchWriteRemovalGate
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchNegativeWrite(Qwen3BatchWriteRemovalGate):
    """Give missed negative evidence a larger share of next-layer visual attention."""

    def _budget(self, source):
        state = self.scratch[source]
        r, u = state['ffn_residual'], state['update']
        score = self.head(self.decoder.norm(r + u)).float()
        top = score.argmax(-1).tolist()
        rows = [i for i, (token, active) in enumerate(zip(top, self.active.tolist(), strict=True))
                if active and token in self.nouns and token not in self.seen[i]]
        if not rows:
            return [], None, []
        targets = [top[i] for i in rows]
        tokens = torch.tensor(targets, device=r.device, dtype=torch.long)[:, None]
        visual, gain, _, _ = readout_contrast(self, source, rows, tokens)
        weak = (visual[:, 0] < 0.35).nonzero().flatten().tolist()
        if not weak:
            return [], None, []
        rows = [rows[i] for i in weak]
        targets = [targets[i] for i in weak]
        relative = visual[weak, 0]
        gain = gain[weak, 0]

        residual = r[rows].float()
        norm = self.decoder.norm
        readout = self.head.weight[targets].float() * norm.weight.float()
        variance = residual.square().mean(-1, keepdim=True) + norm.variance_epsilon
        readout = readout - residual * (readout * residual).mean(-1, keepdim=True) / variance
        values = self.visual_values[source][rows].float()
        projected = (readout @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        writes = torch.einsum('bhm,bhmd,bhd->bm', state['attention'][rows].float(), values, projected)
        negative = (-writes).clamp_min(0)
        total = negative.sum(-1, keepdim=True)
        evidence = negative / total.clamp_min(1e-12)
        similarity = (self.visual_normalized[source][rows] @ F.normalize(u[rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
        deficit = (evidence - preference).clamp_min(0)
        negative_share = (total[:, 0] / writes.abs().sum(-1).clamp_min(1e-12)).clamp(0, 1)
        scale = 3 * drift * negative_share * (0.35 - relative).tanh()
        budget = scale[:, None] * deficit / deficit.amax(-1, keepdim=True).clamp_min(1e-6)
        if not torch.isfinite(budget).all().item():
            raise RuntimeError('Non-finite negative visual evidence budget')
        maxima = budget.amax(-1).tolist()
        effective = [i for i, maximum in enumerate(maxima) if maximum > 0]
        if not effective:
            return [], None, []
        details = [dict(token=targets[i], gain=float(gain[i]), drift=float(drift[i]), budget_max=maxima[i])
                   for i in effective]
        return [rows[i] for i in effective], budget[effective], details
