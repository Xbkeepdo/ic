"""Use a local visual-write ablation to gate target-conditioned attention."""

import torch
from torch.nn import functional as F

from features.llava_decoding_evidence import TAU
from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchWriteRemovalGate(Qwen3BatchLayerFeedback):
    def __init__(self, model, span, nouns):
        super().__init__(model, span, nouns)
        self.sources, self.targets = (31,), (32,)
        self.projections = {}
        self.seen = []

    def _begin_step(self, module, args, kwargs):
        continuation = bool(self.steps)
        super()._begin_step(module, args, kwargs)
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not continuation:
            self.seen = [set() for _ in range(len(ids))]
        else:
            for seen, token in zip(self.seen, ids[:, -1].tolist(), strict=True):
                seen.add(token)

    def _budget(self, source):
        state = self.scratch[source]
        r, u = state['ffn_residual'], state['update']
        logits = self.head(self.decoder.norm(r+u)).float()
        scores, ids = logits.topk(8, dim=-1)
        top = logits.argmax(-1).tolist()
        shortlist = []
        for row, (primary, active) in enumerate(zip(top, self.active.tolist(), strict=True)):
            if not active or primary not in self.nouns or primary in self.seen[row]:
                continue
            options = [(p, t) for p, t in enumerate(ids[row].tolist()) if t != primary and t in self.nouns]
            if options:
                shortlist.append((row, primary, options))
        if not shortlist:
            return [], None, []

        proposed = [row for row, _, _ in shortlist]
        contrast, ffn, _, _ = readout_contrast(self, source, proposed, ids[proposed])
        contrast, ffn = contrast.tolist(), ffn.tolist()
        ranks = ids[proposed].tolist()
        gaps = (scores[proposed,:1] - scores[proposed]).tolist()
        chosen = []
        for (row, primary, options), v, g, words, gap in zip(shortlist, contrast, ffn, ranks, gaps, strict=True):
            # In a tied vocabulary decision, the greedy argmax can fall
            # outside a top-k implementation's arbitrary tie ordering.
            if primary not in words:
                continue
            main = words.index(primary)
            if v[main] >= 0.35:
                continue
            valid = [(p, token) for p, token in options if gap[p] <= 2 and v[p] > 0.35 and v[p]-v[main] >= 0.35]
            if not valid:
                continue
            rank, rival = max(valid, key=lambda item: v[item[0]] - 0.5*gap[item[0]])
            visual_advantage = v[rank] - v[main]
            chosen.append((row, rival, visual_advantage, g[main] - g[rank]))
        if not chosen:
            return [], None, []

        rows, rivals, strengths, ffn_gains = map(list, zip(*chosen, strict=True))
        r32 = r[rows].float()
        norm = self.decoder.norm
        direction = self.head.weight[rivals].float() * norm.weight.float()
        direction = direction - r32 * (direction*r32).mean(-1,keepdim=True) / (
            r32.square().mean(-1,keepdim=True) + norm.variance_epsilon)
        values = self.visual_values[source][rows].float()
        direction = (direction @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        writes = torch.einsum('bhm,bhmd,bhd->bm', state['attention'][rows].float(), values, direction)
        positive = writes.clamp_min(0)
        evidence = positive / positive.sum(-1,keepdim=True).clamp_min(1e-12)
        similarity = (self.visual_normalized[source][rows] @ F.normalize(u[rows].float(),dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0,1)
        deficit = (evidence-preference).clamp_min(0)
        scale = 5 * drift * drift.new_tensor(strengths).tanh()
        budget = scale[:,None] * deficit / deficit.amax(-1,keepdim=True).clamp_min(1e-6)
        assert torch.isfinite(budget).all()
        effective = (budget.amax(-1) > 0).nonzero().flatten().tolist()
        if not effective:
            return [], None, []
        details = [dict(token=rivals[i], gain=ffn_gains[i], drift=float(drift[i]), budget_max=float(budget[i].max()))
                   for i in effective]
        return [rows[i] for i in effective], budget[effective], details

