"""Compare uniform visual attention restoration with target-conditioned redistribution."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F

from features import qwen3_sar_ems_layer_feedback as feedback
from features.llava_decoding_evidence import TAU


class Qwen3BatchLowVisualFeedback(feedback.Qwen3BatchLayerFeedback):
    """Guide layer 29 when layer 28 uses little visual attention for a new entity."""

    def __init__(self, model, span, nouns, *, spatial):
        super().__init__(model, span, nouns)
        self.sources, self.targets = (28,), (29,)
        self.spatial = spatial
        self.seen = []
        self.projection = None

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
        attention = state['attention'].float()
        mass = attention.sum(-1).mean(-1)
        # The trace's real-noun 20th percentile was 0.119 at layer 28.
        gate = 0.5 * (1 - mass / 0.12).clamp(0, 1)
        rows = (self.active & (gate > 0)).nonzero().flatten().tolist()
        if not rows:
            return [], None, []

        r = state['ffn_residual'][rows]
        u = state['update'][rows]
        norm = self.decoder.norm
        post = norm(r + u)
        logits = self.head(post).float()
        scores, ids = logits.topk(8, dim=-1)
        kept = []
        for position, (row, words, values) in enumerate(zip(rows, ids.tolist(), scores.tolist(), strict=True)):
            target = next((t for t, value in zip(words, values, strict=True)
                           if values[0] - value <= 2 and t in self.nouns), None)
            # Token history is causal. This excludes repeats of the same token
            # ID, but cannot see CHAIR canonical class or word boundaries.
            if target is not None and target not in self.seen[row]:
                kept.append((position, row, target))
        if not kept:
            return [], None, []
        positions, rows, targets = map(list, zip(*kept, strict=True))
        weights = self.head.weight[targets].float()
        gain = (weights * (post[positions].float() - norm(r[positions]).float())).sum(-1)
        spatial_budget = mass.new_zeros((len(rows), self.end-self.start))
        drift = mass.new_zeros(len(rows))
        if self.spatial:
            r = r[positions].float()
            direction = weights * norm.weight.float()
            direction = direction - r * (r * direction).mean(-1, keepdim=True) / (
                r.square().mean(-1, keepdim=True) + norm.variance_epsilon)
            if self.projection is None:
                self.projection = self.decoder.layers[source].self_attn.o_proj.weight.float()
            values = self.visual_values[source][rows].float()
            direction = (direction @ self.projection).reshape(len(rows), values.shape[1], -1)
            writes = torch.einsum('bhm,bhmd,bhd->bm', attention[rows], values, direction)
            positive = writes.clamp_min(0)
            evidence = positive / positive.sum(-1,keepdim=True).clamp_min(1e-12)
            similarity = (self.visual_normalized[source][rows] @ F.normalize(u[positions].float(),dim=-1).unsqueeze(-1)).squeeze(-1)
            preference = (similarity / TAU).softmax(-1)
            drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
            deficit = (evidence - preference).clamp_min(0)
            scale = 2.0 * drift * (1 - mass[rows]/0.12).clamp(0,1)
            spatial_budget = scale[:,None] * deficit / deficit.amax(-1,keepdim=True).clamp_min(1e-6)

        budget = torch.cat([spatial_budget, gate[rows,None]],dim=-1)
        if not torch.isfinite(budget).all().item():
            raise RuntimeError('Non-finite visual attention budget')
        info = [dict(token=token, gain=float(g), drift=float(d),budget_max=float(b))
                for token,g,d,b in zip(targets,gain.tolist(),drift.tolist(),budget.amax(-1).tolist(),strict=True)]
        return rows,budget,info

    @contextmanager
    def capture(self):
        def rebalance(logits, budget):
            visual = logits.float()
            spread = visual.abs().std(dim=1, correction=0)
            scale = 0.25 + 0.75 * spread / spread.amax(dim=-1, keepdim=True).clamp_min(1e-6)
            adjusted = visual + (scale * budget[:,:-1])[:,None,:]
            normalized = adjusted - (adjusted.logsumexp(-1,keepdim=True)-visual.logsumexp(-1,keepdim=True))
            return (normalized + budget[:,-1,None,None]).to(logits.dtype)

        # Keep the common boost in float32 until the final BF16 cast, so only
        # spatial reallocation differs at a given fixed state.
        with patch.object(feedback,'rebalance_visual_logits',rebalance):
            with super().capture():
                yield
