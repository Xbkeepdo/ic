"""Require visual opposition and FFN promotion before probe-guided recycling."""

import torch

from features.qwen3_probe_attention_recycle import Qwen3BatchProbeAttentionRecycle
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchProbeConflictGuard(Qwen3BatchProbeAttentionRecycle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_positive = {}
        self.conflict_blocked = {}

    def _budget(self, source):
        rows = [i for i, (token, active) in enumerate(
            zip(self.final_tokens.tolist(), self.active.tolist(), strict=True))
            if active and token in self.nouns and token not in self.seen[i]]
        if not rows:
            return [], None, []
        original_probe = self.probe

        def guarded_probe(features):
            real = original_probe(features)
            selected = (real < .5).nonzero().flatten().tolist()
            permitted = torch.zeros_like(real, dtype=torch.bool)
            if selected:
                candidates = [rows[i] for i in selected]
                visual, gain, _, _ = readout_contrast(
                    self, source, candidates, self.final_tokens[candidates, None])
                valid = (visual[:, 0] < 0) & (gain[:, 0] > 0)
                permitted[selected] = valid
                for slot, valid in zip(selected, valid.tolist(), strict=True):
                    row = rows[slot]
                    self.raw_positive[row] = self.raw_positive.get(row, 0) + 1
                    self.conflict_blocked[row] = self.conflict_blocked.get(row, 0) + int(not valid)
            return torch.where(permitted, real, torch.ones_like(real))

        self.probe = guarded_probe
        try:
            return super()._budget(source)
        finally:
            self.probe = original_probe

    def statistics(self, responses):
        result = super().statistics(responses)
        fields = ('step', 'token', 'probe_p_real', 'visual_support', 'gain', 'drift',
                  'moved_attention_mass', 'actual_log_probability_change', 'accepted')
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [e for e in self.events.get(row, ()) if e['step'] < len(tokens)]
            stats.update(conflict_guard=True, require_negative_visual_support=True,
                         require_positive_ffn_gain=True,
                         raw_probe_positive_calls=self.raw_positive.get(row, 0),
                         blocked_by_conflict_guard=self.conflict_blocked.get(row, 0),
                         recycle_events=[dict({k: e[k] for k in fields}, generated_token=tokens[e['step']])
                                         for e in events if 'accepted' in e])
        return result
