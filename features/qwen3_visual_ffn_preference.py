"""Read visual FFN outputs as an alternative preference representation."""

import torch
from torch.nn import functional as F
from features.qwen3_region_memory import RegionEvidenceReader
from features.qwen3_signal_disagreement_contrast import coupled_priors


def ffn_output_preference(visual_updates, query_update, tau=.03):
    assert visual_updates.ndim == 2 and query_update.shape == visual_updates.shape[1:]
    assert tau > 0 and torch.isfinite(visual_updates).all() and torch.isfinite(query_update).all()
    similarity = F.cosine_similarity(visual_updates.float(), query_update.float()[None], dim=-1, eps=1e-8)
    return (similarity/tau).softmax(0)


def output_priors(evidence, hpre_preference, output_preference):
    current = coupled_priors(evidence, output_preference)
    old = coupled_priors(evidence, hpre_preference)
    return dict(uv_excess=current['excess'], hpre_excess=old['excess'],
                uniform=current['uniform'], rolled_uv_excess=current['rolled_excess'])


class VisualFFNPreferenceReader(RegionEvidenceReader):
    def __init__(self, *args, tracked_rows=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked_rows = None if tracked_rows is None else tuple(tracked_rows)
        self.visual_updates = {}
        self.visual_capture_calls = 0

    def _ffn(self, index):
        original = super()._ffn(index)
        def read(module, args, value):
            if not self._ignore() and self.step == 0:
                assert value.ndim == 3 and value.shape[1] >= self.end
                rows = range(len(value)) if self.tracked_rows is None else self.tracked_rows
                for row in rows:
                    assert (index, row) not in self.visual_updates
                    self.visual_updates[index, row] = value[row, self.start:self.end].detach().clone()
                self.visual_capture_calls += 1
            original(module, args, value)
        return read

    def comparison_distributions(self, row, token):
        evidence, old, old_drift = super().distributions(row, token)
        current = ffn_output_preference(self.visual_updates[19, row], self.current[19]['update'][row])
        drift = float((1-(evidence*current).sqrt().sum()).clamp(0,1))
        return evidence, old, current, old_drift, drift
