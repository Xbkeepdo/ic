"""Require visual opposition and FFN promotion for a new risky object class."""

from unittest.mock import patch

import torch

from features import qwen3_probe_attention_recycle as recycle
from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
from features.qwen3_visual_write_contrast import readout_contrast


class Qwen3BatchFirstClassConflict(Qwen3BatchProbeFirstClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conflict_decisions = {}
        self.contrast_cache_hits = 0

    def _budget(self, source):
        original_probe, original_features = self.probe, self.reader.features
        feature_rows = None
        contrasts, locations = None, {}

        def features(rows, tokens):
            nonlocal feature_rows
            feature_rows = list(rows)
            return original_features(rows, tokens)

        def guarded_probe(values):
            nonlocal contrasts, locations
            assert feature_rows is not None and len(values) == len(feature_rows)
            real = original_probe(values)
            selected = (real < .5).nonzero().flatten().tolist()
            allowed = torch.zeros_like(real, dtype=torch.bool)
            if selected:
                rows = [feature_rows[i] for i in selected]
                targets = self.final_tokens[rows, None]
                contrasts = readout_contrast(self, source, rows, targets)
                locations = {row: index for index, row in enumerate(rows)}
                support, gain = contrasts[:2]
                permitted = (support[:, 0] < 0) & (gain[:, 0] > 0)
                if self.force_probe_gate:
                    permitted[:] = True
                allowed[selected] = permitted
                for index, slot in enumerate(selected):
                    row = feature_rows[slot]
                    self.conflict_decisions.setdefault(row, []).append(dict(step=self.step,
                        token=int(self.final_tokens[row]), probe_p_real=float(real[slot]),
                        visual_support=float(support[index, 0]), gain=float(gain[index, 0]),
                        permitted=bool(permitted[index])))
            return torch.where(allowed, real, torch.ones_like(real))

        def cached_contrast(controller, index, rows, targets):
            if controller is self and index == source and all(row in locations for row in rows):
                assert targets[:, 0].tolist() == self.final_tokens[rows].tolist()
                self.contrast_cache_hits += 1
                positions = [locations[row] for row in rows]
                return tuple(values[positions] for values in contrasts)
            return readout_contrast(controller, index, rows, targets)

        self.probe, self.reader.features = guarded_probe, features
        try:
            # Reuse the sign readout in the unchanged attention action.
            with patch.object(recycle, 'readout_contrast', cached_contrast):
                return super()._budget(source)
        finally:
            self.probe, self.reader.features = original_probe, original_features

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            decisions = [d for d in self.conflict_decisions.get(row, ()) if d['step'] < len(tokens)]
            stats.update(conflict_sign_gate=True, require_negative_visual_support=True,
                require_positive_ffn_gain=True, conflict_decisions=decisions,
                raw_probe_positive_calls=len(decisions),
                blocked_by_conflict_sign=sum(not d['permitted'] for d in decisions),
                contrast_readout_cache_hits=self.contrast_cache_hits)
        return result
