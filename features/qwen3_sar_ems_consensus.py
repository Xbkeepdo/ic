"""Apply later SAR/EMS guidance only when neighboring FFNs nominate the same token."""

from features.qwen3_sar_ems_signed import Qwen3BatchSignedLayerFeedback


class Qwen3BatchConsensusFeedback(Qwen3BatchSignedLayerFeedback):
    def __init__(self, model, span, nouns, strength=5.0):
        super().__init__(model, span, nouns, strength=strength, directional_evidence=True)
        self.sources = (24, 25)
        self.targets = (25, 26)
        self.nominations = {}

    def _budget(self, source):
        rows, budgets, details = super()._budget(source)
        if source == self.sources[0]:
            # Record candidates that had positive support, FFN gain and drift.
            # Layer 25 observes the ordinary SADI trajectory.
            self.nominations = {row: detail['token'] for row, detail in zip(rows, details, strict=True)}
            return [], None, []
        keep = [i for i, (row, detail) in enumerate(zip(rows, details, strict=True))
                if self.nominations.get(row) == detail['token']]
        self.nominations.clear()
        if not keep:
            return [], None, []
        return [rows[i] for i in keep], budgets[keep], [details[i] for i in keep]
