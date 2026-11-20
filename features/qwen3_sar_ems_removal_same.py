"""Apply the visual-removal gate at a layer and replay its last query."""

from features.qwen3_sar_ems_removal_gate import Qwen3BatchWriteRemovalGate


class Qwen3BatchWriteRemovalSame(Qwen3BatchWriteRemovalGate):
    def __init__(self, model, span, nouns, *, layer=35):
        super().__init__(model, span, nouns)
        self.mode = 'same'
        self.sources = self.targets = (layer,)
