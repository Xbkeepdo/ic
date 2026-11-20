"""SADI on zero-based layers 5--12, with adjacent feedback 12->13 ... 17->18.

The original SADI and feedback controllers remain unchanged. These adapters
scope their layer selection to one capture context in one worker process.
"""

from contextlib import contextmanager
from unittest.mock import patch

from features import qwen3_sar_ems_attention_batch as batch_controller
from features import qwen3_sar_ems_layer_feedback as feedback


SADI_LAYERS = tuple(range(5, 13))
SOURCE_LAYERS = tuple(range(12, 18))
TARGET_LAYERS = tuple(range(13, 19))


class Qwen3BatchTruncatedSadi(batch_controller.Qwen3BatchLateAttention):
    """Original SADI batch implementation restricted to layers 5--12."""

    def __init__(self, model, span, nouns):
        super().__init__(model, span, nouns, "sadi")

    @contextmanager
    def capture(self):
        with patch.object(batch_controller, "ATTENTION_LAYERS", SADI_LAYERS):
            with super().capture():
                yield


class Qwen3BatchExpandedNext(feedback.Qwen3BatchLayerFeedback):
    """Refresh a separate budget at each source, then guide its next layer."""

    def __init__(self, model, span, nouns, strength=5.0, *,
                 disable_guidance=False, force_guidance=False):
        super().__init__(model, span, nouns, mode="next", strength=strength,
                         disable_guidance=disable_guidance,
                         force_guidance=force_guidance)
        self.sources = SOURCE_LAYERS
        self.targets = TARGET_LAYERS
        if len(self.decoder.layers) <= TARGET_LAYERS[-1]:
            raise ValueError("Expanded next feedback requires decoder layer 18")

    @contextmanager
    def capture(self):
        with patch.object(feedback, "ATTENTION_LAYERS", SADI_LAYERS):
            with super().capture():
                yield
