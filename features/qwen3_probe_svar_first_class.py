"""Scope the existing first-class action to the repository's native SVAR base."""

from contextlib import contextmanager
from unittest.mock import patch

from features import qwen3_sar_ems_layer_feedback as feedback
from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
from features.qwen3_svar_layer_feedback import svar_visual_logits


class Qwen3BatchSvarFirstClass(Qwen3BatchProbeFirstClass):
    @contextmanager
    def capture(self):
        with patch.object(feedback, 'sadi_visual_logits', svar_visual_logits):
            with super().capture():
                yield self

    def statistics(self, responses):
        result = super().statistics(responses)
        for stats in result:
            stats.update(base_variant='svar', local_I_before_base_edit='svar')
        return result
