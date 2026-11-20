"""Keep the SVAR attention action while replacing only its risk gate."""

from contextlib import contextmanager
from unittest.mock import patch

from features import qwen3_sar_ems_layer_feedback as feedback
from features.qwen3_probe_matched_hash import Qwen3BatchMatchedHashGate
from features.qwen3_svar_layer_feedback import svar_visual_logits


class Qwen3BatchSvarMatchedHash(Qwen3BatchMatchedHashGate):
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
