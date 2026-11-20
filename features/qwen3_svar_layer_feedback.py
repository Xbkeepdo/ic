"""SVAR base with the already verified adjacent-layer SAR/EMS feedback.

Only the base attention operation changes. Evidence/budgets remain 18 -> 19
and 19 -> 20 (zero based); the historical 21/22 late recipe is not used.
The original frozen modules are unchanged on disk.
"""
from contextlib import contextmanager
from unittest.mock import patch

from features import qwen3_sar_ems_layer_feedback as feedback


def svar_visual_logits(visual):
    """Exact original SVAR update, including its BF16 in-place arithmetic.

    Input is [batch, query_heads, visual_positions]. The mean is over heads
    separately for each patch. Returning the same view allows the inherited
    controller's subsequent copy_ call to preserve this original operation.
    """
    return visual.add_(0.5 * visual.abs().mean(dim=1, keepdim=True))


class Qwen3BatchSvarLayerFeedback(feedback.Qwen3BatchLayerFeedback):
    def __init__(self, model, span, nouns, strength=5.0, *,
                 disable_guidance=False, force_guidance=False):
        super().__init__(model, span, nouns, mode="next", strength=strength,
                         disable_guidance=disable_guidance, force_guidance=force_guidance)

    @contextmanager
    def capture(self):
        # The parent already scopes its process-global attention patch to one
        # generation. Scope the base operator substitution to that same context.
        with patch.object(feedback, "sadi_visual_logits", svar_visual_logits):
            with super().capture():
                yield
