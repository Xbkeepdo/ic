"""Fuse the frozen SAM-V2 and image-only semantic grounding scores.

The equal log-odds mean is a score, not a calibrated posterior. Its
operating point is calibrated only on the archived source real examples.
"""

import torch
from features.qwen3_risk_i_el_probe import FullRiskIElProbe
from features.qwen3_static_semantic_probe import FullSemanticProbe


def consensus_real_score(native, semantic):
    assert native.shape == semantic.shape
    assert torch.isfinite(native).all() and torch.isfinite(semantic).all()
    assert ((native >= 0) & (native <= 1)).all()
    assert ((semantic >= 0) & (semantic <= 1)).all()
    a = native.double().clamp(1e-6, 1-1e-6)
    b = semantic.double().clamp(1e-6, 1-1e-6)
    return ((torch.logit(a) + torch.logit(b)) * .5).sigmoid()


class FullSemanticConsensusProbe(torch.nn.Module):
    def __init__(self, native_directory, semantic_directory):
        super().__init__()
        self.native = FullRiskIElProbe(native_directory)
        self.semantic = FullSemanticProbe(semantic_directory)

    @torch.inference_mode()
    def forward(self, features):
        assert features.shape[-1] == 108
        return consensus_real_score(self.native(features), self.semantic(features[..., 72:108]))
