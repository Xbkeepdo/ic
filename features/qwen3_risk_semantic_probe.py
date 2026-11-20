"""Frozen risk/E_l probe, with an explicit projection of native108 vectors."""

import torch

from features.qwen3_online_i_el import FullIElProbe


class FullRiskSemanticProbe(FullIElProbe):
    @torch.inference_mode()
    def forward(self, features):
        if features.shape[-1] == 108:
            features = torch.cat((features[..., :36], features[..., 72:108]), -1)
        assert features.shape[-1] == 72
        return super().forward(features)
