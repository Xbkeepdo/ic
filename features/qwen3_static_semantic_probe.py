"""Use the existing E_l-only probe without attention-dependent inputs."""

from pathlib import Path

import numpy as np
import torch

from features.qwen3_online_i_el import FullIElProbe


class FullSemanticProbe(FullIElProbe):
    def __init__(self, directory):
        torch.nn.Module.__init__(self)
        directory = Path(directory)
        state = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
        with np.load(directory/'input_scaler.npz', allow_pickle=False) as scaler:
            for key in ('mean', 'scale'):
                self.register_buffer(key, torch.from_numpy(scaler[key].copy()))
        for name, key in (('weight0', 'net.0.weight'), ('bias0', 'net.0.bias'),
                          ('weight1', 'net.3.weight'), ('bias1', 'net.3.bias')):
            self.register_buffer(name, state[key])
        assert self.weight0.shape == (128, 36) and self.weight1.shape == (1, 128)
        assert self.mean.shape == self.scale.shape == (36,)
