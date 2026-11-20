"""Check the KL optimum, conservation, route signs, scale and rollback indexing."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_svar_attention_midpoint import conservative_attention_budget, Qwen3BatchSvarAttentionMidpoint
from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass


def verify():
    a = torch.tensor([[[.1, .2, .3, .4]]])
    delta = torch.tensor([[[.1, 0., 0., -.1]]])
    actual, _ = conservative_attention_budget(a, delta, 'geometric')

    def objective(x):
        q = np.array([x, .5 - x])
        return float((q * np.log(q / np.array([.1, .4]))).sum()
                     + (q * np.log(q / np.array([.2, .3]))).sum())

    optimum = minimize_scalar(objective, bounds=(1e-8, .5 - 1e-8), method='bounded', options={'xatol': 1e-12})
    assert optimum.success
    assert abs(float((a + actual)[0, 0, 0]) - optimum.x) < 1e-7
    torch.manual_seed(43)
    maximum_error = 0.
    for _ in range(20):
        attention = torch.randn(3, 4, 18).mul(3).softmax(-1).to(torch.bfloat16).float()
        rho = torch.rand(3, 1, 1) * .5
        donor = rho * attention[:, :, 12:]
        mass = donor.sum(-1, keepdim=True)
        destination = torch.rand(3, 4, 6).softmax(-1)
        budget = torch.zeros_like(attention)
        budget[:, :, 2:8] = mass * destination
        budget[:, :, 12:] = -donor
        for mode in ('half', 'geometric'):
            rewrite, _ = conservative_attention_budget(attention, budget, mode)
            assert (rewrite[budget == 0] == 0).all()
            assert (rewrite[budget > 0] >= 0).all() and (rewrite[budget < 0] <= 0).all()
            assert (attention + rewrite).min() >= -1e-7
            error = float(rewrite.sum(-1).abs().max())
            maximum_error = max(maximum_error, error)
            assert error < 1e-6
            for scale in (1e-8, 1e-14, 1e-24):
                scaled, _ = conservative_attention_budget(attention * scale, budget * scale, mode)
                torch.testing.assert_close(scaled / scale, rewrite, atol=2e-7, rtol=2e-4)
    obj = object.__new__(Qwen3BatchSvarAttentionMidpoint)
    obj.geometry_mode, obj.geometry_calls = 'geometric', 0
    obj.start, obj.end = 0, 2
    obj.scratch = {19: dict(scores=torch.zeros(3, 1, 1, 4), values=torch.zeros(3, 1, 4, 1))}
    obj.cache_backup = {20: (torch.tensor([42, 84]), torch.tensor([43, 85]))}

    def original_budget(self, source):
        changes = torch.tensor([[[1e-40, 0., -1e-40, 0.]], [[.01, 0., -.01, 0.]]])
        details = [dict(predicted_log_probability_change=-.01, moved_attention_mass=.01, budget_max=.01)
                   for _ in range(2)]
        return [2, 0], changes, details

    with patch.object(Qwen3BatchSvarFirstClass, '_budget', original_budget):
        rows, _, _ = obj._budget(19)
    assert rows == [0] and obj.cache_backup[20][0].tolist() == [84]
    return dict(passed=True, scipy_KL_optimum=optimum.x, maximum_mass_error=maximum_error,
        scales=[1, 1e-8, 1e-14, 1e-24], signed_routes_preserved=True,
        no_prompt_change=True, zeroed_row_cache_backup_alignment=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
