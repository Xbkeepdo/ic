from __future__ import annotations

import numpy as np
import torch

from features.dgst_t import _solve_transport_problem
from features.logs_risk_f_log import _transport_problem, _validate_settings


def test_endpoint_shift_matches_the_full_reference_cost():
    source = torch.tensor([0.55, 0.25, 0.15, 0.05], dtype=torch.float32)
    target = torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=torch.float32)
    visual = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
        dtype=torch.float32,
    )
    query = torch.tensor([0.25, 1.0], dtype=torch.float32)

    problem, shift, support_size = _transport_problem(
        source=source,
        target=target,
        visual_states=visual,
        prediction_state=query,
        top_k=4,
    )
    local_source, local_target, visual_cost = problem
    # top_k=4 means source Top-2 union target Top-2.  Reconstruct the endpoint
    # terms directly on that support and compare one exact EMD solve.
    support = torch.unique(
        torch.cat((torch.topk(source, 2).indices, torch.topk(target, 2).indices)),
        sorted=True,
    )
    penalty = 1.0 - torch.nn.functional.cosine_similarity(
        query.unsqueeze(0), visual, dim=-1
    )
    local_penalty = penalty.index_select(0, support).numpy()
    full_cost = (
        visual_cost
        + local_penalty[:, None]
        + local_penalty[None, :]
    )

    shifted = _solve_transport_problem(problem, "emd") + shift
    direct = _solve_transport_problem(
        (local_source, local_target, full_cost), "emd"
    )
    assert support_size <= 4
    assert np.isclose(shifted, direct, rtol=1e-6, atol=1e-7)


def test_selected_settings_require_exact_emd_and_even_total_budget():
    _validate_settings(0.03, 32, "emd")
    for args in ((0.0, 32, "emd"), (0.03, 31, "emd"), (0.03, 32, "sinkhorn")):
        try:
            _validate_settings(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid settings to fail: {args}")
