"""Check destination-only isolation and sparse effective-row alignment on CPU."""

from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features import qwen3_probe_attention_recycle as recycle
from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass
from features.qwen3_svar_q_destination import Qwen3BatchSvarQDestination, negative_write_destination


def verify():
    # A fully specified one-head example: reference uses FFN factor[1,.25]
    # on Q=[.25,.75], while the ablation must move exactly[.05,.15].
    reference = torch.tensor([[[0., .2 * 4 / 7, .2 * 3 / 7, 0., -.12, -.08]]])
    q = torch.tensor([[[.25, .75]]])
    actual = negative_write_destination(reference, q, 1, 3)
    torch.testing.assert_close(actual, torch.tensor([[[0., .05, .15, 0., -.12, -.08]]]), atol=1e-7, rtol=0)
    effects = torch.tensor([[[0., -2., -1., 0., 1., 3.]]])
    assert abs(float((actual * effects).sum()) + .61) < 1e-7
    maximum_mass_error = 0.
    torch.manual_seed(43)
    for _ in range(20):
        attention = torch.randn(3, 4, 18).softmax(-1).to(torch.bfloat16).float()
        effects = torch.randn_like(attention)
        negative = (-attention[:, :, 2:8] * effects[:, :, 2:8]).clamp_min(0)
        weights = recycle.normalize_nonnegative(negative)
        valid = weights.sum(-1, keepdim=True) > 0
        donors = -.4 * attention[:, :, 12:] * (effects[:, :, 12:] > 0) * valid
        mass = -donors.sum(-1, keepdim=True)
        reference = torch.zeros_like(attention)
        reference[:, :, 2:8] = mass * recycle.normalize_nonnegative(weights * torch.rand_like(weights).add(.25))
        reference[:, :, 12:] = donors
        assert (reference.abs().amax((1, 2)) > 0).all()
        actual = negative_write_destination(reference, weights, 2, 8)
        assert (attention + actual).min() >= -1e-7 and (actual * effects).sum((1, 2)).max() <= 1e-6
        torch.testing.assert_close(actual[:, :, 2:8].sum(-1), mass[..., 0], atol=1e-7, rtol=1e-6)
        maximum_mass_error = max(maximum_mass_error, float(actual.sum(-1).abs().max()))
        for scale in (1e-8, 1e-14, 1e-24):
            scaled = negative_write_destination(reference * scale, weights, 2, 8)
            torch.testing.assert_close(scaled / scale, actual, atol=1e-7, rtol=1e-5)
    obj = object.__new__(Qwen3BatchSvarQDestination)
    obj.start, obj.end, obj.negative_destination_captures = 1, 3, 0
    gradient_calls = []

    def gradient(index, rows):
        gradient_calls.append((index, rows))

    obj._gradient = gradient
    original_normalize = recycle.normalize_nonnegative

    def original_budget(self, source):
        # One ineffective middle row; parent returns reversed effective rows.
        self._gradient(source, [2, 1, 0])
        negative = torch.tensor([[[1., 3.]], [[0., 0.]], [[3., 1.]]])
        recycle.normalize_nonnegative(negative.sum(1))
        first = recycle.normalize_nonnegative(negative)
        destination = recycle.normalize_nonnegative(first * torch.tensor([1., .25]))
        budget = torch.zeros(2, 1, 6)
        budget[:, :, 1:3] = .2 * destination[[2, 0]]
        budget[:, :, 4:] = torch.tensor([-.12, -.08])
        details = [dict(predicted_log_probability_change=-.1, moved_attention_mass=.2, budget_max=.2) for _ in range(2)]
        return [0, 2], budget, details

    with patch.object(Qwen3BatchSvarFirstClass, '_budget', original_budget):
        rows, budget, details = obj._budget(19)
    assert rows == [0, 2] and gradient_calls == [(19, [2, 1, 0])]
    torch.testing.assert_close(budget[:, 0, 1:3], torch.tensor([[.15, .05], [.05, .15]]), atol=1e-7, rtol=0)
    assert obj._gradient is gradient and recycle.normalize_nonnegative is original_normalize
    assert obj.negative_destination_captures == 1
    assert all(d['maximum_nonvisual_budget_difference'] == 0 and d['first_order_log_probability_change_upper_bound'] == 0
               and 'predicted_log_probability_change' not in d for d in details)

    def failed_budget(_self, _source):
        raise RuntimeError('simulated aborted parent proposal')

    try:
        with patch.object(Qwen3BatchSvarFirstClass, '_budget', failed_budget):
            obj._budget(19)
    except RuntimeError as exc:
        assert str(exc) == 'simulated aborted parent proposal'
    else:
        raise AssertionError('Expected simulated parent failure')
    assert obj._gradient is gradient and recycle.normalize_nonnegative is original_normalize
    return dict(passed=True, closed_form_first_order_change=-.61, maximum_mass_error=maximum_mass_error,
                scales=[1, 1e-8, 1e-14, 1e-24], nonvisual_entries_exact=True,
                reordered_effective_rows_correct=True, gradient_calls_in_one_proposal=1,
                wrappers_restored_after_failure=True, scope='CPU math/integration only, not a GPU decoding or efficacy result.')


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
