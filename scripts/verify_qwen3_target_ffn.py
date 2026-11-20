"""Check the extra FFN derivative independently of the attention derivative."""

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_probe_early_recycle import Qwen3BatchProbeEarlyRecycle
from features.qwen3_target_ffn import Qwen3BatchTargetFFN, target_promoting_component


def verify():
    update = torch.tensor([[2., -3., 4., -5.], [0., 0., 0., 0.]])
    gradient = torch.tensor([[1., 2., -1., -2.], [1., -1., 1., -1.]])
    component, attribution = target_promoting_component(update, gradient)
    assert torch.equal(component, torch.tensor([[2., 0., 0., -5.], [0., 0., 0., 0.]]))
    assert torch.equal(component * gradient, attribution.clamp_min(0))
    for scale in (1e-8, 1., 1e8):
        scaled, _ = target_promoting_component(update, gradient * scale)
        assert torch.equal(scaled, component)
    assert torch.equal(target_promoting_component(update, torch.zeros_like(update))[0], torch.zeros_like(update))

    torch.manual_seed(43)
    width, batch = 5, 3

    class Tail(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(width, width, bias=False)
        def forward(self, hidden, **_kwargs):
            return hidden + .03 * self.linear(hidden).tanh()

    layer = SimpleNamespace(mlp=torch.nn.Sequential(torch.nn.Linear(width, 8), torch.nn.Tanh(),
        torch.nn.Linear(8, width)), post_attention_layernorm=torch.nn.RMSNorm(width, eps=1e-6))
    decoder = SimpleNamespace(layers=[None]*19+[layer]+[Tail() for _ in range(16)],
        norm=torch.nn.RMSNorm(width, eps=1e-6))
    obj = object.__new__(Qwen3BatchTargetFFN)
    residual = torch.randn(batch, width)
    obj.scratch = {19: {'ffn_residual': residual}}
    obj.decoder, obj.head = decoder, torch.nn.Linear(width, 7, bias=False)
    obj.tail_kwargs = dict(past_key_values=SimpleNamespace(layers=[]),
        position_embeddings=(torch.ones(1), torch.ones(1)), cache_position=torch.tensor([8]))
    obj.final_tokens = torch.tensor([1, 3, 5])
    obj.maximum_probe_hidden_difference = 0.
    obj.gradient_probes = obj.ffn_output_gradient_reads = 0
    rows = [2, 0]

    def forward(added=None):
        ffn = layer.mlp(layer.post_attention_layernorm(residual[:, None]))
        if added is not None:
            ffn = ffn + added
        current = residual[:, None] + ffn
        for tail in decoder.layers[20:]:
            current = tail(current)
        normalized = decoder.norm(current)
        objective = obj.head(normalized).float()[:, 0].log_softmax(-1)[rows, obj.final_tokens[rows]].sum()
        return objective, normalized

    with torch.no_grad():
        obj.reference_hidden = forward()[1]
        obj.scratch[19]['update'] = layer.mlp(layer.post_attention_layernorm(residual[:, None]))[:, 0]
    expected = Qwen3BatchProbeEarlyRecycle._gradient(obj, 19, rows)
    actual = obj._gradient(19, rows)
    assert torch.equal(expected, actual), (expected - actual).abs().max()
    maximum_error = 0.
    delta = .002
    for position, row in enumerate(rows):
        for column in range(width):
            plus = torch.zeros(batch, 1, width)
            plus[row, 0, column] = delta
            derivative = float((forward(plus)[0] - forward(-plus)[0]) / (2*delta))
            observed = float(obj.ffn_output_gradient[position, column])
            maximum_error = max(maximum_error, abs(derivative - observed))
    assert maximum_error < 3e-4, maximum_error
    assert obj.ffn_gradient_rows == (2, 0) and obj.maximum_probe_hidden_difference == 0
    assert not obj.replaying_tail and not obj._replaying
    return dict(passed=True, residual_gradient_bitwise_matches_parent=True,
        ffn_output_gradient_finite_difference_max_error=maximum_error,
        reversed_sparse_rows=True, empty_and_signed_coordinate_cases=True,
        positive_gradient_scale_invariance=True, per_probe_backward_calls=1,
        scope='CPU derivative/mathematical checks, not GPU generation or efficacy.')


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
