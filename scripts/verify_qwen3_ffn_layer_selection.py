"""Independently check signed routing and four-layer derivatives in one pass."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from features.qwen3_ffn_layer_selection import LAYERS, Qwen3BatchFFNLayerSelection, signed_candidate
from features.qwen3_probe_early_recycle import Qwen3BatchProbeEarlyRecycle


def verify():
    attention = torch.tensor([[[.1, .1, .1, .1, .3, .3]]])
    values = torch.zeros(1, 1, 6, 2)
    values[0, 0, :, 0] = torch.tensor([0., -2., -1., 0., 1., 3.])
    gradient = torch.tensor([[1., 0.]])
    patches = torch.eye(2)[None]
    update = torch.tensor([[1., 0.]])
    delta, info = signed_candidate(attention.log(), values, gradient, torch.eye(2), patches,
        update, gradient, 1, 3, 4)
    drift = 1-(2/3)**.5
    mass = .3*drift
    expected = torch.tensor([[[0., mass/3, 2*mass/3, 0., -mass/2, -mass/2]]])
    torch.testing.assert_close(delta, expected, atol=1e-7, rtol=0)
    assert abs(float(info['score'][0])-drift) < 1e-7
    assert abs(float(info['predicted_log_probability_change'][0])+10*mass/3) < 1e-7
    for scale in (1e-8, 1e-14, 1e-24):
        actual, scaled = signed_candidate(attention.log(), values, gradient*scale, torch.eye(2),
            patches, update, gradient*scale, 1, 3, 4)
        torch.testing.assert_close(actual, delta, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(scaled['score']/scale, info['score'], atol=1e-7, rtol=1e-6)
    _, opposite = signed_candidate(attention.log(), values, gradient, torch.eye(2), patches,
        update, -gradient, 1, 3, 4)
    assert opposite['score'][0] == 0

    torch.manual_seed(43)
    width, batch = 5, 3

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = torch.nn.RMSNorm(width, eps=1e-6)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(width, 8), torch.nn.Tanh(), torch.nn.Linear(8, width))
            self.attention = torch.nn.Linear(width, width, bias=False)
            self.add_residual = self.add_ffn = None
        def forward(self, hidden, **_kwargs):
            residual = hidden + .03*self.attention(hidden).tanh()
            if self.add_residual is not None:
                residual = residual + self.add_residual
            update = self.mlp(self.post_attention_layernorm(residual))
            if self.add_ffn is not None:
                update = update + self.add_ffn
            return residual + update

    decoder = SimpleNamespace(layers=[None]*19+[Layer() for _ in range(17)],
        norm=torch.nn.RMSNorm(width, eps=1e-6))
    obj = object.__new__(Qwen3BatchFFNLayerSelection)
    residual = torch.randn(batch, width)
    obj.decoder, obj.head = decoder, torch.nn.Linear(width, 7, bias=False)
    obj.scratch = {19: {'ffn_residual': residual}}
    obj.tail_kwargs = dict(past_key_values=SimpleNamespace(layers=[]),
        position_embeddings=(torch.ones(1), torch.ones(1)), cache_position=torch.tensor([8]))
    obj.final_tokens = torch.tensor([1, 3, 5])
    obj.enabled, obj.maximum_probe_hidden_difference, obj.gradient_probes = True, 0., 0
    rows = [2, 0]

    def forward(source=None, kind=None, perturbation=None, capture=False):
        layer = decoder.layers[19]
        before = residual[:, None]
        if source == 19 and kind == 'residual':
            before = before + perturbation
        update = layer.mlp(layer.post_attention_layernorm(before))
        if source == 19 and kind == 'ffn':
            update = update + perturbation
        if capture:
            obj.scratch[19]['update'] = update[:, 0].detach()
        current = before + update
        with ExitStack() as stack:
            for index in LAYERS[1:]:
                if capture:
                    obj.scratch[index] = {}
                    def save_input(_m, args, i=index):
                        obj.scratch[i]['ffn_residual'] = args[0][:, 0].detach()
                    def save_output(_m, _args, out, i=index):
                        obj.scratch[i]['update'] = out[:, 0].detach()
                    stack.callback(decoder.layers[index].post_attention_layernorm.register_forward_pre_hook(save_input).remove)
                    stack.callback(decoder.layers[index].mlp.register_forward_hook(save_output).remove)
            if source is not None and source > 19:
                setattr(decoder.layers[source], 'add_'+kind, perturbation)
                stack.callback(setattr, decoder.layers[source], 'add_'+kind, None)
            for tail in decoder.layers[20:]:
                current = tail(current)
        normalized = decoder.norm(current)
        objective = obj.head(normalized).float()[:, 0].log_softmax(-1)[rows, obj.final_tokens[rows]].sum()
        return objective, normalized

    with torch.no_grad():
        obj.reference_hidden = forward(capture=True)[1]
    expected = Qwen3BatchProbeEarlyRecycle._gradient(obj, 19, rows)
    actual = obj._gradient(19, rows)
    assert torch.equal(expected, actual), dict(max_abs=float((expected-actual).abs().max()), expected=expected.tolist(), actual=actual.tolist())
    assert obj.gradient_rows == (2, 0) and set(obj.layer_gradients) == set(LAYERS)
    maximum_error, step = 0., .002
    with torch.no_grad():
        for layer in LAYERS:
            for kind, index in (('residual', 0), ('ffn', 1)):
                for position, row in enumerate(rows):
                    for column in range(width):
                        perturbation = torch.zeros(batch, 1, width)
                        perturbation[row, 0, column] = step
                        finite = float((forward(layer, kind, perturbation)[0]-forward(layer, kind, -perturbation)[0])/(2*step))
                        maximum_error = max(maximum_error, abs(finite-float(obj.layer_gradients[layer][index][position, column])))
    assert maximum_error < 6e-4, maximum_error
    counts = [(len(decoder.layers[i].post_attention_layernorm._forward_pre_hooks), len(decoder.layers[i].mlp._forward_hooks)) for i in LAYERS]
    original = obj.reference_hidden
    obj.reference_hidden = original + 1
    try:
        obj._gradient(19, rows)
    except AssertionError:
        pass
    else:
        raise AssertionError('Expected deliberate native-state mismatch')
    finally:
        obj.reference_hidden = original
    assert obj.enabled and not obj._replaying and not obj.replaying_tail
    assert counts == [(len(decoder.layers[i].post_attention_layernorm._forward_pre_hooks), len(decoder.layers[i].mlp._forward_hooks)) for i in LAYERS]
    return dict(passed=True, closed_form_routing=True, residual19_gradient_bitwise_matches_parent=True,
        layers=list(LAYERS), derivative_types=['attention_residual', 'FFN_output'],
        finite_difference_max_error=maximum_error, sparse_reordered_rows=True,
        global_gradient_scale_invariance=True, nonpromoting_ffn_score_zero=True,
        wrappers_and_flags_restored_on_failure=True, backward_calls_per_probe=1,
        scope='CPU derivative/math checks; GPU parity, memory, mixed-layer replay and efficacy still separate.')


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
