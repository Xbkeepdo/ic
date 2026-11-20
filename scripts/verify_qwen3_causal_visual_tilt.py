"""CPU checks for actual-query attention and native/refreshed source routing."""

from pathlib import Path
from types import SimpleNamespace
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch

from features.qwen3_causal_visual_tilt import Qwen3BatchCausalVisualTilt, current_attention
from scripts.run_qwen3_signed_feedback import write


def verify():
    torch.manual_seed(43)
    query = torch.randn(3, 4, 1, 8)
    key, value = torch.randn(3, 2, 7, 8), torch.randn(3, 2, 7, 8)
    mask = torch.zeros(3, 1, 1, 7)
    mask[..., 0] = -torch.inf
    expanded_key, expanded_value = key.repeat_interleave(2, 1), value.repeat_interleave(2, 1)
    scores = (query @ expanded_key.transpose(-1, -2)) * 8**-.5 + mask
    reference = scores.softmax(-1)
    context = (reference @ expanded_value).transpose(1, 2).contiguous()
    rows = [2, 0]
    coefficient = torch.tensor([[.8, .3, .5], [.4, .2, .1]])
    zero = current_attention(query, key, value, mask, 8**-.5, 2, (2, 5), rows,
                             torch.zeros_like(coefficient), 1.)
    assert torch.equal(zero[0], context) and torch.equal(zero[1], reference)
    before = tuple(x.clone() for x in (query, key, value, mask, coefficient))
    result, probability, shift = current_attention(query, key, value, mask, 8**-.5,
                                                  2, (2, 5), rows, coefficient, 1.)
    assert torch.equal(result[1], context[1]) and torch.equal(probability[1], reference[1])
    assert (shift >= 0).all()
    assert (probability[rows, :, :, 2:5].sum(-1) > reference[rows, :, :, 2:5].sum(-1)).all()
    # The ratio between two unmasked nonvisual positions is unchanged.
    torch.testing.assert_close(probability[rows, :, :, 1]/probability[rows, :, :, 6],
                               reference[rows, :, :, 1]/reference[rows, :, :, 6])
    assert all(torch.equal(x, y) for x, y in zip(before, (query, key, value, mask, coefficient)))
    permutation = [0, 2]
    reordered = current_attention(query, key, value, mask, 8**-.5, 2, (2, 5),
                                   permutation, coefficient.flip(0), 1.)
    assert torch.equal(reordered[0], result)

    # Two orthogonal visual patches have hand-computable source distributions.
    # The refreshed attention and FFN preference both swap; native data must
    # stay intact. Using either stale tensor fails this symmetry check.
    owner = object.__new__(Qwen3BatchCausalVisualTilt)
    owner.tilt_mode = 'evidence'
    owner.final_tokens = torch.tensor([0])
    owner.refreshed_norm_calls = 0
    owner.start, owner.end = 0, 2
    eye = torch.eye(2)
    owner.reader = SimpleNamespace(write_norms={(20, 0): torch.tensor([.8, .2])},
        values={20: eye.reshape(1, 1, 2, 2)}, visual={20: eye[None]},
        columns={0: 0}, _static=lambda *_: {'log_s': torch.zeros(2, 1)})
    owner.scratch = {20: {'update': torch.tensor([[1., 0.]])}}
    owner.decoder = SimpleNamespace(layers={20: SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=eye)))})
    native, _ = owner._source_coefficients(20, [0])
    fresh, _ = owner._source_coefficients(20, [0], torch.tensor([[[.2, .8]]]),
                                          torch.tensor([[0., 1.]]))
    drift = 1-torch.sqrt(torch.tensor(.8))
    torch.testing.assert_close(native[0], torch.stack((drift*4/9, drift)), atol=1e-6, rtol=0)
    torch.testing.assert_close(fresh[0], native[0].flip(0), atol=1e-6, rtol=0)
    assert owner.refreshed_norm_calls == 1
    assert torch.equal(owner._source_coefficients(20, [0])[0], native)
    assert torch.equal(owner.scratch[20]['update'], torch.tensor([[1., 0.]]))
    return dict(passed=True,zero_operator_exact=True,unselected_rows_exact=True,
        visual_mass_increases=True,nonvisual_logit_ratios_preserved=True,
        reordered_rows_exact=True,inputs_not_modified=True,
        refreshed_attention_and_ffn_both_used=True,native_source_not_overwritten=True,
        scope='CPU routing/operator checks. Original108, causal timing and KV require GPU preflight.')


if __name__ == '__main__':
    output = REPO/'outputs/qwen3_causal_visual_tilt_cpu'
    output.mkdir(exist_ok=True)
    result = verify()
    write(output/'verification.json', result)
    print(__import__('json').dumps(result, indent=2))
