"""CPU checks of visual-mass matching, directionality and exact null controls."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from features.qwen3_mass_matched_visual import matched_visual_coefficients, mass_matched_attention


def verify(output):
    assert not output.exists()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(43)
    max_residual = max_mass_difference = max_nonvisual_difference = 0.
    for scale in (.01, 1., 10., 50.):
        scores = torch.randn(4, 5, 1, 27, generator=generator)*scale
        values = torch.randn(4, 5, 27, 7, generator=generator)
        rows = [3, 1]; span = (4, 21)
        contrast = torch.rand(2, 17, generator=generator)*2-1
        old_scores, old_values = scores.clone(), values.clone()
        context, probability, coefficient, residual = mass_matched_attention(scores, values, span, rows, contrast)
        assert ((coefficient >= 0) & (coefficient <= 1)).all()
        fixed = scores.clone(); fixed[rows, :, 0, 4:21] += .5*fixed[rows, :, 0, 4:21].abs()
        reference = fixed.softmax(-1, dtype=torch.float32)
        mass_difference = float((probability[rows, :, 0, 4:21].sum(-1)-reference[rows, :, 0, 4:21].sum(-1)).abs().max())
        nonvisual = [*range(4), *range(21, 27)]
        nonvisual_difference = float((probability[rows][:, :, 0, nonvisual]-reference[rows][:, :, 0, nonvisual]).abs().max())
        max_residual = max(max_residual, float(residual.abs().max()))
        max_mass_difference = max(max_mass_difference, mass_difference)
        max_nonvisual_difference = max(max_nonvisual_difference, nonvisual_difference)
        assert mass_difference < 2e-6 and nonvisual_difference < 2e-6
        untouched = [0, 2]
        native = scores.softmax(-1)
        assert torch.equal(probability[untouched], native[untouched])
        assert torch.equal(scores, old_scores) and torch.equal(values, old_values)
        reversed_result = mass_matched_attention(scores, values, span, rows[::-1], contrast.flip(0))
        assert torch.equal(reversed_result[0], context) and torch.equal(reversed_result[1], probability)
        zero_result = mass_matched_attention(scores, values, span, rows, torch.zeros_like(contrast))
        assert torch.equal(zero_result[1], reference)
        assert torch.equal(zero_result[0], (reference @ values).transpose(1, 2).contiguous())
        constant_result = mass_matched_attention(scores, values, span, rows, torch.full_like(contrast, .7))
        assert torch.equal(constant_result[0], zero_result[0]) and torch.equal(constant_result[1], reference)
    # Equal source logits: the supported patch gains, the other loses, at
    # identical total visual mass. Reversing the evidence reverses the move.
    logits = torch.tensor([[[-2., -2.]]])
    positive, residual = matched_visual_coefficients(logits, torch.tensor([[1., -1.]]))
    negative, _ = matched_visual_coefficients(logits, torch.tensor([[-1., 1.]]))
    assert positive[0, 0, 0] > .5 > positive[0, 0, 1]
    assert torch.equal(positive.flip(-1), negative)
    zero, zero_residual = matched_visual_coefficients(torch.zeros(2, 3, 4), torch.randn(2, 4, generator=generator))
    assert torch.equal(zero, torch.full_like(zero, .5)) and torch.equal(zero_residual, torch.zeros_like(zero_residual))
    # Actual eager decoding casts the FP32 softmax to BF16 before @V. Measure
    # that rounding separately; it is not exact probability-mass conservation.
    scores = torch.randn(3, 4, 1, 37, generator=generator).bfloat16()
    values = torch.randn(3, 4, 37, 5, generator=generator).bfloat16()
    contrast = torch.rand(2, 23, generator=generator)*2-1
    rows, span = [0, 2], (5, 28)
    context, probability, coefficient, _ = mass_matched_attention(scores, values, span, rows, contrast)
    fixed = scores.float().clone(); fixed[rows, :, 0, 5:28] += .5*fixed[rows, :, 0, 5:28].abs()
    fixed_probability = fixed.softmax(-1).to(values.dtype)
    bf16_mass_difference = float((probability[rows, :, 0, 5:28].float().sum(-1)
                                 -fixed_probability[rows, :, 0, 5:28].float().sum(-1)).abs().max())
    assert bf16_mass_difference <= 2**-7
    zero_result = mass_matched_attention(scores, values, span, rows, torch.zeros_like(contrast))
    assert torch.equal(zero_result[1], fixed_probability)
    assert torch.equal(zero_result[0], (fixed_probability @ values).transpose(1, 2).contiguous())
    native = scores.softmax(-1, dtype=torch.float32).to(values.dtype)
    assert torch.equal(probability[[1]], native[[1]])
    result = dict(passed=True, source_sha256={str(REPO/name): hashlib.sha256((REPO/name).read_bytes()).hexdigest()
        for name in ('features/qwen3_mass_matched_visual.py', 'scripts/verify_qwen3_mass_matched_visual.py')},
        maximum_log_visual_partition_residual=max_residual, maximum_visual_mass_difference=max_mass_difference,
        maximum_nonvisual_probability_difference=max_nonvisual_difference,
        bf16_visual_mass_difference_after_cast=bf16_mass_difference,
        matching_precision='FP32 visual partition; BF16 probabilities have separately measured rounding error.',
        zero_and_constant_direction_exact_uniform=True, unselected_rows_and_inputs_exact=True,
        reversed_rows_exact=True, two_patch_direction_reverses=True,
        scope='CPU tensor operator only; not decoder/KV or mitigation-effect validation.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    verify(parser.parse_args().output)
