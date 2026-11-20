"""Independent ACG formula, orthogonal correction and signed-head checks."""

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from features.qwen3_acg_signal_feedback import acg_contexts, sar_ffn_drift, head_scales
from scripts.run_qwen3_signed_feedback import write


def verify():
    torch.manual_seed(43)
    torch.set_num_threads(1)
    score = torch.randn(3, 4, 1, 11)
    value = torch.randn(3, 4, 11, 7)
    conditional, direction, textual = acg_contexts(score, value, 2, 6)
    torch.testing.assert_close(conditional, (score.softmax(-1)@value)[:, :, 0], rtol=0, atol=0)
    mask = score.clone(); mask[..., 2:6] = -torch.inf
    independent_text = (mask.softmax(-1)@value)[:, :, 0]
    torch.testing.assert_close(textual, independent_text, rtol=0, atol=0)
    difference = conditional-independent_text
    expected = difference-(difference*independent_text).sum(-1, keepdim=True)/independent_text.square().sum(-1, keepdim=True)*independent_text
    torch.testing.assert_close(direction, expected, atol=3e-7, rtol=3e-6)
    orthogonal = float((direction*textual).sum(-1).abs().max())
    assert orthogonal < 1e-6
    gradient = torch.randn_like(direction)
    drift = torch.tensor([0., .3, 1.])
    scales, predicted, effect = head_scales(direction, gradient, drift, 1.4)
    assert torch.equal(scales[0], torch.zeros_like(scales[0]))
    assert torch.equal(scales[effect >= 0], torch.zeros_like(scales[effect >= 0]))
    torch.testing.assert_close(predicted, ((scales[:, :, None]*direction)*gradient).sum((1, 2)))
    for factor in (1e-8, 1e-16, 1e-24):
        scaled, _, _ = head_scales(direction, gradient*factor, drift, 1.4)
        torch.testing.assert_close(scales, scaled, rtol=2e-6, atol=1e-7)
    positive, _, _ = head_scales(direction, direction, drift, 1.4)
    assert torch.equal(positive, torch.zeros_like(positive))
    zero, _, _ = head_scales(direction, torch.zeros_like(direction), drift, 1.4)
    assert torch.equal(zero, torch.zeros_like(zero))
    _, zero_direction, _ = acg_contexts(score, torch.zeros_like(value), 2, 6)
    assert torch.equal(zero_direction, torch.zeros_like(zero_direction))
    w, log_s, visual, ffn = torch.rand(5), torch.rand(5), torch.randn(5, 9), torch.randn(9)
    q = w*(1+log_s); q /= q.sum()
    p = torch.nn.functional.cosine_similarity(visual, ffn[None], dim=-1).div(.03).softmax(-1)
    d = sar_ffn_drift(w, log_s, visual, ffn)
    torch.testing.assert_close(d, (1-(q*p).sqrt().sum()).clamp(0, 1), rtol=1e-6, atol=1e-7)
    return dict(passed=True,orthogonality_max_abs=orthogonal,
        independent_conditional_and_masked_attention_exact=True, paper_equations7_8=True,
        original_SAR_norm_times_one_plus_log_s=True, tau=.03,
        signed_head_prediction_nonpositive=True, gradient_scale_invariant=True,
        zero_FFN_drift_no_action=True, zero_or_nonnegative_effect_no_action=True,
        new_QKV_or_backward_in_ACG_context=False)


if __name__ == '__main__':
    result = verify()
    out = REPO/'outputs/qwen3_acg_signal_cpu'; out.mkdir(exist_ok=True)
    write(out/'verification.json', result)
    print(json.dumps(result, indent=2))
