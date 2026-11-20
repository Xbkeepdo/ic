"""Check tiny-gradient distributions, zero heads and scale invariance on CPU."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.qwen3_probe_attention_recycle import normalize_nonnegative


def verify():
    torch.manual_seed(43)
    original = torch.rand(2, 4, 17)
    original[:, 0] = 0
    preference = torch.rand(2, 17).softmax(-1)
    reference = original.double() / original.double().sum(-1, keepdim=True).clamp_min(1e-100)
    reports = []
    for scale in (1., 1e-8, 1e-14, 1e-24):
        negative = original * scale
        heads = normalize_nonnegative(negative)
        torch.testing.assert_close(heads, reference.float(), rtol=1e-6, atol=1e-7)
        evidence = normalize_nonnegative(negative.sum(1))
        deficit = (evidence - preference).clamp_min(0)
        spatial = .25 + .75 * deficit / deficit.amax(-1, keepdim=True).clamp_min(1e-12)
        destination = normalize_nonnegative(heads * spatial[:, None])
        torch.testing.assert_close(destination[:, 1:].sum(-1), torch.ones(2, 3), rtol=0, atol=2e-7)
        assert (destination[:, 0] == 0).all()
        removed = torch.rand(2, 4, 13) * .01
        removed[:, 0] = 0
        added = removed.sum(-1, keepdim=True) * destination
        discrepancy = (added.sum(-1) - removed.sum(-1)).abs().max().item()
        assert discrepancy < 1e-7
        reports.append(dict(gradient_scale=scale, max_mass_discrepancy=discrepancy))
    return dict(passed=True, cases=reports)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
