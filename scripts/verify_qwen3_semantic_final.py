"""Independent small-tensor checks for semantic SAR and static cache reuse."""

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import torch
from features.qwen3_semantic_final_feedback import StaticSemanticWrites, blocked_write_gram
from features.llava_sar_ems_attention import visual_write_norms, rebalance_visual_logits


def verify():
    torch.manual_seed(917)
    h, m, d, dim, vocab = 3, 11, 4, 12, 41
    values, weight, hpre = torch.randn(h, m, d), torch.randn(dim, h*d), torch.randn(m, dim)
    head = torch.nn.Linear(dim, vocab, bias=False)
    norm = torch.nn.RMSNorm(dim)
    attention = torch.rand(h, m)
    direct = torch.stack([weight @ (values[:, j]*attention[:, j, None]).flatten() for j in range(m)]).norm(dim=-1)
    gram = blocked_write_gram(values, weight, chunk=3)
    computed = visual_write_norms(gram, attention)
    torch.testing.assert_close(computed, direct, rtol=1e-6, atol=2e-6)
    tokens = [2, 7, 13]
    cache = StaticSemanticWrites(hpre, values, weight, norm, head, tokens, chunk=3)
    cache.check(hpre.clone(), values.clone(), weight, norm, head, tokens)
    logits = head(norm(hpre)).float()
    log_s = torch.log1p(vocab*logits.softmax(-1)[:, tokens])
    torch.testing.assert_close(cache.log_s, log_s, rtol=1e-6, atol=1e-6)
    for mode in ('full_sar', 'semantic_increment'):
        expected = direct*(log_s[:, 1]+(mode == 'full_sar'))
        expected /= expected.sum()
        torch.testing.assert_close(cache.distribution(attention, 7, mode), expected, rtol=1e-6, atol=1e-7)
    try:
        cache.check(hpre+1, values, weight, norm, head, tokens)
    except AssertionError:
        pass
    else:
        raise AssertionError('Cross-image cache reuse was not rejected')
    scores, budget = torch.randn(2, h, m), torch.rand(2, m)
    for permute in (False, True):
        edited = rebalance_visual_logits(scores, budget, permute=permute)
        torch.testing.assert_close(scores.logsumexp(-1), edited.logsumexp(-1), rtol=1e-6, atol=1e-6)
    return dict(passed=True, direct_write_norm_max_error=float((direct-computed).abs().max()),
        semantic_max_error=float((log_s-cache.log_s).abs().max()),
        invalid_image_cache_rejected=True, visual_mass_and_permuted_mass_preserved=True)


if __name__ == '__main__':
    result = verify()
    out = REPO / 'outputs/qwen3_semantic_final_cpu'
    out.mkdir(exist_ok=True)
    (out / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
