"""Compare evidence-guided and FFN-preference-guided post-SADI attention."""

import math
import torch


def dual_attention_logits(native, positive, negative, alpha=1., beta=.1):
    assert native.shape == positive.shape == negative.shape
    assert native.ndim == 1 and alpha >= 0 and 0 < beta <= 1
    assert all(torch.isfinite(x).all() for x in [native, positive, negative])
    scores = (1+alpha)*positive-alpha*negative
    allowed = native >= native.max()+math.log(beta)
    return scores.masked_fill(~allowed, -torch.inf)


PAIRS = dict(sar_ffn=('sar', 'ffn'), sar_uv=('sar', 'uv'),
             uniform_ffn=('uniform', 'ffn'), sar_uniform=('sar', 'uniform'),
             sar_rolled_ffn=('sar', 'rolled_ffn'), sar_only=('sar', 'sar'))
