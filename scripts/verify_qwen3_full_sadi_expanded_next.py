"""CPU proof that semantic compression preserves original selected targets."""

import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import write,sha


def run(output):
    import torch
    from features.llava_sar_ems_attention import SarPatchTerms
    from features.qwen3_full_sadi_expanded_next import CompactSarPatchTerms
    assert not output.exists();torch.manual_seed(43);tokens=frozenset([0,4,19,40,96]);comparisons=0
    for dtype in [torch.float32,torch.bfloat16]:
        values=torch.randn(4,9,3).to(dtype);weight=torch.randn(12,12).to(dtype);hidden=torch.randn(9,12).to(dtype)
        head=torch.nn.Linear(12,97,bias=False).to(dtype);norm=torch.nn.RMSNorm(12).to(dtype)
        full=SarPatchTerms(values,weight,hidden,norm,head);compact=CompactSarPatchTerms.from_full(full,tokens)
        assert compact.log_vocab==full.log_vocab and torch.equal(compact.log_normalizer,full.log_normalizer)
        for _ in range(20):
            attention=torch.rand(4,9).to(dtype)
            for token in tokens:
                expected=full.distribution(attention,token);actual=compact.distribution(attention,token)
                assert torch.equal(expected,actual);comparisons+=1
        selected=torch.tensor(sorted(tokens))
        assert torch.equal(full.log_semantics(selected),compact.log_semantics(selected))
        assert compact.visual_logits.numel()<full.visual_logits.numel()
        try:compact.distribution(attention,2)
        except KeyError:pass
        else:raise AssertionError('Uncached target must not silently use another column.')
    assert not torch.cuda.is_initialized()
    write(output,dict(passed=True,exact_distribution_comparisons=comparisons,cuda_initialized=False,
        scope='CPU selected-logit semantics/normalizer/Gram only; GPU original2 and expanded controls remain required.',
        source_sha256={str(REPO/'features/qwen3_full_sadi_expanded_next.py'):sha(REPO/'features/qwen3_full_sadi_expanded_next.py'),
            str(REPO/'features/llava_sar_ems_attention.py'):sha(REPO/'features/llava_sar_ems_attention.py'),str(Path(__file__)):sha(Path(__file__))}))
    print('CPU_COMPACT_SAR_EXACT',comparisons)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);run(parser.parse_args().output)
