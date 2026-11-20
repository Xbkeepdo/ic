"""CPU truth-table for acceptance ablation; GPU math remains a separate check."""

import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from features.qwen3_region_commit import should_commit
from scripts.run_qwen3_signed_feedback import write,sha


def run(output):
    import torch
    cases=[]
    for mode in ['wordchange','probability','direct','shifted']:
        for changed in [False,True]:
            for after in [-3.,-2.,-1.]:
                for zero,reject in [(False,False),(True,False),(False,True)]:
                    result=should_commit(mode,-2.,after,10,11 if changed else 10,zero=zero,reject=reject)
                    expected=(not zero and not reject and
                        (mode in ['direct','shifted'] or (after < -2. and (changed or mode=='probability'))))
                    assert result==expected
                    cases.append(dict(mode=mode,changed=changed,after=after,zero=zero,reject=reject,result=result))
    assert not torch.cuda.is_initialized()
    assert not output.exists()
    write(output,dict(passed=True,cases=len(cases),cuda_initialized=False,
        scope='Commit predicates only. Does not establish GPU replay equivalence or mitigation efficacy.',
        source_sha256={str(REPO/'features/qwen3_region_commit.py'):sha(REPO/'features/qwen3_region_commit.py'),
            str(Path(__file__)):sha(Path(__file__))}))
    print('CPU_COMMIT_PREDICATES_PASSED',len(cases))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);run(parser.parse_args().output)
