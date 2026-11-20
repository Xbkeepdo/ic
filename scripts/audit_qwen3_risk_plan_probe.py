"""Independently verify saved transport objectives and masked decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def audit(root):
    output=root/'independent_risk_plan_audit.json'
    assert not output.exists()
    records=[]
    for shard in [0,1]:
        assert json.loads((root/f'worker{shard}.json').read_bytes())['passed']
        for case in json.loads((root/f'results{shard}.json').read_bytes()):
            problem=case['problem']
            a=np.array(problem['source'],np.float64);b=np.array(problem['target'],np.float64)
            cost=np.array(problem['cost']);plan=np.array(problem['plan']);endpoint=np.array(problem['endpoint'])
            a/=a.sum();b/=b.sum();b[-1]=1-b[:-1].sum();size=len(a)
            assert np.max(abs(plan.sum(1)-a))<1e-7 and np.max(abs(plan.sum(0)-b))<1e-7
            constraints=np.zeros((2*size-1,size*size))
            for i in range(size):constraints[i,i*size:(i+1)*size]=1
            for j in range(size-1):constraints[size+j,j::size]=1
            optimum=linprog(cost.ravel(),A_eq=constraints,b_eq=np.r_[a,b[:-1]],bounds=(0,None),method='highs',
                            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            assert optimum.success,optimum.message
            objective=float((plan*cost).sum());lp_error=abs(objective-optimum.fun)
            assert lp_error<1e-7
            # The saved objective uses a Torch FP64 reduction. NumPy has a
            # different sum order; use the same 1e-12 objective tolerance for
            # both its direct check and its addition to the endpoint term.
            assert abs(objective-problem['transport'])<1e-12
            penalty=float(np.dot(np.array(problem['source'],np.float32),np.array(problem['endpoint'],np.float32))
                          +np.dot(np.array(problem['target'],np.float32),np.array(problem['endpoint'],np.float32)))
            assert penalty==problem['penalty']
            assert abs(objective+penalty-problem['risk'])<1e-12
            assert problem['amount']==.5*problem['risk']/(1+problem['risk'])
            endpoint_plan=float((plan*(endpoint[:,None]+endpoint[None,:])).sum())
            endpoint_independent=float((np.outer(a,b)*(endpoint[:,None]+endpoint[None,:])).sum())
            assert abs(endpoint_plan-endpoint_independent)<1e-10
            ids=np.array(case['eligible_ids']);choices={};unmasked={};deltas={};conditional={}
            for trial in case['trials']:
                logits=np.array(trial['eligible_logits'],np.float32)
                assert int(ids[logits.argmax()])==trial['new_token']
                probability=np.exp(logits.astype(np.float64)-logits.max());probability/=probability.sum()
                name=trial['policy'];choices[name]=trial['new_token_text'];unmasked[name]=trial['unmasked_token_text']
                deltas[name]=trial['target_logp']-case['native_logp']
                conditional[name]=float(probability[np.flatnonzero(ids==case['token_id'])[0]])
                assert trial['all36_cache_unchanged']and trial['action_layers']==list(range(20,36))
            record=dict(image_id=case['image_id'],step=case['step'],native=case['native_text'],choices=choices,unmasked=unmasked,
                target_logp_delta=deltas,conditional_target_probability=conditional,support=size,risk=problem['risk'],
                transport=objective,penalty=penalty,lp_objective_error=lp_error,
                off_diagonal_plan_mass=float(plan.sum()-np.trace(plan)),
                endpoint_constant_difference=abs(endpoint_plan-endpoint_independent))
            records.append(record)
    report=dict(passed=True,decisions=18,plans=6,independent_solver='SciPy HiGHS LP, not original POT network simplex',
        lp_absolute_tolerance=1e-7,fp64_reduction_tolerance=1e-12,
        endpoint_penalty_marginal_constancy_verified=True,masked_argmax_independently_verified=True,
        unmasked_full_vocabulary_not_independently_verified=True,records=records)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
