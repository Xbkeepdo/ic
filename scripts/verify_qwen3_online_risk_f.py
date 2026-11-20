"""Check exact OT+endpoint cost against an independent LP and portable gate."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.qwen3_online_risk_f import CompactRiskFGate, exact_risk, native_write_norms


def reference_lp(problem, penalties):
    a, b, cost = problem
    a, b = a.astype(np.float64), b.astype(np.float64)
    a /= a.sum()
    b /= b.sum()
    count = len(a)
    equations = lil_matrix((2*count, count*count))
    for i in range(count):
        equations[i, i*count:(i+1)*count] = 1
        equations[count+i, i::count] = 1
    # Both endpoint coefficients are exactly one.
    joint = cost.astype(np.float64)+penalties[:, None]+penalties[None]
    result = linprog(joint.flatten(), A_eq=equations.tocsr(), b_eq=np.r_[a, b],
                     bounds=(0, None), method='highs')
    assert result.success, result.message
    return result.fun


def verify(gate_path):
    torch.manual_seed(1729)
    hpre, prediction, update = torch.randn(37, 12), torch.randn(12), torch.randn(12)
    attention, log_s = torch.rand(3, 37), torch.rand(37)
    actual, parts, problem, penalties = exact_risk(hpre, prediction, update, attention, log_s)
    expected = reference_lp(problem, penalties)
    assert abs(actual-expected) < 2e-6, (actual, expected)
    assert parts['support'] <= 32 and parts['penalty'] > 0
    values, weights = torch.randn(3, 37, 4), torch.randn(12, 12)
    reference = torch.stack([sum(weights[:, h*4:(h+1)*4] @ values[h, j]*attention[h, j]
                                 for h in range(3)) for j in range(37)]).norm(dim=-1)
    computed = native_write_norms(values, attention, weights)
    torch.testing.assert_close(computed, reference, rtol=2e-6, atol=2e-6)
    gate = CompactRiskFGate(gate_path)
    p = gate.parameters
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = p['mean'], p['scale'], p['scale']**2
    scaler.n_features_in_, scaler.n_samples_seen_ = 4, 9384
    mlp = MLPClassifier(hidden_layer_sizes=(128,))
    mlp.coefs_, mlp.intercepts_ = [p['weight0'], p['weight1']], [p['bias0'], p['bias1']]
    mlp.n_layers_, mlp.n_outputs_, mlp.out_activation_, mlp.n_features_in_ = 3, 1, 'logistic', 4
    mlp.classes_ = np.array([0, 1])
    pipeline = Pipeline([('scale', scaler), ('mlp', mlp)])
    x = (np.random.default_rng(43).normal(size=(257, 4))*p['scale']+p['mean']).astype(np.float32)
    a, b = gate.predict(x), pipeline.predict_proba(x)[:, 1]
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)
    return dict(passed=True, exact_transport_vs_joint_endpoint_lp_error=abs(actual-expected),
        native_writes_vs_head_sum_max_error=float((reference-computed).abs().max()),
        portable_vs_sklearn_max_error=float(np.max(np.abs(a-b))), threshold=gate.threshold,
        columns=list(zip(p['feature_names'].tolist(), p['layers'].tolist())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    with torch.inference_mode():
        result = verify(args.gate)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
