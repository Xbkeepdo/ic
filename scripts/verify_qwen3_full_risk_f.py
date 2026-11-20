"""Replay the original72-dimensional checkpoints on the frozen old test split."""

import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from features.qwen3_online_i_el import FullIElProbe
from scripts.run_qwen3_signed_feedback import sha,write

TRAIN=REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'


def verify():
    torch.set_num_threads(1)
    compact=TRAIN/'compact_features.npz'
    assert sha(compact)=='0175471b249a6256772511d8de72fb58acd243723891d586d97c6e95e0625184'
    output={}
    with np.load(compact,allow_pickle=False) as data,torch.inference_mode():
        mask=data['test_mask'];labels=data['label'][mask]
        assert len(labels)==2367
        for name,columns,expected in [('risk+F_log',('risk','F_log'),.9019113513333166),
                                      ('I+E_l',('I','E_l'),.9017643403481553)]:
            path=TRAIN/name/'seed43'
            model=FullIElProbe(path).eval()
            features=np.concatenate([data[key][mask] for key in columns],axis=1)
            assert features.shape==(2367,72)
            probabilities=model(torch.from_numpy(features)).numpy()
            auc=float(roc_auc_score(labels,probabilities))
            assert abs(auc-expected)<1e-12,(name,auc,expected)
            output[name]=dict(real_positive_auroc=auc,hall_positive_aupr=float(average_precision_score(1-labels,1-probabilities)),
                input_order=[f'{key}0..35' for key in columns],output_is_real_probability=True,
                model_sha256=sha(path/'model.pt'),scaler_sha256=sha(path/'input_scaler.npz'))
    return dict(passed=True,old_test_targets=2367,compact_sha256=sha(compact),checkpoints=output)


if __name__=='__main__':
    result=verify()
    out=REPO/'outputs/qwen3_full_risk_f_cpu';out.mkdir(exist_ok=True)
    write(out/'verification.json',result)
    print(json.dumps(result,indent=2))
