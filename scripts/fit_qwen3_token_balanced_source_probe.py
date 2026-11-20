"""Keep the source features/probe recipe; balance labels within target token.

Train-only tokens with both labels receive equal total real/hall probability.
The sampler first draws an eligible token uniformly in expectation, then its
label uniformly, then an example uniformly. The model never receives token ID.
"""

import argparse
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import fit_qwen3_native_sadi_source_probe as original
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def prepare(root):
    import json
    import numpy as np
    original.prepare(root)
    plan=read(root/'protocol.json')
    source=Path(plan['original_source'])
    with np.load(source/'compact_features.npz',allow_pickle=False) as data:
        mask=data['train_mask'];tokens=data['target_token_id'][mask];labels=data['label'][mask]
    assert len(tokens)==9384 and labels.sum()==7278
    weights=np.zeros(len(tokens),np.float64);groups=[]
    for token in np.unique(tokens):
        selected=tokens==token
        real=int((selected&(labels==1)).sum());hall=int((selected&(labels==0)).sum())
        if not real or not hall:continue
        weights[selected&(labels==1)]=1/real;weights[selected&(labels==0)]=1/hall
        assert abs(weights[selected&(labels==1)].sum()-1)<1e-12
        assert abs(weights[selected&(labels==0)].sum()-1)<1e-12
        groups.append(dict(token_id=int(token),real=real,hall=hall))
    assert len(groups)>1 and np.isfinite(weights).all() and (weights>0).any()
    assert abs(weights.sum()-2*len(groups))<1e-10
    np.savez_compressed(root/'sampling_weights.npz',weight=weights,train_token=tokens,train_label=labels)
    write(root/'sampling_audit.json',dict(passed=True,source_train_targets=len(tokens),
        eligible_tokens=len(groups),eligible_targets=int((weights>0).sum()),
        excluded_single_class_targets=int((weights==0).sum()),groups=groups,
        expected_equal_label_mass_per_token=True,source_train_only=True))
    plan['source_sha256'].update({str(Path(__file__)):sha(Path(__file__)),
        str(root/'sampling_weights.npz'):sha(root/'sampling_weights.npz'),
        str(root/'sampling_audit.json'):sha(root/'sampling_audit.json')})
    plan.update(training_objective='BCE with token-and-label-balanced replacement sampling.',
        sampling=dict(weight='For eligible token y and label c:1/n_train(y,c); otherwise0.',
            eligible='Both real and hallucination labels exist in source TRAIN for the exact target token ID.',
            normalization='Expected uniform token; expected0.5 real and0.5 hallucination conditional on each token.',
            num_draws_per_epoch=9384,replacement=True,
            drop_last='Same batch256/drop_lastTrue;36 updates and9216 used draws per complete epoch.',
            eligible_tokens=len(groups),eligible_targets=int((weights>0).sum()),
            scaler='Unchanged standardizer uses all9384 source-train targets; the sampler changes neural training draws only.',
            seed='The three fixed training seeds control initialization, dropout and sampling; image split unchanged.'),
        reason='Check target-word label priors after source adaptation raised overall development AUC without improving same-token ordering or covering verified fork.',
        no_token_identity_model_input=True,no_current_sampling_or_threshold_fit=True)
    # This new root is not yet exposed to a worker. Seal the augmented recipe
    # before any fitting; the original immutable fit recipe is untouched.
    (root/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    target=root/'source/scripts'/Path(__file__).name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(__file__).read_bytes())
    print(dict(prepared=True,root=str(root),eligible_tokens=len(groups),eligible_targets=int((weights>0).sum())),flush=True)


def fit(root,device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,WeightedRandomSampler
    from scripts import train_torch_probe_feature_sets as trainer
    plan=read(root/'protocol.json')
    assert plan['training_objective']=='BCE with token-and-label-balanced replacement sampling.'
    with np.load(root/'sampling_weights.npz',allow_pickle=False) as handle:
        weights=torch.from_numpy(handle['weight'].copy())
        labels=handle['train_label'].copy()
    assert len(weights)==plan['sampling']['num_draws_per_epoch']==9384
    calls=[]

    def loader(dataset,*args,**kwargs):
        if kwargs.get('shuffle',False):
            assert len(dataset)==len(weights)
            # MatrixDataset has already applied the original real-positive
            # label convention. Keep exact row alignment with saved weights.
            np.testing.assert_array_equal(dataset.y.numpy().reshape(-1),labels.astype(np.float32))
            kwargs=dict(kwargs,shuffle=False,sampler=WeightedRandomSampler(weights,len(weights),replacement=True))
            calls.append(dict(rows=len(dataset),sample_count=len(weights),replacement=True,
                positive_weighted_mass=float(weights[torch.from_numpy(labels==1)].sum()),
                negative_weighted_mass=float(weights[torch.from_numpy(labels==0)].sum())))
        return DataLoader(dataset,*args,**kwargs)

    with patch.object(trainer,'DataLoader',loader):
        original.fit(root,device)
    assert len(calls)==len(plan['seeds'])==3
    write(root/'sampling_verification.json',dict(passed=True,training_loader_calls=calls,
        no_token_identity_feature=True,normalizer_and_evaluation_loaders_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    try:
        prepare(args.root) if args.prepare else fit(args.root,args.device)
    except Exception:
        if args.root.exists() and not (args.root/'failure.json').exists():
            write(args.root/'failure.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
