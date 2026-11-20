"""Check the natural image/background ratio=1 guard without model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

REPO=Path(__file__).resolve().parents[1]
SNAP=REPO/'scripts/experiment_snapshots'


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(root):
    assert not root.exists()
    specificity=SNAP/'qwen3_semantic_specificity32'
    assert read(specificity/'analysis.json')['passed']
    paths=[Path(__file__),specificity/'protocol.json',specificity/'analysis.json',specificity/'background_prior27.npz',specificity/'case_rows.json']
    paths += [SNAP/f'qwen3_risk_plan_breadth32/semantic19.shard{s}.npz'for s in [0,1]]
    paths += [SNAP/f'qwen3_visual_semantics_cache/image{i}.npz'for i in [243882,353411]]
    paths += [SNAP/f'qwen3_post_sadi_ffn_attenuation_probe/results{s}.json'for s in [0,1]]
    root.mkdir();(root/'source').mkdir();(root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256={str(p.resolve()):sha(p)for p in paths},
        threshold=1.,definition='rho=mean_visual sum_alias(V*p)/background_class_mean. Weak iff rho<1, without tuning. LOO current image within27; old two diagnostic images use the full disjoint27 prior.',
        case_metrics='All69; Real-positive log-ratio AUC on63 automatically labelled cases and weak counts by label. No fitted threshold.',
        old_six='All6 old positions; separately show G19>0 and rho<1 intersection. No new attention proposals or generation.',
        scope='Cached-signal qualification only. A below-background class can still be present; no factuality or mitigation claim.'))


def execute(root):
    protocol=read(root/'protocol.json')
    for name,wanted in protocol['source_sha256'].items():assert sha(Path(name))==wanted
    base=SNAP/'qwen3_semantic_specificity32'
    with np.load(base/'background_prior27.npz',allow_pickle=False)as d:
        tokens=d['token_ids'].copy();mapping=d['token_classes'].copy();names=d['class_names'].tolist();prior=d['class_prior'].copy()
    cols=[np.flatnonzero(mapping==c)for c in names];means={}
    for shard in [0,1]:
        with np.load(SNAP/f'qwen3_risk_plan_breadth32/semantic19.shard{shard}.npz',allow_pickle=False)as d:
            assert np.array_equal(tokens,d['token_ids'])
            for index,image in enumerate(d['image_ids']):
                mass=np.expm1(d['log_s'][index].astype(np.float64))
                means[int(image)]=np.array([mass[:,c].sum(1).mean()for c in cols])
    rows=[]
    for case in read(base/'case_rows.json'):
        image=case['image_id'];column=names.index(case['canonical'])
        background=np.mean([v for i,v in means.items()if i!=image],axis=0)
        ratio=float(means[image][column]/max(background[column],1e-30))
        rows.append(dict(image_id=image,step=case['step'],canonical=case['canonical'],automatic_label=case['automatic_label'],rho=ratio,weak=ratio<1))
    labelled=[r for r in rows if r['automatic_label']!='unaligned'];positive=np.array([r['automatic_label']=='real'for r in labelled])
    ranks=rankdata([np.log(max(r['rho'],1e-30))for r in labelled]);n=int(positive.sum());negative=len(positive)-n
    auc=float((ranks[positive].sum()-n*(n+1)/2)/(n*negative))
    old=[]
    for shard,image in [(0,243882),(1,353411)]:
        with np.load(SNAP/f'qwen3_visual_semantics_cache/image{image}.npz',allow_pickle=False)as d:
            assert np.array_equal(tokens,d['token_ids']);mass=np.expm1(d['log_s'][19].astype(np.float64))
            current=np.array([mass[:,c].sum(1).mean()for c in cols])
        for case in read(SNAP/f'qwen3_post_sadi_ffn_attenuation_probe/results{shard}.json'):
            canonical=str(mapping[np.flatnonzero(tokens==case['token_id'])[0]]);column=names.index(canonical)
            ratio=float(current[column]/max(prior[column],1e-30));gain=case['native_ffn_target_logit_gain']
            old.append(dict(image_id=image,step=case['step'],canonical=canonical,rho=ratio,weak=ratio<1,
                gain=gain,positive_gain=gain>0,intersection=ratio<1 and gain>0,automatic_hallucination=case['automatic_hallucination']))
    summary={label:dict(count=sum(r['automatic_label']==label for r in rows),weak=sum(r['automatic_label']==label and r['weak']for r in rows))for label in ['real','hallu','unaligned']}
    result=dict(passed=True,threshold=1.,margin_not_used=True,all69=summary,real_positive_logrho_auc=auc,
        old_six=old,rows=rows,neural_model_calls=0,scope=protocol['scope'])
    write(root/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='rows'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare(args.root)if args.prepare else execute(args.root)
