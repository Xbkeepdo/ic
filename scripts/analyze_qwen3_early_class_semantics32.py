"""Compare layers3/4 with cached19 and qualify fixed nearest-class rivals."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

REPO=Path(__file__).resolve().parents[1];SNAP=REPO/'scripts/experiment_snapshots';EPS=1e-30


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(root):
    assert not root.exists()
    early=SNAP/'qwen3_early_semantics32';old=SNAP/'qwen3_risk_plan_breadth32'
    assert read(early/'pipeline_complete.json')['passed']
    ground,=(SNAP/'qwen3_live_preference_contrast100/evaluations/sentence_v1').glob('chair_ground_truth.*.json')
    paths=[Path(__file__),early/'protocol.json',early/'pipeline_complete.json',ground,
        SNAP/'qwen3_relative_evidence_probe/head_class_neighbors.json',
        SNAP/'qwen3_semantic_specificity32/case_rows.json',SNAP/'qwen3_semantic_specificity32/background_prior27.npz']
    for shard in [0,1]:paths += [early/f'classes34.shard{shard}.npz',early/f'worker{shard}.json',old/f'semantic19.shard{shard}.npz',
        SNAP/f'qwen3_post_sadi_ffn_attenuation_probe/results{shard}.json',SNAP/f'qwen3_dual_attention_contrast_probe/results{shard}.json']
    paths += [SNAP/f'qwen3_visual_semantics_cache/image{i}.npz'for i in [243882,353411]]
    root.mkdir();(root/'source').mkdir();(root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256={str(p.resolve()):sha(p)for p in paths},ground=str(ground),
        layers=[3,4,19],primary_layer=3,control_layer=4,reference_layer=19,
        image_methods=['raw','loo_ratio','loo_patch_logratio'],
        rival='Exactly the single nearest output-head class prototype; no manual semantic groups or GT-dependent candidate choice. Semantic eligibility iff rival rho>max(1,target rho).',
        cases='All69 prior positions, same automatic label groups. Semantic eligibility counted before FFN gain, whose cached value is available only for old6.',
        old_six='Global rho from disjoint27 class priors. Intersection with current source19 G_y>0; do not use early-layer gain.',
        maps='For old6 only: positive difference log1p(rival_class_mass/prior_r)-log1p(target_class_mass/prior_y), multiplied by normalized I19 recovered from Q19/(1+log_s19_y). All3/4/19 maps retained.',
        decision='Qualification only; not generation or500. Report every layer/method and both real/false examples. Layers chosen with knowledge of earlier two-image analyses, not blind selection.'))


def auc(labels,scores):
    labels=np.array(labels,bool);n=int(labels.sum());negative=len(labels)-n
    return float((rankdata(scores)[labels].sum()-n*(n+1)/2)/(n*negative))if n and negative else None


def execute(root):
    request=read(root/'protocol.json');started=time.time()
    for name,wanted in request['source_sha256'].items():assert sha(Path(name))==wanted,name
    early=SNAP/'qwen3_early_semantics32';old=SNAP/'qwen3_risk_plan_breadth32'
    cache={layer:{}for layer in [3,4,19]}
    for shard in [0,1]:
        worker=read(early/f'worker{shard}.json');assert worker['passed']and sha(early/f'classes34.shard{shard}.npz')==worker['cache_sha256']
        with np.load(early/f'classes34.shard{shard}.npz',allow_pickle=False)as d:
            names=d['class_names'].tolist();tokens=d['token_ids'].copy();mapping=d['token_classes'].copy()
            assert d['layers'].tolist()==[3,4]
            for l,layer in enumerate(d['layers'].tolist()):
                for j,image in enumerate(d['image_ids']):cache[layer][int(image)]=d['class_mass'][l,j].copy()
        columns=[np.flatnonzero(mapping==name)for name in names]
        with np.load(old/f'semantic19.shard{shard}.npz',allow_pickle=False)as d:
            assert np.array_equal(tokens,d['token_ids'])
            for j,image in enumerate(d['image_ids']):
                scaled=np.expm1(d['log_s'][j].astype(np.float64))
                cache[19][int(image)]=np.stack([scaled[:,c].sum(1)for c in columns],axis=1)
    ids=sorted(cache[19]);assert len(ids)==27 and all(sorted(cache[l])==ids for l in cache)
    means={l:np.stack([cache[l][i].mean(0)for i in ids])for l in cache}
    priors={l:v.mean(0)for l,v in means.items()}
    with np.load(SNAP/'qwen3_semantic_specificity32/background_prior27.npz',allow_pickle=False)as d:
        assert d['class_names'].tolist()==names and np.allclose(d['class_prior'],priors[19],atol=1e-12,rtol=1e-12)
    graph=read(SNAP/'qwen3_relative_evidence_probe/head_class_neighbors.json')
    assert graph['classes']==names and graph['tokens']==tokens.tolist()
    neighbor={c:graph['neighbors'][c][0]['name']for c in names}
    scores={};loo={}
    for layer in cache:
        loo[layer]={i:np.delete(means[layer],j,axis=0).mean(0)for j,i in enumerate(ids)}
        scores[layer]=dict(raw=means[layer],loo_ratio=np.stack([means[layer][j]/np.maximum(loo[layer][i],EPS)for j,i in enumerate(ids)]),
            loo_patch_logratio=np.stack([np.log1p(cache[layer][i]/np.maximum(loo[layer][i],EPS)).mean(0)for i in ids]))
    # Labels enter only after all background priors and score matrices exist.
    ground=read(Path(request['ground']))['objects'];truth=np.array([[c in ground[str(i)]for c in names]for i in ids],bool)
    metrics={};image_rows=[]
    for layer in cache:
        metrics[str(layer)]={}
        for method,value in scores[layer].items():
            order=np.argsort(-value,axis=1,kind='stable');top1=truth[np.arange(27),order[:,0]]
            count=np.take_along_axis(truth,order[:,:3],axis=1).sum(1)
            metrics[str(layer)][method]=dict(top1_hit=int(top1.sum()),images=27,top3_precision=float(count.sum()/81),
                macro_image_auc=float(np.mean([auc(t,s)for t,s in zip(truth,value)if t.any()and not t.all()])))
        for j,image in enumerate(ids):
            image_rows.append(dict(layer=layer,image_id=image,annotated=ground[str(image)],predictions={m:[names[k]for k in np.argsort(-v[j],kind='stable')[:5]]for m,v in scores[layer].items()}))
    cases=read(SNAP/'qwen3_semantic_specificity32/case_rows.json');case_rows=[];summary={}
    for layer in cache:
        selected=[]
        for case in cases:
            image=case['image_id'];target=case['canonical'];rival=neighbor[target]
            rho=means[layer][ids.index(image)]/np.maximum(loo[layer][image],EPS)
            y=float(rho[names.index(target)]);r=float(rho[names.index(rival)])
            row=dict(layer=layer,image_id=image,step=case['step'],target=target,rival=rival,target_rho=y,rival_rho=r,
                semantic_eligible=r>max(1.,y),automatic_label=case['automatic_label'],context=case['context'])
            selected.append(row);case_rows.append(row)
        summary[str(layer)]={label:dict(cases=sum(c['automatic_label']==label for c in selected),
            semantic_eligible=sum(c['automatic_label']==label and c['semantic_eligible']for c in selected))for label in ['real','hallu','unaligned']}
    old_rows=[];maps={}
    for shard,image in [(0,243882),(1,353411)]:
        with np.load(SNAP/f'qwen3_visual_semantics_cache/image{image}.npz',allow_pickle=False)as d:
            assert np.array_equal(tokens,d['token_ids']);logs={l:d['log_s'][l].astype(np.float64)for l in cache}
        mass={l:np.stack([np.expm1(logs[l])[:,c].sum(1)for c in columns],axis=1)for l in cache}
        dual={(r['image_id'],r['step']):r for r in read(SNAP/f'qwen3_dual_attention_contrast_probe/results{shard}.json')}
        for case in read(SNAP/f'qwen3_post_sadi_ffn_attenuation_probe/results{shard}.json'):
            column=int(np.flatnonzero(tokens==case['token_id'])[0]);target=str(mapping[column]);rival=neighbor[target];gain=case['native_ffn_target_logit_gain']
            weight=np.array(dual[image,case['step']]['evidence'])/(1+logs[19][:,column]);weight/=weight.sum()
            for layer in cache:
                mean=mass[layer].mean(0);y=names.index(target);r=names.index(rival)
                rho_y=float(mean[y]/max(priors[layer][y],EPS));rho_r=float(mean[r]/max(priors[layer][r],EPS))
                difference=np.maximum(np.log1p(mass[layer][:,r]/max(priors[layer][r],EPS))-np.log1p(mass[layer][:,y]/max(priors[layer][y],EPS)),0)
                heat=weight*difference;key=f'layer{layer}_image{image}_step{case["step"]}';maps[key]=heat
                old_rows.append(dict(layer=layer,image_id=image,step=case['step'],target=target,rival=rival,
                    target_rho=rho_y,rival_rho=rho_r,gain19=gain,semantic_eligible=rho_r>max(1.,rho_y),
                    actual_eligible=gain>0 and rho_r>max(1.,rho_y),positive_map_mass=float(heat.sum()),map_key=key,
                    automatic_hallucination=case['automatic_hallucination']))
    np.savez_compressed(root/'background_priors.npz',layers=np.array([3,4,19]),class_names=np.array(names),
        class_prior=np.stack([priors[l]for l in [3,4,19]]),image_ids=np.array(ids),token_ids=tokens,token_classes=mapping)
    np.savez_compressed(root/'old6_rival_maps.npz',**maps)
    write(root/'image_rows.json',image_rows);write(root/'case_rows.json',case_rows);write(root/'old6_rivals.json',old_rows)
    result=dict(passed=True,seconds=time.time()-started,image_metrics=metrics,semantic_eligibility=summary,old_six=old_rows,
        count_images=27,count_cases_per_layer=69,layer19_prior_matches_existing=True,neural_model_calls=0,
        scope='Readout/eligibility qualification only, not generation, physical correction or500.')
    write(root/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items()if k!='old_six'},ensure_ascii=False))
    print('OLD_TWO_GAIN_POSITIVE',json.dumps([r for r in old_rows if r['gain19']>0],ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare(a.root)if a.prepare else execute(a.root)
