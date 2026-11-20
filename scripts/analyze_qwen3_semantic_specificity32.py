"""CPU-only semantic evidence audit using the frozen source19 cache."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

REPO=Path(__file__).resolve().parents[1]
SNAP=REPO/'scripts/experiment_snapshots'
EPS=1e-30


def read(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def inputs(kind):
    if kind=='snapshots':
        source=SNAP/'qwen3_risk_plan_breadth32';mapping=SNAP/'qwen3_visual_semantics_cache/image353411.npz'
        labels=SNAP/'qwen3_full_risk_f_trace/labelled_positions.json'
        lexicon=SNAP/'qwen3_live_visual_ffn_contrast10/object_lexicon.json'
        ground,=(SNAP/'qwen3_live_preference_contrast100/evaluations/sentence_v1').glob('chair_ground_truth.*.json')
    else:
        source=REPO/'outputs/qwen3_risk_plan_breadth32_20261120';mapping=REPO/'outputs/qwen3_visual_semantics_cache_20261120/image353411.npz'
        labels=REPO/'outputs/qwen3_full_risk_f_trace_20261118/labelled_positions.json'
        lexicon=REPO/'outputs/qwen3_clva_signal_preflight10_v2_20261120/object_lexicon.json'
        ground,=(REPO/'outputs/qwen3_live_preference_contrast100_20261120/evaluations/sentence_v1').glob('chair_ground_truth.*.json')
    return dict(source=source,mapping=mapping,labels=labels,lexicon=lexicon,ground=ground)


def prepare(root,kind):
    assert not root.exists();p=inputs(kind)
    assert read(p['source']/'pipeline_complete.json')['passed']
    files=[Path(__file__),p['mapping'],p['labels'],p['lexicon'],p['ground'],p['source']/'protocol.json',p['source']/'pipeline_complete.json']
    files += [p['source']/f'{name}{shard}{ext}'for shard in [0,1]
              for name,ext in [('semantic19.shard','.npz'),('worker','.json'),('results','.json')]]
    root.mkdir();(root/'source').mkdir()
    (root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),sources={k:str(v)for k,v in p.items()},source_kind=kind,
        source_sha256={str(x.resolve()):sha(x)for x in files},
        images=27,cases=69,classes=62,tokens=309,epsilon=EPS,
        image_methods=['alias_sum','alias_mean','alias_max','loo_ratio','loo_patch_logratio'],
        case_methods=['attention_raw','attention_lift','attention_background','ffn_background','uniform_background'],
        background='Equal-image mean of V*p=expm1(log_s); leave the current image out, no labels. LOO patch evidence=log1p(class mass/background class mean). Save full27 prior separately for future development.',
        image_metrics='All27: annotated top1 hit, top3 precision/recall, macro within-image class AUC; report GT class coverage. No GT in calibration.',
        case_metrics='All69: native class rank/margin against62 classes. Real-positive margin AUC on63 existing automatically labelled positions only; six unaligned retained separately. Not a new mitigation evaluation.',
        attention_recovery='Within original S only, weights proportional to b/log_s(y), where b is original target marginal. No claim of recovering full-image attention or write norms.',
        limitations='62 classes from309 complete single-token forms; excludes multi-token aliases and18 COCO classes. Annotation incompleteness and noun/color ambiguity retained.',
        controls='Report all methods; original raw readout and uniform-background distinguish alias bias and attention specificity. No threshold fitting or MLP.',
        reference='https://proceedings.mlr.press/v139/zhao21c.html',
        scope='CPU cached-signal diagnosis, no model calls/new captions/CHAIR/F1/independent500. Changes to normalized Q cannot be inferred from a classwise constant rescaling alone.'))


def check(plan):
    for name,wanted in plan['source_sha256'].items():assert sha(Path(name))==wanted,name


def background_log_evidence(mass,prior):return np.log1p(mass/np.maximum(prior,EPS))


def auc(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float);positive=int(labels.sum());negative=len(labels)-positive
    return None if not positive or not negative else float((rankdata(scores)[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def execute(root):
    plan=read(root/'protocol.json');check(plan);p={k:Path(v)for k,v in plan['sources'].items()};started=time.time()
    with np.load(p['mapping'],allow_pickle=False)as saved:
        tokens=saved['token_ids'].copy();names=saved['class_names'].tolist();mapping=saved['token_classes'].copy()
        old_mass=np.expm1(saved['log_s'][19].astype(np.float64))
        old_expected=saved['class_E_l_roundtrip'][19].copy()
    assert len(tokens)==309 and len(names)==62 and len(set(tokens.tolist()))==309
    columns=[np.flatnonzero(mapping==name)for name in names];counts=np.array([len(c)for c in columns])
    assert np.all(counts>0)and sum(counts)==309
    lexicon=read(p['lexicon']);assert lexicon['token_ids']==tokens.tolist()
    old_aggregate=np.array([old_mass[:,c].sum()for c in columns])
    assert np.allclose(old_aggregate/old_aggregate.sum(),old_expected/old_expected.sum(),atol=1e-6,rtol=1e-6)
    token_class={int(t):str(c)for t,c in zip(tokens,mapping)};token_column={int(t):i for i,t in enumerate(tokens)}
    cache={};raw={};model_plan=read(p['source']/'protocol.json')
    for shard in [0,1]:
        worker=read(p['source']/f'worker{shard}.json');filename=p['source']/f'semantic19.shard{shard}.npz'
        assert worker['passed']and sha(filename)==worker['semantic_cache_sha256']
        with np.load(filename,allow_pickle=False)as data:
            assert np.array_equal(data['token_ids'],tokens)
            assert data['batch_ids'].tolist()==model_plan['groups'][shard]['ids']
            assert str(data['protocol_sha256'])==sha(p['source']/'protocol.json')
            for row,image in enumerate(data['image_ids']):
                log_s=data['log_s'][row].astype(np.float64);scaled=np.expm1(log_s)
                assert np.isfinite(scaled).all()and np.min(scaled)>=0 and int(image)not in cache
                class_mass=np.stack([scaled[:,c].sum(1)for c in columns],axis=1)
                cache[int(image)]=dict(log_s=log_s,scaled=scaled,class_mass=class_mass)
        for case in read(p['source']/f'results{shard}.json'):
            key=case['image_id'],case['step'];assert key not in raw;raw[key]=case
    assert len(cache)==27 and len(raw)==69
    ids=sorted(cache);means=np.stack([cache[i]['class_mass'].mean(0)for i in ids])
    token_means=np.stack([cache[i]['scaled'].mean(0)for i in ids]);priors={}
    image_scores={name:[]for name in plan['image_methods']}
    for index,image in enumerate(ids):
        prior=np.delete(means,index,axis=0).mean(0);priors[image]=prior
        item=cache[image];mass=item['class_mass'];background=background_log_evidence(mass,prior)
        assert np.isfinite(background).all()and np.min(background)>=0
        factors=np.linspace(.7,3.1,len(names))
        assert np.allclose(background_log_evidence(mass*factors,prior*factors),background,atol=1e-12,rtol=1e-12)
        values=dict(alias_sum=means[index],alias_mean=means[index]/counts,
            alias_max=np.array([item['scaled'][:,c].mean(0).max()for c in columns]),
            loo_ratio=means[index]/np.maximum(prior,EPS),loo_patch_logratio=background.mean(0))
        for name,score in values.items():image_scores[name].append(score)
    image_scores={k:np.stack(v)for k,v in image_scores.items()}
    # Scores and background priors are complete before image annotations enter.
    ground=read(p['ground'])['objects'];truth=np.array([[c in ground[str(i)]for c in names]for i in ids],bool)
    image_metrics={};image_rows=[]
    for method,scores in image_scores.items():
        orders=np.argsort(-scores,axis=1,kind='stable');top1=truth[np.arange(len(ids)),orders[:,0]]
        top3=np.take_along_axis(truth,orders[:,:3],axis=1).sum(1);available=truth.sum(1)
        image_metrics[method]=dict(top1_hit=int(top1.sum()),images=len(ids),top3_precision=float(top3.sum()/(3*len(ids))),
            top3_recall=float(np.mean(top3/np.maximum(available,1))),macro_image_auc=float(np.mean([auc(t,s)for t,s in zip(truth,scores)if t.any()and not t.all()])))
    for index,image in enumerate(ids):
        image_rows.append(dict(image_id=image,annotated_classes=ground[str(image)],covered_classes=[c for c in ground[str(image)]if c in names],
            predictions={method:[dict(name=names[j],score=float(scores[index,j]),annotated=bool(truth[index,j]))
                for j in np.argsort(-scores[index],kind='stable')[:5]]for method,scores in image_scores.items()}))
    labels={(r['image_id'],r['step'],r['token_id']):r for r in read(p['labels'])};case_rows=[]
    for key,case in sorted(raw.items()):
        image,step=key;target=case['token_id'];item=cache[image];support=np.array(case['support'])
        target_log_s=item['log_s'][support,token_column[target]];b=np.array(case['problem']['target'],float)
        recoverable=target_log_s>0;weights=np.divide(b,target_log_s,out=np.zeros_like(b),where=recoverable)
        assert weights.sum()>0;weights/=weights.sum()
        reconstructed=weights*target_log_s;reconstructed/=reconstructed.sum()
        assert np.allclose(reconstructed,b/b.sum(),atol=1e-7,rtol=1e-6)
        a=np.array(case['problem']['source'],float);a/=a.sum();mass=item['class_mass'];bg=background_log_evidence(mass,priors[image])
        weighted=weights@mass[support]
        scores=dict(attention_raw=weighted,attention_lift=weighted/np.maximum(mass.mean(0),EPS),
            attention_background=weights@bg[support],ffn_background=a@bg[support],uniform_background=bg.mean(0))
        canonical=token_class[target];column=names.index(canonical);matched=labels.get((image,step,target))
        row=dict(image_id=image,step=step,token_id=target,native_text=case['native_text'],canonical=canonical,
            automatic_label='unaligned'if matched is None else('hallu'if matched['automatic_hallucination']else'real'),
            context=None if matched is None else matched['context'],support=len(support),zero_log_s_positions=int((~recoverable).sum()),methods={})
        for method,score in scores.items():
            others=np.arange(len(names))!=column;logscore=np.log(np.maximum(score,EPS));order=np.argsort(-score,kind='stable')
            row['methods'][method]=dict(target_rank=float(rankdata(-score)[column]),
                margin=float(logscore[column]-logscore[others].max()),
                top_class=names[int(order[0])],top_class_annotated=bool(names[int(order[0])]in ground[str(image)]),
                target_score=float(score[column]))
        case_rows.append(row)
    assert len(case_rows)==69
    labelled=[r for r in case_rows if r['automatic_label']!='unaligned'];assert len(labelled)==63
    case_metrics={method:dict(real_positive_margin_auc=auc([r['automatic_label']=='real'for r in labelled],
        [r['methods'][method]['margin']for r in labelled]),
        target_top1={label:sum(r['methods'][method]['target_rank']==1 for r in case_rows if r['automatic_label']==label)
            for label in ['real','hallu','unaligned']})for method in plan['case_methods']}
    np.savez_compressed(root/'background_prior27.npz',class_names=np.array(names),token_ids=tokens,token_classes=mapping,
        class_prior=means.mean(0),token_prior=token_means.mean(0),image_ids=np.array(ids),epsilon=EPS,
        source_protocol_sha256=sha(p['source']/'protocol.json'))
    write(root/'image_rows.json',image_rows);write(root/'case_rows.json',case_rows)
    check(plan)
    write(root/'analysis.json',dict(passed=True,seconds=time.time()-started,image_metrics=image_metrics,case_metrics=case_metrics,
        images=27,cases=69,labelled_cases=63,classes=62,classes_present_in_annotations=int(truth.any(0).sum()),
        images_with_covered_gt=int(truth.any(1).sum()),alias_counts=dict(zip(names,counts.tolist())),
        mathematical_checks=dict(old_class_roundtrip=True,classwise_scaling_invariance=True,
            within_support_attention_reconstruction=True,no_image_labels_used_in_scores=True),
        neural_model_calls=0,scope='Cached-signal diagnosis; no new mitigation/CHAIR/F1/independent500.'))
    print(json.dumps(read(root/'analysis.json'),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--sources',choices=['snapshots','outputs'],default='snapshots')
    args=parser.parse_args();prepare(args.root,args.sources)if args.prepare else execute(args.root)
