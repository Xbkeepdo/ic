"""Index existing source semantics for a possible SADI probe adaptation.

No model is loaded. Saved selected-target probabilities are converted to
log_s in one ragged memory map. Other feature and caption inputs remain in
their existing files; no full vocabulary projection is repeated.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

REPO=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda:handle.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path,value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def records(node):
    if isinstance(node,dict):
        if {'image_id','response_token_idx','target_token_id','matrices'}<=set(node):
            yield node
        else:
            for value in node.values():yield from records(value)
    elif isinstance(node,(list,tuple)):
        for value in node:yield from records(value)


def matrix(record,name):
    value=record['matrices']
    if isinstance(value,dict):
        value=value[name]
        if isinstance(value,dict):value=[value[str(i)] if str(i) in value else value[i] for i in range(36)]
    else:
        value=[layer[name] for layer in value]
    return np.asarray(value,dtype=np.float32)


def prepare(root):
    assert not root.exists(),'Inspect existing cache rather than rebuild it.'
    train=REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'
    source=Path(read(train/'protocol.json')['source'])
    generation=source.parent/'generations.json'
    current=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119/cohort.json'
    validation=REPO/'outputs/qwen3_mass_matched_visual_reserved500_20261119/cohort.json'
    paths=[Path(__file__),source,generation,source.parent/'audit_summary.json',train/'protocol.json',train/'compact_features.npz',current,validation]
    started=time.time();inputs={str(p):sha(p) for p in paths}
    with np.load(train/'compact_features.npz',allow_pickle=False) as handle:data={k:handle[k].copy() for k in handle.files}
    captions=read(generation)
    ids=data['image_id'];positions=data['response_token_idx'];tokens=data['target_token_id'];count=len(ids)
    assert count==11751 and ids.shape==positions.shape==tokens.shape
    keys=[(int(i),int(p),int(t)) for i,p,t in zip(ids,positions,tokens)]
    assert len(set(keys))==count
    for image,position,token in keys:
        response=captions[str(image)]['response_token_ids']
        assert 0<=position<len(response) and response[position]==token,(image,position,token)
    assert not set(map(int,ids))&set(read(current)['coco100'])
    assert not set(map(int,ids))&set(read(validation)['coco500'])
    prototype=[]
    for mask in [data['train_mask'],data['test_mask']]:
        ordered=list(dict.fromkeys(map(int,ids[mask])))
        prototype.extend(ordered[:4])
    root.mkdir()
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=inputs,
        source_schema=read(source.parent/'audit_summary.json')['schema'],
        purpose='Reuse existing per-target spatial semantics for a future SADI-domain source feature trace.',
        no_generation=True,no_model_loading=True,no_training=True,
        data='Same source3948 images/11751 targets, original labels and3200/800 image split unchanged.',
        prototype_images=prototype,prototype_selection='First4 source-train and first4 source-test image IDs in original compact order; not current test labels.',
        disjoint_current100_and_reserved500=True,
        legacy_target_log_s='Normalized OT target; rebuild raw L from saved probabilities and independently verify that target.',
        cache='One float32 ragged log_s memory map and row offsets; no duplicated raw attention, FFN distributions, captions or full vocabulary logits.'))
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available>source.stat().st_size*2+4*1024**3,('insufficient RAM',available)
    print('LOADING_EXISTING_SOURCE_PICKLE',source.stat().st_size,'available',available,flush=True)
    with source.open('rb') as handle:blob=pickle.load(handle)
    table={}
    for record in records(blob):
        key=int(record['image_id']),int(record['response_token_idx']),int(record['target_token_id'])
        assert key not in table,key
        table[key]=record
    assert set(table)==set(keys),(len(table),len(keys))
    visual=np.asarray([int(table[key]['num_visual_tokens']) for key in keys],np.int32)
    assert int(visual.min())>=1 and int(visual.max())<=400
    vocab={int(record['vocabulary_size']) for record in table.values()};assert len(vocab)==1;vocabulary=vocab.pop()
    lengths=visual.astype(np.int64)*36;offsets=np.concatenate([np.zeros(1,np.int64),np.cumsum(lengths)])
    cache=np.lib.format.open_memmap(root/'log_s.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),))
    max_target_distribution_error=max_semantic_sum_error=0.
    for row,key in enumerate(keys):
        record=table[key]
        assert record['schema']=='ic-log-s-hpre-reextract-v2'
        assert record['tau']==.03 and record['transport_top_k']==32
        assert record['semantic_cost_weight']==record['endpoint_penalty_weight']==1.
        for name,field in [('risk','risk'),('I','I'),('E_l','E_l')]:
            np.testing.assert_array_equal(np.asarray(record[field],np.float32),data[name][row])
        probability=matrix(record,'object_probability_norm_hpre')
        # The legacy name target_log_s denotes the normalized transport
        # target, not L_j itself. Rebuild L_j from the saved vocabulary
        # probabilities, then verify that it reproduces that transport target.
        logs=np.log1p(vocabulary*probability).astype(np.float32)
        target=matrix(record,'target_log_s');attention=matrix(record,'attention')
        assert logs.shape==probability.shape==target.shape==attention.shape==(36,int(visual[row]))
        assert np.isfinite(logs).all() and (logs>=0).all()
        weighted=attention*logs;total=weighted.sum(-1,keepdims=True)
        expected=np.divide(weighted,total,out=np.full_like(weighted,1/weighted.shape[-1]),where=total>0)
        error=float(np.max(np.abs(target-expected)));assert error<1e-5
        semantic_error=float(np.max(np.abs(probability.sum(-1)-data['E_l'][row])))
        np.testing.assert_allclose(probability.sum(-1),data['E_l'][row],rtol=1e-5,atol=1e-7)
        max_target_distribution_error=max(max_target_distribution_error,error);max_semantic_sum_error=max(max_semantic_sum_error,semantic_error)
        cache[offsets[row]:offsets[row+1]]=logs.ravel()
    cache.flush();del cache
    np.savez_compressed(root/'index.npz',image_id=ids,response_token_idx=positions,target_token_id=tokens,
        label=data['label'],train_mask=data['train_mask'],test_mask=data['test_mask'],visual_tokens=visual,offsets=offsets)
    for path,value in inputs.items():assert sha(Path(path))==value,path
    result=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),images=len(set(map(int,ids))),targets=count,
        prototype_images=prototype,vocabulary_size=vocabulary,visual_token_range=[int(visual.min()),int(visual.max())],
        maximum_reconstructed_target_distribution_error=max_target_distribution_error,maximum_semantic_sum_error=max_semantic_sum_error,
        original_feature_and_target_identity_verified=True,
        output_sha256={name:sha(root/name) for name in ['log_s.npy','index.npz']},
        cache_bytes=(root/'log_s.npy').stat().st_size,no_model_calls=True,seconds=time.time()-started)
    write(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    prepare(parser.parse_args().root)
