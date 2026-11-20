"""Trace archived source targets under complete native SADI, with fresh semantics.

Captions, labels and image split are reused. No captions are newly decoded.
Every forced response prefix uses the real one-token KV-cache trajectory;
the failed parallel-teacher and old spatial-probability shortcuts are absent.
"""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def source_root():
    return REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'


def prepare(root,full=False):
    import numpy as np
    assert not root.exists(),'Inspect or resume the existing explicit plan; do not overwrite it.'
    cache=REPO/'outputs/qwen3_sadi_source_semantic_cache_20261119_v2'
    source=source_root();generation=Path(read(source/'protocol.json')['source']).parent/'generations.json'
    index=np.load(cache/'index.npz',allow_pickle=False)
    ids=index['image_id'];unique=list(dict.fromkeys(map(int,ids)))
    if full:
        preflight=REPO/'outputs/qwen3_native_sadi_source_preflight_20261119/verification.json'
        assert read(preflight)['passed']
        selected=unique;shards=2
    else:
        preflight=None;selected=read(cache/'protocol.json')['prototype_images'];shards=1
    buckets=defaultdict(list)
    for image in selected:
        counts=set(map(int,index['visual_tokens'][ids==image]));assert len(counts)==1
        buckets[counts.pop()].append(image)
    groups=[]
    for visual,images in buckets.items():
        for offset in range(0,len(images),16):groups.append(dict(ids=images[offset:offset+16],visual_tokens=visual))
    paths=[Path(__file__),REPO/'features/qwen3_source_native_capture.py',
        REPO/'features/qwen3_native_risk_i_el.py',REPO/'features/qwen3_online_risk_f.py',
        REPO/'features/dgst_t.py',REPO/'features/qwen3_sar_ems_attention_batch.py',
        REPO/'features/llava_sadi.py',REPO/'features/llava_decoding_evidence.py',
        REPO/'models/qwen3_vl_wrapper.py',REPO/'models/dgst_capture.py',
        REPO/'models/base_wrapper.py',REPO/'utils/config_utils.py',
        REPO/'scripts/run_qwen3_semantic_final_pilot.py',
        REPO/'scripts/generation_time_evidence_drift_qwen3_batched.py',
        REPO/'scripts/generation_time_evidence_drift_qwen3.py',
        REPO/'configs/model_configs_coco4000_512_endac_baselines_v2.yaml',
        cache/'protocol.json',cache/'verification.json',cache/'index.npz',
        source/'protocol.json',source/'compact_features.npz',generation]
    if preflight:paths.append(preflight)
    hashes={str(path):sha(path) for path in paths}
    current=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119/cohort.json'
    reserved=REPO/'outputs/qwen3_mass_matched_visual_reserved500_20261119/cohort.json'
    assert not set(selected)&set(read(current)['coco100'])
    assert not set(selected)&set(read(reserved)['coco500'])
    hashes.update({str(p):sha(p) for p in [current,reserved]})
    root.mkdir();(root/'groups').mkdir();(root/'source').mkdir()
    for path in paths:
        if path.is_relative_to(REPO) and path.suffix in ('.py','.yaml'):
            destination=root/'source'/path.relative_to(REPO)
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
    plan=dict(fixed_at=time.time(),source_sha256=hashes,full_source=full,
        images=selected,groups=groups,shards=shards,max_batch=16,source=str(source),generation=str(generation),
        layers=list(range(5,19)),feature_order=['risk0:36','I0:36','E_l0:36'],
        source_labels_and_original_image_split_unchanged=True,
        old_captions='Teacher-forced raw source captions; not fresh SADI-generated text.',
        trajectory='Native one-query SADI, real KV, complete5--18. No parallel token chunks.',
        semantics='Recompute from actual native-prefill hpre with final norm and full-vocabulary denominator, once per image/layer.',
        reuse='Original images, captions, target IDs/indices, labels and image split. Features reused across future training seeds.',
        no_current_or_reserved_training=True,no_probe_training=True,
        controls='Preflight repeats first fixed group without collector, then compares all last hidden states and final36-layer KV exactly.',
        failed_shortcuts='Do not reuse old log_s.npy values. Do not use parallel teacher forcing.',
        saved='108 numbers per source target, row identity, timing and counters; no hidden states, attention arrays or full logits.')
    write(root/'protocol.json',plan)
    print(dict(stage='prepared',images=len(selected),groups=len(groups),shards=shards),flush=True)


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_source_native_capture import SourceNativeCapture
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    plan=read(root/'protocol.json');assert 0<=shard<plan['shards']
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    old=np.load(Path(plan['source'])/'compact_features.npz',allow_pickle=False)
    captions=read(Path(plan['generation']));model_wrapper=build_wrapper()
    model=model_wrapper.model;core=model.model;device=model_wrapper.device
    all_images=np.asarray(old['image_id']);rows_by_image={image:np.flatnonzero(all_images==image) for image in plan['images']}
    encoder_calls=[0]
    hook=model.model.visual.register_forward_hook(lambda *_:encoder_calls.__setitem__(0,encoder_calls[0]+1))

    def force(inputs,responses,stop,capture=None,targets=None):
        core.rope_deltas=None;past=None;prompt=inputs['input_ids'].shape[1]
        digest=hashlib.sha256();features={}
        with Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture():
            from contextlib import nullcontext
            with capture.capture() if capture is not None else nullcontext():
                for step in range(stop+1):
                    if step==0:
                        current=dict(inputs);current['cache_position']=torch.arange(prompt,device=device)
                    else:
                        previous=[response[step-1] if step<=len(response) else model_wrapper.tokenizer.eos_token_id for response in responses]
                        current=dict(input_ids=torch.tensor(previous,device=device)[:,None],
                            attention_mask=torch.ones((len(responses),prompt+step),device=device,dtype=inputs['attention_mask'].dtype),
                            cache_position=torch.tensor([prompt+step-1],device=device),past_key_values=past)
                    output=core(**current,use_cache=True,return_dict=True);past=output.past_key_values
                    # Full-source runs avoid a host synchronization at every
                    # forced token. Exact trajectory hashing is preflight only.
                    if not plan['full_source']:
                        digest.update(output.last_hidden_state[:,-1].contiguous().view(torch.int16).cpu().numpy().tobytes())
                    if capture is not None and step in targets:
                        requested=targets[step]
                        vectors=capture.features([r[0] for r in requested],[r[2] for r in requested]).cpu().numpy()
                        for (_,row,_),vector in zip(requested,vectors,strict=True):
                            assert row not in features;features[row]=vector
        if not plan['full_source']:
            assert len(past.layers)==36
            for layer in past.layers:
                for value in (layer.keys,layer.values):
                    digest.update(value.contiguous().view(torch.int16).cpu().numpy().tobytes())
        return features,digest.hexdigest()

    started=time.time()
    with torch.inference_mode():
        for group_index,group in enumerate(plan['groups']):
            if group_index%plan['shards']!=shard:continue
            destination=root/'groups'/f'{group_index:04d}'
            assert not destination.with_suffix('.npz').exists(), 'No implicit resume or duplicated compute.'
            batches=list(same_span_batches(model_wrapper,group['ids'],COCO_ROOT,16))
            assert len(batches)==1
            span,items=batches[0];assert [i for i,_ in items]==group['ids']
            assert span[1]-span[0]==group['visual_tokens']
            inputs={key:torch.cat([item[key] for _,item in items],dim=0).to(device) for key in items[0][1]}
            targets=defaultdict(list);responses=[]
            for batch,image in enumerate(group['ids']):
                response=captions[str(image)]['response_token_ids'];responses.append(response)
                for row in rows_by_image[image]:
                    step=int(old['response_token_idx'][row]);token=int(old['target_token_id'][row])
                    assert response[step]==token
                    targets[step].append((batch,int(row),token))
            stop=max(targets);allowed=sorted({token for triples in targets.values() for _,_,token in triples})
            capture=SourceNativeCapture(model,span,allowed)
            began=time.time();before=encoder_calls[0]
            vectors,actual_digest=force(inputs,responses,stop,capture,targets)
            native_encoder_calls=encoder_calls[0]-before
            assert native_encoder_calls==1
            expected=sum(len(rows_by_image[image]) for image in group['ids']);assert len(vectors)==expected
            assert capture.risk_solver_calls==36*expected and capture.feature_calls==expected
            identity=sorted(vectors);matrix=np.stack([vectors[row] for row in identity])
            assert matrix.shape==(expected,108) and np.isfinite(matrix).all()
            exact=True
            if not plan['full_source'] and group_index==0:
                _,reference_digest=force(inputs,responses,stop)
                exact=actual_digest==reference_digest
                assert exact,'Feature collection changed the unchanged native SADI trajectory or KV.'
            np.savez_compressed(destination.with_suffix('.npz'),row=np.asarray(identity),features=matrix)
            information=dict(group=group_index,shard=shard,images=group['ids'],targets=expected,
                max_response_index=stop,batch=len(items),seconds=time.time()-began,
                native_prefill_encoder_calls=native_encoder_calls,head_projection_calls=capture.head_calls,
                feature_calls=capture.feature_calls,risk_solver_calls=capture.risk_solver_calls,
                semantics_seconds=capture.static_seconds,feature_seconds=capture.feature_seconds,
                exact_control=exact if not plan['full_source'] and group_index==0 else 'first_preflight_group',
                no_parallel_teacher=True,no_old_log_s_cache=True,
                feature_sha256=sha(destination.with_suffix('.npz')))
            write(destination.with_suffix('.json'),information);print(information,flush=True)
            del capture,inputs,items,batches
    hook.remove()
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    write(root/f'worker{shard}.json',dict(completed=True,shard=shard,seconds=time.time()-started,
        maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),source_hashes_verified=True))


def audit(root):
    import numpy as np
    plan=read(root/'protocol.json')
    for shard in range(plan['shards']):assert read(root/f'worker{shard}.json')['completed']
    source=np.load(Path(plan['source'])/'compact_features.npz',allow_pickle=False)
    rows=[];values=[];timing=[]
    for group_index,group in enumerate(plan['groups']):
        prefix=root/'groups'/f'{group_index:04d}'
        information=read(prefix.with_suffix('.json'));assert sha(prefix.with_suffix('.npz'))==information['feature_sha256']
        with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as data:
            assert set(map(int,source['image_id'][data['row']]))==set(group['ids'])
            assert data['features'].shape==(len(data['row']),108)
            rows.extend(data['row'].tolist());values.extend(data['features'])
        assert information['native_prefill_encoder_calls']==1
        assert information['risk_solver_calls']==information['targets']*36
        timing.append(information['seconds'])
    assert len(set(rows))==len(rows)
    expected=np.flatnonzero(np.isin(source['image_id'],plan['images']))
    order=np.argsort(rows);rows=np.asarray(rows)[order];values=np.asarray(values)[order]
    np.testing.assert_array_equal(rows,expected)
    assert np.isfinite(values).all() and values.shape==(len(rows),108)
    if not plan['full_source']:assert read(root/'groups/0000.json')['exact_control'] is True
    np.savez_compressed(root/'features.npz',row=rows,features=values)
    for name,expected_sha in plan['source_sha256'].items():assert sha(Path(name))==expected_sha,name
    result=dict(passed=True,images=len(plan['images']),targets=len(rows),groups=len(plan['groups']),
        feature_sha256=sha(root/'features.npz'),protocol_sha256=sha(root/'protocol.json'),
        complete_source_row_identity=True,source_hashes_verified=True,
        source_labels_and_original_split_unchanged=True,features_finite=True,
        group_gpu_seconds_sum=sum(timing),no_old_log_s_reuse=True,no_parallel_teacher=True,
        no_new_captions=True,no_training=True)
    write(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--full',action='store_true')
    parser.add_argument('--worker',type=int);parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root,args.full)
        elif args.worker is not None:worker(args.root,args.worker)
        elif args.audit:audit(args.root)
        else:parser.error('Choose prepare, worker or audit.')
    except Exception:
        if args.root.exists():
            name=f'failure.worker{args.worker}.json' if args.worker is not None else 'failure.json'
            if not (args.root/name).exists():write(args.root/name,dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
