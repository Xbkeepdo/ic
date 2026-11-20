"""Read native next-token choices at fixed archived source target positions.

Reuse the original first two B16 source groups and forced source prefixes.
Read one head at target steps, not every text step. Recompute only one108
feature vector in each group to anchor this audit to the saved source trace.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def prepare(root):
    assert not root.exists()
    producer=REPO/'outputs/qwen3_native_sadi_source_features_20261119'
    verified=read(producer/'verification.json');assert verified['passed']
    plan=read(producer/'protocol.json');groups=plan['groups'][:2]
    assert all(len(group['ids'])==16 for group in groups)
    paths=[Path(__file__),producer/'protocol.json',producer/'verification.json',producer/'features.npz',
           Path(plan['source'])/'compact_features.npz',Path(plan['generation']),
           REPO/'features/qwen3_source_native_capture.py',REPO/'features/coco_object_history.py',
           REPO/'coco-labeling/coco_chair.py',
           REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119/object_lexicon.json']
    hashes=dict(plan['source_sha256']);hashes.update({str(p):sha(p) for p in paths})
    for path,expected in hashes.items():assert sha(Path(path))==expected,path
    root.mkdir()
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,
        producer=str(producer),source=plan['source'],generation=plan['generation'],groups=groups,shards=2,
        selection='First two original276 source groups, before inspecting their next-token consistency.',
        trajectory='Same original B16, prompt, fullSADI5--18 and forced archived raw response IDs; real KV.',
        head='Project all16 current rows only at source-target steps, then read requested rows. No full-vocabulary read at non-target steps.',
        feature_anchor='First source row in each group; original108 readout, assert rtol1e-5/atol1e-6 to stored source.',
        interpretation='Conditional native argmax under a forced old prefix; this is not a freshly generated SADI caption.',
        saved='Source token, native top1/top5, probability, rank, source label, runtime-vocabulary membership and optional known canonical class.',
        no_training=True,no_threshold_changes=True,no_new_captions=True))
    print(dict(prepared=True,groups=2,images=32),flush=True)


def worker(root,shard):
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_source_native_capture import SourceNativeCapture
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    plan=read(root/'protocol.json');assert shard in [0,1]
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    old=np.load(Path(plan['source'])/'compact_features.npz',allow_pickle=False)
    captions=read(Path(plan['generation']));group=plan['groups'][shard]
    wrapper=build_wrapper();model=wrapper.model;core=model.model
    (span,items),=same_span_batches(wrapper,group['ids'],COCO_ROOT,16)
    assert [i for i,_ in items]==group['ids'] and span[1]-span[0]==group['visual_tokens']
    inputs={key:torch.cat([item[key] for _,item in items],dim=0).to(wrapper.device) for key in items[0][1]}
    targets=defaultdict(list);responses=[]
    for batch,image in enumerate(group['ids']):
        response=captions[str(image)]['response_token_ids'];responses.append(response)
        for row in np.flatnonzero(old['image_id']==image):
            index=int(old['response_token_idx'][row]);token=int(old['target_token_id'][row])
            assert response[index]==token;targets[index].append((batch,int(row),token))
    wanted=sorted({token for values in targets.values() for _,_,token in values})
    anchor=min(row for values in targets.values() for _,row,_ in values)
    with np.load(Path(plan['producer'])/'features.npz',allow_pickle=False) as saved:
        assert saved['row'][anchor]==anchor;reference=saved['features'][anchor].copy()
    allowed=read(REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119/object_lexicon.json')['token_ids']
    parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(wrapper.tokenizer,allowed)
    capture=SourceNativeCapture(model,span,wanted)
    core.rope_deltas=None;past=None;prompt=inputs['input_ids'].shape[1]
    encoder=[0];handle=model.model.visual.register_forward_hook(lambda *_:encoder.__setitem__(0,encoder[0]+1))
    results=[];anchor_check=None;head_calls=0;started=time.time()
    with torch.inference_mode(),Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture(),capture.capture():
        for step in range(max(targets)+1):
            if step==0:
                current=dict(inputs);current['cache_position']=torch.arange(prompt,device=wrapper.device)
            else:
                ids=[response[step-1] if step<=len(response) else wrapper.tokenizer.eos_token_id for response in responses]
                current=dict(input_ids=torch.tensor(ids,device=wrapper.device)[:,None],
                    attention_mask=torch.ones((16,prompt+step),device=wrapper.device,dtype=inputs['attention_mask'].dtype),
                    past_key_values=past,cache_position=torch.tensor([prompt+step-1],device=wrapper.device))
            output=core(**current,use_cache=True,return_dict=True);past=output.past_key_values
            if step not in targets:continue
            logits=model.get_output_embeddings()(output.last_hidden_state[:,-1:]).float()[:,0];head_calls+=1
            for batch,row,token in targets[step]:
                values=logits[batch];indices=values.topk(5).indices.tolist();native=indices[0]
                context=wrapper.tokenizer.decode(responses[batch][max(0,step-8):step+9],skip_special_tokens=True)
                results.append(dict(row=row,image_id=int(old['image_id'][row]),step=step,label_real=int(old['label'][row]),
                    source_train=bool(old['train_mask'][row]),source_token=token,source_text=wrapper.tokenizer.decode([token]),
                    native_top1=native,native_text=wrapper.tokenizer.decode([native]),native_top5=indices,
                    native_top5_text=[wrapper.tokenizer.decode([i]) for i in indices],
                    source_probability=float((values[token]-values.logsumexp(0)).exp()),
                    native_probability=float((values[native]-values.logsumexp(0)).exp()),
                    native_minus_source_logit=float(values[native]-values[token]),
                    source_rank=int((values>values[token]).sum())+1,exact_token_match=native==token,
                    source_runtime_eligible=token in allowed,native_runtime_eligible=native in allowed,
                    source_class=classes.get(token),native_class=classes.get(native),context=context))
                if row==anchor:
                    actual=capture.features([batch],[token]).cpu().numpy()[0]
                    np.testing.assert_allclose(actual,reference,rtol=1e-5,atol=1e-6)
                    anchor_check=dict(passed=True,row=row,maximum_absolute_error=float(np.abs(actual-reference).max()),
                        maximum_relative_error=float(np.max(np.abs(actual-reference)/np.maximum(np.abs(reference),1e-6))))
    handle.remove();assert encoder[0]==1 and anchor_check is not None
    assert len(results)==sum(len(values) for values in targets.values())
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    write(root/f'results{shard}.json',results)
    write(root/f'worker{shard}.json',dict(passed=True,shard=shard,images=group['ids'],targets=len(results),
        head_calls_at_target_steps=head_calls,encoder_calls=encoder[0],anchor_check=anchor_check,
        seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print(dict(shard=shard,targets=len(results),seconds=time.time()-started),flush=True)


def analyze(root):
    plan=read(root/'protocol.json');rows=[]
    for shard in [0,1]:
        assert read(root/f'worker{shard}.json')['passed'];rows.extend(read(root/f'results{shard}.json'))
    assert len({r['row'] for r in rows})==len(rows)==101
    def stats(selected):
        return dict(targets=len(selected),exact_token_matches=sum(r['exact_token_match'] for r in selected),
            source_in_native_top5=sum(r['source_token'] in r['native_top5'] for r in selected),
            native_runtime_eligible=sum(r['native_runtime_eligible'] for r in selected),
            known_same_canonical=sum(r['source_class'] is not None and r['source_class']==r['native_class'] for r in selected))
    output=dict(passed=True,protocol_sha256=sha(root/'protocol.json'),all=stats(rows),
        real=stats([r for r in rows if r['label_real']==1]),hall=stats([r for r in rows if r['label_real']==0]),
        runtime_source_real=stats([r for r in rows if r['label_real']==1 and r['source_runtime_eligible']]),
        runtime_source_hall=stats([r for r in rows if r['label_real']==0 and r['source_runtime_eligible']]),
        no_new_captions=True,no_fitting=True,no_population_or_mitigation_claim=True,
        source_hashes_verified=True)
    for name,expected in plan['source_sha256'].items():assert sha(Path(name))==expected,name
    write(root/'analysis.json',output);print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1])
    parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root)
        elif args.worker is not None:worker(args.root,args.worker)
        elif args.analyze:analyze(args.root)
        else:parser.error('Choose prepare, worker or analyze.')
    except Exception:
        if args.root.exists():
            name=f'failure.worker{args.worker}.json' if args.worker is not None else 'failure.json'
            if not (args.root/name).exists():write(args.root/name,dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
