"""Compare native object alternatives using semantic support in visual writes.

Six old development positions, unchanged SADI prefixes, no GT-based selection.
All attention trials are rolled back. This is not a caption evaluation.
"""

import argparse
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha

SOURCE=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
PREVIOUS=REPO/'outputs/qwen3_region_counterevidence_probe_20261120'


def prepare(root):
    assert not root.exists()
    assert read(REPO/'outputs/qwen3_region_guidance100_20261120/analysis_recovery_complete.json')['passed']
    old=read(PREVIOUS/'protocol.json');hashes=dict(old['source_sha256'])
    files=[Path(__file__),PREVIOUS/'protocol.json',PREVIOUS/'results0.json',PREVIOUS/'results1.json',
        SOURCE/'object_lexicon.json',REPO/'features/coco_object_history.py',
        REPO/'scripts/run_qwen3_region_guidance.py']
    hashes.update({str(p):sha(p) for p in files})
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    root.mkdir();(root/'source').mkdir()
    (root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source=str(SOURCE),groups=old['groups'],source_sha256=hashes,
        topk=32,read_layer=19,action_layers=list(range(20,36)),gamma=5,
        support='Class alias probability sum; S_write=sum(normalized native patch-write norms * class probability); S_uniform=mean(class probability); S_FFN=sum(native FFN spatial preference * class probability).',
        candidates='Classes represented by object tokens in actual final top32, plus the original class. Never insert an image-labelled correct answer.',
        rival='Highest support non-target candidate; independently rank by write-weighted or uniform support.',
        confidence='relu(S_rival-S_target)/(S_rival+S_target), denominator floor1e-30.',
        location='Peak of relu(p_rival-p_target) times native write norm (weighted), or peak of the unweighted positive semantic difference (uniform); fixed3x3.',
        budget='5*HellingerSquared(Q_original_target_fullSAR,P_native_FFN)*confidence.',
        controls=['write_weighted','uniform_semantic','write_weighted_shifted'],
        action='Boost selected ROI using the existing visual-mass-preserving operator; native values retained; all proposals rolled back.',
        cache='One selected309 probability image-prefix read per relevant image at layer19, reused across positions/classes. Save only selected probabilities, native write norms and FFN preference for CPU followups; no hidden or full-vocabulary arrays.',
        scope='Old6 forced fixed-prefix diagnostics, including online-negative gates as explicit controls. No fresh captions or improvement claim.'))


def worker(root,shard):
    import numpy as np
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_region_memory import RegionEvidenceReader,rectangle
    from features.qwen3_region_consistency import RegionConsistency
    from features import qwen3_region_consistency as region
    from features.qwen3_online_risk_f import native_write_norms
    from features.qwen3_bounded_visual_tilt import evidence_preference
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.probe_qwen3_region_counterevidence import signed_attention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan=read(root/'protocol.json')
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    allowed=read(SOURCE/'object_lexicon.json')['token_ids']
    classes=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py').token_classes(wrapper.tokenizer,allowed)
    aliases={}
    for token in allowed:aliases.setdefault(classes[token],[]).append(token)
    span,inputs,_photos=batch(wrapper,group['ids']);assert span[1]-span[0]==group['visual_tokens']
    reference={r['image_id']:r for s in [0,1] for g in source_groups(SOURCE/f'coco100/sadi.shard{s}.jsonl') for r in g}
    old={(r['image_id'],r['step']):r for r in read(PREVIOUS/f'results{shard}.json')}
    responses=[reference[i]['response_token_ids'] for i in group['ids']]
    reader=RegionEvidenceReader(model,span,allowed);vision=[];encoders=[]
    h=core.visual.register_forward_hook(lambda _m,_a,output:vision.append(output))
    count=core.visual.blocks[0].register_forward_hook(lambda *_:encoders.append(1))
    cases={}
    for case in group['cases']:cases.setdefault(case['step'],[]).append(case)
    past=None;prompt=inputs['input_ids'].shape[1];core.rope_deltas=None;records=[];saved=set();started=time.time()
    with torch.inference_mode(),Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture(),reader.capture():
        for step in range(max(cases)+1):
            if step==0:current=dict(inputs,cache_position=torch.arange(prompt,device=wrapper.device))
            else:
                tokens=[t[step-1] if step<=len(t) else wrapper.tokenizer.eos_token_id for t in responses]
                current=dict(input_ids=torch.tensor(tokens,device=wrapper.device)[:,None],
                    attention_mask=torch.ones((len(responses),prompt+step),device=wrapper.device,dtype=inputs['attention_mask'].dtype),
                    past_key_values=past,cache_position=torch.tensor([prompt+step-1],device=wrapper.device))
            out=core(**current,use_cache=True,return_dict=True);past=out.past_key_values
            assert reader.step==step
            if step==0:
                assert len(vision)==1;h.remove()
                memory=RegionConsistency(wrapper,inputs,span,vision.pop(),out.last_hidden_state[:,-1:])
            if step not in cases:continue
            native=out.last_hidden_state[:,-1:].clone();logits=reader.head(native).float()[:,0]
            for case in cases[step]:
                row,token=case['row'],case['token_id'];target=classes[token]
                before=float(logits[row].log_softmax(0)[token])
                assert int(logits[row].argmax())==token and abs(before-old[case['image_id'],step]['native_logp'])<1e-7
                state=reader.current[19]
                values=qwen.repeat_kv(reader.values[19][row:row+1],reader.decoder.layers[19].self_attn.num_key_value_groups)[0]
                norms=native_write_norms(values,state['attention'][row],reader.decoder.layers[19].self_attn.o_proj.weight)
                cached=reader._static(19,row)['log_s'];probabilities=cached.expm1()/reader.head.weight.shape[0]
                q,p=evidence_preference(norms,cached[:,reader.columns[token]],reader.visual[19][row],state['update'][row])
                drift=float((1-(q*p).sqrt().sum()).clamp(0,1));weights=norms/norms.sum().clamp_min(1e-30)
                native_top=logits[row].topk(32).indices.tolist()
                candidates=sorted({classes[t] for t in native_top if t in classes}|{target})
                support={};maps={}
                for canonical in candidates:
                    columns=[reader.columns[t] for t in aliases[canonical]];v=probabilities[:,columns].sum(-1);maps[canonical]=v
                    support[canonical]=dict(write=float((weights*v).sum()),uniform=float(v.mean()),
                        ffn=float((p*v).sum()),alias_count=len(columns),native_tokens=[t for t in native_top if classes.get(t)==canonical])
                if row not in saved:
                    np.savez_compressed(root/f'visual19.shard{shard}.row{row}.npz',probabilities=probabilities.cpu().numpy(),
                        token_ids=np.asarray(reader.tokens,dtype=np.int64),image_id=case['image_id'],span=np.asarray(span),prompt=prompt,batch_ids=np.asarray(group['ids']))
                    saved.add(row)
                np.savez_compressed(root/f'query19.{case["image_id"]}.{step}.npz',write_norms=norms.cpu().numpy(),ffn_preference=p.cpu().numpy(),q_target=q.cpu().numpy())
                before_cache=digest([t for layer in past.layers for t in [layer.keys,layer.values]])
                hidden,_=memory.attenuate(row,past,reader,(0,0,3,3),0.,native,zero=True);assert torch.equal(hidden,native)
                trials=[]
                for policy in plan['controls']:
                    score='uniform' if policy=='uniform_semantic' else 'write'
                    others=[c for c in candidates if c!=target]
                    rival=max(others,key=lambda c:(support[c][score],c)) if others else None
                    if rival is None:
                        trials.append(dict(policy=policy,rival=None,skipped='No alternative object class in native top32'));continue
                    a,b=support[target][score],support[rival][score]
                    confidence=max(0.,b-a)/max(1e-30,a+b)
                    heat=(maps[rival]-maps[target]).clamp_min(0)*(1 if score=='uniform' else norms)
                    rect=rectangle(heat,*memory.grid(row),'shifted' if policy.endswith('shifted') else 'evidence')
                    strength=5*drift*confidence
                    if strength:
                        with patch.object(region,'mask_attention',signed_attention):
                            hidden,error=memory.attenuate(row,past,reader,rect,-strength,native)
                    else:hidden,error=native,0.
                    proposed=reader.head(hidden).float()[row,0];top=proposed.topk(5).indices.tolist()
                    after=float(proposed.log_softmax(0)[token]);new=int(proposed.argmax())
                    trials.append(dict(policy=policy,rival=rival,confidence=confidence,strength=strength,rect=rect,
                        top5_ids=top,top5_text=[wrapper.tokenizer.decode([t]) for t in top],
                        target_logp_delta=after-before,changed=new!=token,would_accept=new!=token and after<before,
                        visual_mass_relative_error=error))
                assert digest([t for layer in past.layers for t in [layer.keys,layer.values]])==before_cache
                record=dict(case,target_class=target,native_logp=before,native_top32=[dict(token=t,text=wrapper.tokenizer.decode([t])) for t in native_top],
                    support=support,drift=drift,trials=trials,zero_and_full36_cache_rollback_exact=True)
                records.append(record);print(dict(image=case['image_id'],step=step,target=target,support=support,
                    proposals=[(t['policy'],t['rival'],t.get('top5_text'),t.get('target_logp_delta')) for t in trials]),flush=True)
    count.remove();assert len(encoders)==1 and len(records)==len(group['cases'])
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    write(root/f'results{shard}.json',records)
    write(root/f'worker{shard}.json',dict(passed=True,cases=len(records),encoder_calls=1,no_new_captions=True,
        all_trials_rolled_back=True,seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:prepare(a.root) if a.prepare else worker(a.root,a.worker)
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
