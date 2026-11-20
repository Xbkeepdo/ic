"""Test context-compatible alternatives guided by the original108 detector."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha

PREVIOUS=REPO/'outputs/qwen3_region_rivals_probe_20261120'
SOURCE=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'


def prepare(root):
    assert not root.exists() and read(PREVIOUS/'pipeline_complete.json')['passed']
    old=read(PREVIOUS/'protocol.json');hashes=dict(old['source_sha256'])
    files=[Path(__file__),REPO/'features/qwen3_candidate_risk_i_el.py',REPO/'features/qwen3_native_risk_i_el.py',
        REPO/'features/qwen3_region_guidance.py',REPO/'features/qwen3_risk_i_el_probe.py',
        PREVIOUS/'protocol.json',PREVIOUS/'results0.json',PREVIOUS/'results1.json']
    for s in [0,1]:files.extend(PREVIOUS.glob(f'visual19.shard{s}.*.npz'))
    hashes.update({str(p):sha(p) for p in files})
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    root.mkdir();(root/'source').mkdir()
    for path in files:
        if path.suffix=='.py':(root/'source'/path.name).write_bytes(path.read_bytes())
    baseline=read(SOURCE/'protocol.json')
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,groups=old['groups'],probe=baseline['probe'],
        candidates='Original target plus one highest-native-logp token per other object class in native top32, excluding classes already in the actual prefix.',
        sam_rival='Maximize native class probability mass within top32 * original108 p_real. Original threshold remains unchanged for online eligibility.',
        calibrated_rival='Maximize native class probability mass within top32 * lift; lift=write-weighted semantic probability / whole-image mean semantic probability.',
        support_map='relu(p_rival/mean(p_rival)-p_target/mean(p_target))*native patch-write norm; fixed3x3 peak.',
        confidence='For SAM: relu(p_real_rival-p_real_target)/(p_real_rival+p_real_target); calibrated control uses analogous lift difference.',
        budget='5 * native fullSAR/FFN HellingerSquared * confidence; shift control retains SAM rival and budget.',
        policies=['sam','calibrated','sam_shifted'],layers=list(range(20,36)),
        cache='Candidate-independent writes computed once per layer; native static semantics shared across candidates/positions. Existing layer19 selected309 probability cache checked exactly before use.',
        proof='Native original target/logp and original108 gate verified against frozen source, candidate feature implementation checked exactly against original reader; all trials rollback.',
        no_new_captions=True,no_ground_truth_candidate_insertion=True))


def worker(root,shard):
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_region_guidance import LiveRegionReader
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_candidate_risk_i_el import candidate_features
    from features.qwen3_region_memory import rectangle
    from features.qwen3_region_consistency import RegionConsistency
    from features import qwen3_region_consistency as region
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from features.qwen3_bounded_visual_tilt import evidence_preference
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    from scripts.probe_qwen3_region_counterevidence import signed_attention
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan=read(root/'protocol.json')
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    allowed=read(SOURCE/'object_lexicon.json')['token_ids'];parser=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    classes=parser.token_classes(wrapper.tokenizer,allowed);aliases={}
    for token in allowed:aliases.setdefault(classes[token],[]).append(token)
    span,inputs,_=batch(wrapper,group['ids']);prompt=inputs['input_ids'].shape[1]
    baseline={r['image_id']:r for s in [0,1] for g in source_groups(SOURCE/f'coco100/sadi.shard{s}.jsonl') for r in g}
    responses=[baseline[i]['response_token_ids'] for i in group['ids']]
    old={(r['image_id'],r['step']):r for r in read(PREVIOUS/f'results{shard}.json')}
    flags=SimpleNamespace(suspended=False,_replaying=False,replaying_tail=False,anchor=None,tail_kwargs=None)
    reader=LiveRegionReader(model,span,allowed,owner=flags)
    probe=FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    vision=[];hidden=[];encoders=[];cases={};past=None;records=[];checked_features=False;started=time.time()
    for case in group['cases']:cases.setdefault(case['step'],[]).append(case)
    def source(_m,_a,value):
        if not flags.suspended:flags.anchor=value[:,-1:].clone()
    def tail(_m,_a,kwargs):
        if not flags.suspended:flags.tail_kwargs=dict(kwargs)
    def finish(_m,_a,out):
        if not flags.suspended:hidden[:]=[out.last_hidden_state[:,-1:].clone()]
    core.rope_deltas=None
    with torch.inference_mode(),ExitStack() as stack:
        stack.enter_context(Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture())
        stack.enter_context(reader.capture())
        stack.callback(reader.decoder.layers[19].register_forward_hook(source).remove)
        stack.callback(reader.decoder.layers[20].register_forward_pre_hook(tail,with_kwargs=True).remove)
        stack.callback(reader.decoder.register_forward_hook(finish).remove)
        capture=core.visual.register_forward_hook(lambda _m,_a,out:vision.append(out));stack.callback(capture.remove)
        stack.callback(core.visual.blocks[0].register_forward_hook(lambda *_:encoders.append(1)).remove)
        for step in range(max(cases)+1):
            if step==0:current=dict(inputs,cache_position=torch.arange(prompt,device=wrapper.device))
            else:
                previous=[t[step-1] if step<=len(t) else wrapper.tokenizer.eos_token_id for t in responses]
                current=dict(input_ids=torch.tensor(previous,device=wrapper.device)[:,None],
                    attention_mask=torch.ones((len(responses),prompt+step),device=wrapper.device,dtype=inputs['attention_mask'].dtype),
                    past_key_values=past,cache_position=torch.tensor([prompt+step-1],device=wrapper.device))
            output=model(**current,use_cache=True,return_dict=True,logits_to_keep=1);past=output.past_key_values
            assert reader.step==step
            if step==0:
                assert len(vision)==1;capture.remove();memory=RegionConsistency(wrapper,inputs,span,vision.pop(),hidden[0])
            if step not in cases:continue
            native=hidden[0].clone();logits=output.logits.float()[:,0]
            for case in cases[step]:
                row,token=case['row'],case['token_id'];target=classes[token]
                logp=logits[row].log_softmax(0);before=float(logp[token])
                assert int(logits[row].argmax())==token and abs(before-old[case['image_id'],step]['native_logp'])<1e-7
                prior=parser.classes(wrapper.tokenizer.decode(responses[row][:step],skip_special_tokens=True,clean_up_tokenization_spaces=False))
                top=logits[row].topk(32).indices.tolist();candidates={target:token}
                for t in top:
                    if t in classes and classes[t] not in prior:candidates.setdefault(classes[t],t)
                names=list(candidates);tokens=list(candidates.values());features,norms=candidate_features(reader,row,tokens)
                if not checked_features:
                    original=reader.features([row],[token]);assert torch.equal(features[:1],original);checked_features=True
                real=probe(features).tolist();p_real=dict(zip(names,real,strict=True))
                assert abs(p_real[target]-case['cached_p_real'])<1e-5
                np.savez_compressed(root/f'features108.{case["image_id"]}.{step}.npz',features=features.cpu().numpy(),tokens=np.asarray(tokens))
                static=reader._static(19,row)['log_s'];probability=static.expm1()/reader.head.weight.shape[0]
                with np.load(PREVIOUS/f'visual19.shard{shard}.row{row}.npz') as cache:
                    assert cache['batch_ids'].tolist()==group['ids'] and cache['prompt']==prompt and cache['token_ids'].tolist()==list(reader.tokens)
                    assert np.array_equal(probability.cpu().numpy(),cache['probabilities'])
                q,p=evidence_preference(norms,static[:,reader.columns[token]],reader.visual[19][row],reader.current[19]['update'][row])
                drift=float((1-(q*p).sqrt().sum()).clamp(0,1));weights=norms/norms.sum().clamp_min(1e-30)
                maps={c:probability[:,[reader.columns[t] for t in aliases[c]]].sum(-1) for c in names}
                mean={c:float(v.mean()) for c,v in maps.items()}
                lift={c:float((weights*v).sum())/max(mean[c],1e-30) for c,v in maps.items()}
                native_mass={c:sum(float(logp[t].exp()) for t in top if classes.get(t)==c) for c in names}
                before_cache=digest([t for layer in past.layers for t in [layer.keys,layer.values]])
                zero,_=memory.attenuate(row,past,reader,(0,0,3,3),0.,native,zero=True);assert torch.equal(zero,native)
                trials=[]
                for policy in plan['policies']:
                    score=lift if policy=='calibrated' else p_real
                    others=[c for c in names if c!=target]
                    if not others:trials.append(dict(policy=policy,rival=None,skipped='No unseen alternative class'));continue
                    rival=max(others,key=lambda c:(native_mass[c]*score[c],c))
                    confidence=max(0.,score[rival]-score[target])/max(1e-30,score[rival]+score[target])
                    heat=(maps[rival]/max(mean[rival],1e-30)-maps[target]/max(mean[target],1e-30)).clamp_min(0)*norms
                    rect=rectangle(heat,*memory.grid(row),'shifted' if policy=='sam_shifted' else 'evidence')
                    strength=5*drift*confidence
                    if strength:
                        with patch.object(region,'mask_attention',signed_attention):proposal,error=memory.attenuate(row,past,reader,rect,-strength,native)
                    else:proposal,error=native,0.
                    final=reader.head(proposal).float()[row,0];new=int(final.argmax());after=float(final.log_softmax(0)[token]);five=final.topk(5).indices.tolist()
                    trials.append(dict(policy=policy,rival=rival,confidence=confidence,strength=strength,rect=rect,
                        top5_ids=five,top5_text=[wrapper.tokenizer.decode([t]) for t in five],
                        target_logp_delta=after-before,changed=new!=token,would_accept=new!=token and after<before,visual_mass_relative_error=error))
                assert digest([t for layer in past.layers for t in [layer.keys,layer.values]])==before_cache
                record=dict(case,target=target,prior_classes=sorted(prior),p_real=p_real,lift=lift,native_class_mass=native_mass,
                    native_top32=[dict(token=t,text=wrapper.tokenizer.decode([t]),logp=float(logp[t])) for t in top],
                    drift=drift,trials=trials,full36_rollback_and_zero_exact=True,selected309_cache_exact=True)
                records.append(record);print(dict(image=case['image_id'],step=step,target=target,p_real=p_real,lift=lift,
                    proposals=[(t['policy'],t['rival'],t.get('top5_text'),t.get('target_logp_delta')) for t in trials]),flush=True)
    assert len(encoders)==1 and len(records)==len(group['cases'])
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    write(root/f'results{shard}.json',records)
    write(root/f'worker{shard}.json',dict(passed=True,cases=len(records),encoder_calls=1,all_trials_rolled_back=True,
        original_feature_reader_exact=checked_features,no_new_captions=True,seconds=time.time()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:prepare(a.root) if a.prepare else worker(a.root,a.worker)
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
