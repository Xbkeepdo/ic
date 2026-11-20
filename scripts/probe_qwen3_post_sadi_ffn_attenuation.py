"""Separate evidence attention and whole-FFN attenuation on frozen prefixes."""

import argparse
import math
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.probe_qwen3_signal_erasure_contrast import CACHE,check

PREVIOUS=REPO/'outputs/qwen3_dual_attention_contrast_probe_20261120'


def prepare(root):
    assert not root.exists()
    old=read(PREVIOUS/'protocol.json');check(old)
    assert read(PREVIOUS/'pipeline_complete.json')['passed']and read(PREVIOUS/'independent_dual_attention_audit.json')['passed']
    cases=[r for s in [0,1]for r in read(PREVIOUS/f'results{s}.json')]
    constant=sum(.5*r['drift']for r in cases)/len(cases)
    paths=[Path(__file__),REPO/'features/qwen3_post_sadi_ffn_attenuation.py',PREVIOUS/'protocol.json',
           PREVIOUS/'pipeline_complete.json',PREVIOUS/'independent_dual_attention_audit.json']
    paths += [PREVIOUS/f'results{s}.json'for s in [0,1]]
    plan=dict(old);plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p)for p in paths}}
    plan.pop('pairs',None)
    plan.update(fixed_at=time.time(),previous=str(PREVIOUS),ffn_constant=constant,
        modes=['attention','ffn','joint','joint_constant','joint_shifted','ffn_constant'],
        action='Full SADI5--18 retained. Actual source19 FFN update scaled by1-gamma without another sourceMLP call; tail20--35 attention mixes visual distribution toward Q with eta=.5D. Each proposal currentKV is fully restored.',
        strengths='D=HellingerSquared(Q,P_hpre) from unchanged native source19. Gamma=.5D or pre-fixed mean(.5D) over six earlier cached positions. Constant comparison matches nominal total FFN attenuation across this six-case probe only.',
        decoding='Primary: direct proposal logits with common native plausibility beta.1 mask. Also report unmasked actual greedy token as a diagnostic, without claiming its independent full-vocabulary verification.',
        relation='FADE-inspired whole FFN attenuation is auxiliary to complete original SADI and SAR attention. Earlier auxiliary experiments used sparse target-margin correction or FFN direction alignment, not this scalar native-update scaling.',
        reference='https://arxiv.org/html/2606.29431v4',
        scope='Old six diagnostic prefixes. No new captions or500. No labels used in coefficient construction. All six arms retained.',
        advance='Inspect wrong and real targets versus attention-only,FFN-only,constant and shifted evidence controls, then complete captions before100/500.')
    check(plan);root.mkdir();(root/'source').mkdir()
    for p in [Path(__file__),REPO/'features/qwen3_post_sadi_ffn_attenuation.py']:(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',plan)


def verify(root):
    import torch
    from features.qwen3_post_sadi_ffn_attenuation import attenuated_anchor,configurations
    gen=torch.Generator().manual_seed(43);checks=0
    for dtype in [torch.float32,torch.bfloat16]:
        for row in range(3):
            residual=torch.randn((3,1,17),generator=gen).to(dtype);update=torch.randn((3,17),generator=gen).to(dtype)
            anchor=residual+update[:,None]
            assert attenuated_anchor(anchor,residual,update,row,0.)is anchor
            for amount in [.125,.5]:
                value=attenuated_anchor(anchor,residual,update,row,amount)
                assert torch.equal(value[[i for i in range(3)if i!=row]],anchor[[i for i in range(3)if i!=row]])
                scaled=(update[row].float()*(1-amount)).to(dtype)
                expected=(residual[row,0].float()+scaled.float()).to(dtype)
                assert torch.equal(value[row,0],expected)
                assert torch.equal(anchor,residual+update[:,None]);checks+=1
    q=torch.tensor([.1,.3,.2,.4]);c=configurations(q,.4,.15)
    assert c['joint'][1:]==(.2,.2)and c['attention'][1:]==(.2,0.)and c['ffn'][1:]==(0.,.2)
    assert torch.equal(c['joint_shifted'][0].roll(-2),q);checks+=2
    plan=read(root/'protocol.json')
    old=[r for s in [0,1]for r in read(PREVIOUS/f'results{s}.json')]
    assert abs(len(old)*plan['ffn_constant']-sum(.5*r['drift']for r in old))<1e-15;checks+=1
    write(root/'math_proof.json',dict(passed=True,checks=checks,cuda_initialized=torch.cuda.is_initialized()))


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_region_memory import RegionMemory
    from features.qwen3_post_sadi_ffn_attenuation import FFNResidualReader,replay_auxiliary,configurations
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    from scripts.qwen3_caption_groups_with_cache import source_groups
    plan=read(root/'protocol.json');check(plan);assert read(root/'math_proof.json')['passed']
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    span,inputs,_=batch(wrapper,group['ids'])
    archived={r['image_id']:r for s in [0,1]for g in source_groups(Path(plan['source'])/f'coco100/sadi.shard{s}.jsonl')for r in g}
    responses=[archived[i]['response_token_ids']for i in group['ids']]
    image=group['cases'][0]['image_id'];row=group['cases'][0]['row']
    with np.load(CACHE/f'image{image}.npz',allow_pickle=False)as saved:
        assert saved['batch_ids'].tolist()==group['ids']and saved['span'].tolist()==list(span)
        assert int(saved['prompt_length'])==inputs['input_ids'].shape[1]
        tokens=saved['token_ids'].tolist();log_s=torch.from_numpy(saved['log_s'][19].copy()).to(wrapper.device)
    reader=FFNResidualReader(model,span,tokens);reader.static[19,row]=dict(log_s=log_s)
    old={(r['image_id'],r['step']):r for r in read(PREVIOUS/f'results{shard}.json')}
    vision=[];encoder=[]
    handle=core.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1))
    capture=core.visual.register_forward_hook(lambda _m,_a,out:vision.append(out))
    prompt=inputs['input_ids'].shape[1];past=None;core.rope_deltas=None
    cases={c['step']:c for c in group['cases']};results=[];zero_checked=False;started=time.time()
    def cache_digest():return digest([t for layer in past.layers for t in [layer.keys,layer.values]])
    with torch.inference_mode(),Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture(),reader.capture():
        for step in range(max(cases)+1):
            if step==0:current=dict(inputs,cache_position=torch.arange(prompt,device=wrapper.device))
            else:
                previous=[r[step-1]if step<=len(r)else wrapper.tokenizer.eos_token_id for r in responses]
                current=dict(input_ids=torch.tensor(previous,device=wrapper.device)[:,None],
                    attention_mask=torch.ones((len(responses),prompt+step),device=wrapper.device,dtype=inputs['attention_mask'].dtype),
                    past_key_values=past,cache_position=torch.tensor([prompt+step-1],device=wrapper.device))
            output=core(**current,use_cache=True,return_dict=True);past=output.past_key_values
            assert reader.step==step and all(past.get_seq_length(i)==prompt+step for i in range(36))
            if step==0:
                assert len(vision)==len(encoder)==1;capture.remove()
                memory=RegionMemory(wrapper,inputs,span,vision.pop(),output.last_hidden_state[:,-1:])
            if step not in cases:continue
            case=cases[step];token=case['token_id'];row=case['row']
            reference=output.last_hidden_state[:,-1:].clone();native=model.get_output_embeddings()(reference).float()[row,0]
            assert int(native.argmax())==token==responses[row][step]
            assert abs(float(native.log_softmax(0)[token])-old[image,step]['native_logp'])<1e-7
            fingerprint=cache_digest();q,p,drift=reader.distributions(row,token)
            assert q.tolist()==old[image,step]['evidence']and p.tolist()==old[image,step]['preference']
            if not zero_checked:
                zero,calls=replay_auxiliary(memory,reader,span,row,past,reference,q,0.,0.)
                assert not calls and torch.equal(model.get_output_embeddings()(zero).float()[row,0],native)
                assert cache_digest()==fingerprint;zero_checked=True
            eligible=(native>=native.max()+math.log(.1)).nonzero().flatten()
            record=dict(case,native_logp=float(native.log_softmax(0)[token]),native_text=wrapper.tokenizer.decode([token]),
                drift=drift,eligible_ids=eligible.tolist(),native_logits=native[eligible].tolist(),trials=[])
            residual=reader.ffn_residual[row,0];update=reader.current[19]['update'][row]
            local_before=reader.decoder.norm(residual);local_after=reader.decoder.norm(residual+update)
            record['native_ffn_target_logit_gain']=float(model.get_output_embeddings().weight[token].float()@(local_after.float()-local_before.float()))
            for name,(prior,attention_amount,ffn_amount)in configurations(q,drift,plan['ffn_constant']).items():
                original_anchor=reader.anchor
                hidden,calls=replay_auxiliary(memory,reader,span,row,past,reference,prior,attention_amount,ffn_amount)
                assert reader.anchor is original_anchor and cache_digest()==fingerprint
                assert calls==(plan['action_layers']if attention_amount else [])
                logits=model.get_output_embeddings()(hidden).float()[row,0]
                allowed_logits=logits[eligible];chosen=int(eligible[allowed_logits.argmax()]);unmasked=int(logits.argmax())
                record['trials'].append(dict(policy=name,attention_amount=attention_amount,ffn_amount=ffn_amount,
                    new_token=chosen,new_token_text=wrapper.tokenizer.decode([chosen]),changed=chosen!=token,
                    unmasked_token=unmasked,unmasked_token_text=wrapper.tokenizer.decode([unmasked]),
                    eligible_logits=allowed_logits.tolist(),target_logp=float(logits.log_softmax(0)[token]),
                    all36_cache_unchanged=True,source_anchor_restored=True,attention_calls=calls))
            results.append(record);write(root/f'results{shard}.json',results)
            print(dict(image=image,step=step,native=record['native_text'],choices={t['policy']:t['new_token_text']for t in record['trials']}),flush=True)
    handle.remove();assert len(encoder)==1 and reader.head_calls==0 and len(results)==len(cases);check(plan)
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,encoder_calls=len(encoder),
        cached_semantic_head_calls=reader.head_calls,cases=len(results),zero_exact=zero_checked,all_trials_rolled_back=True,
        no_generation=True,additional_source19_mlp_forwards=0,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:
        if args.prepare:prepare(args.root)
        elif args.verify:verify(args.root)
        elif args.worker is not None:worker(args.root,args.worker)
        else:parser.error('Select prepare,verify or worker.')
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
