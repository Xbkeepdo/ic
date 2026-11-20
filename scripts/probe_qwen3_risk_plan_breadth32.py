"""All cached eligible positions in the first two original SADI B16 groups."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.probe_qwen3_signal_erasure_contrast import check
from scripts.probe_qwen3_risk_plan_attention import verify
from scripts.qwen3_caption_groups_with_cache import source_groups

PREVIOUS=REPO/'outputs/qwen3_risk_plan_attention_probe_20261120'
FEATURES=REPO/'outputs/qwen3_full_risk_f_trace_20261118'
CONTROLS=REPO/'outputs/qwen3_live_preference_contrast100_20261120'
LEXICON=REPO/'outputs/qwen3_clva_signal_preflight10_v2_20261120/object_lexicon.json'
MODEL=Path('/root/rivermind-data/models/qwen3_vl_8b')


def prepare(root):
    assert not root.exists()
    old=read(PREVIOUS/'protocol.json');check(old)
    assert read(PREVIOUS/'pipeline_complete.json')['passed']and read(PREVIOUS/'independent_risk_plan_audit.json')['passed']
    groups=[];paths=[Path(__file__),PREVIOUS/'protocol.json',PREVIOUS/'pipeline_complete.json',
        PREVIOUS/'independent_risk_plan_audit.json',FEATURES/'protocol.json',LEXICON]
    for shard in [0,1]:
        captions=CONTROLS/f'coco100/sadi.shard{shard}.jsonl';features=FEATURES/f'features.shard{shard}.jsonl'
        first=source_groups(captions)[0];ids=[r['image_id']for r in first];assert len(ids)==16
        all_features=[json.loads(line)for line in features.read_bytes().splitlines()]
        cases=[dict(image_id=r['image_id'],row=ids.index(r['image_id']),step=r['step'],token_id=r['token_id'],
                    canonical=r['canonical'],cached_risk19=r['native_features'][19],
                    cached_f19=r['native_features'][55],cached_i19=r['native_I'][19])
               for r in all_features if r['image_id']in ids]
        assert len(cases)==len({(r['image_id'],r['step'])for r in cases})
        assert not {243882,353411}.intersection(ids)
        for case in cases:assert first[case['row']]['response_token_ids'][case['step']]==case['token_id']
        groups.append(dict(shard=shard,ids=ids,cases=sorted(cases,key=lambda r:(r['step'],r['row'])),captions=str(captions)))
        paths.extend([captions,features])
    assert sum(len(g['cases'])for g in groups)==69
    model_files={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                 for p in [*sorted(MODEL.glob('*.safetensors')),MODEL/'config.json',MODEL/'model.safetensors.index.json']}
    assert len(model_files)>=3
    plan=dict(old);plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p)for p in paths}}
    plan.update(fixed_at=time.time(),previous=str(PREVIOUS),groups=groups,images=32,cases=69,model_files=model_files,
        selection='First complete B16 group of each original shard, all69 previously eligible positions. No false/real or score filtering; none from two earlier diagnostic images. Same repeated-development100, not independent test.',
        cached_feature_check='Every original19-layer risk/F_log/I against frozen full-risk source; F/I exact and risk float32 within1e-6 relative tolerance.',
        saved='Scalar case/proposal logits and sparse plans; reusable source19 selected309 log_s with original group/span/prompt and visual-hpre SHA. No full hidden/attention/vocabulary dumps.',
        advance='Broader mechanism screen only; inspect changed false/real contexts and full descriptions before100 or reserved500. Existing6 alone gave fork->nap also in independent coupling, no unique false correction.')
    check(plan);root.mkdir();(root/'source').mkdir()
    for p in [Path(__file__),REPO/'features/qwen3_risk_plan_attention.py']:(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',plan)


def worker(root,shard):
    import numpy as np
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    from features.qwen3_region_memory import RegionMemory,RegionEvidenceReader
    from features.qwen3_risk_plan_attention import original_risk_plan,plan_attention
    from features.qwen3_online_risk_f import native_write_norms
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    plan=read(root/'protocol.json');check(plan);assert read(root/'math_proof.json')['passed']
    for name,stat in plan['model_files'].items():assert Path(name).stat().st_size==stat['size']and Path(name).stat().st_mtime_ns==stat['mtime_ns']
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model;core=model.model
    span,inputs,_=batch(wrapper,group['ids']);source=source_groups(Path(group['captions']))[0]
    assert [r['image_id']for r in source]==group['ids']
    responses=[r['response_token_ids']for r in source];tokens=read(LEXICON)['token_ids'];assert len(tokens)==309
    reader=RegionEvidenceReader(model,span,tokens);vision=[];encoder=[]
    handle=core.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1))
    capture=core.visual.register_forward_hook(lambda _m,_a,out:vision.append(out))
    prompt=inputs['input_ids'].shape[1];past=None;core.rope_deltas=None
    cases=defaultdict(list)
    for case in group['cases']:cases[case['step']].append(case)
    results=[];zero_checked=False;started=time.time();risk_errors=[]
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
            reference=output.last_hidden_state[:,-1:].clone();all_native=model.get_output_embeddings()(reference).float()[:,0]
            for case in cases[step]:
                row=case['row'];token=case['token_id'];native=all_native[row]
                assert int(native.argmax())==token==responses[row][step]
                fingerprint=cache_digest();state=reader.current[19];log_s=reader._static(19,row)['log_s'][:,reader.columns[token]]
                args=(reader.visual[19][row],state['hpre'][row],state['update'][row],state['attention'][row],log_s)
                support,kernels,problem=original_risk_plan(*args)
                risk_error=abs(float(np.float32(problem['risk']))-case['cached_risk19']);risk_errors.append(risk_error)
                assert risk_error<=1e-7+1e-6*abs(case['cached_risk19'])
                values=qwen.repeat_kv(reader.values[19][row:row+1],reader.decoder.layers[19].self_attn.num_key_value_groups)[0]
                norms=native_write_norms(values,state['attention'][row],reader.decoder.layers[19].self_attn.o_proj.weight)
                f_log=float((norms*(1+log_s)).sum());intensity=float(norms.sum())
                assert f_log==case['cached_f19']and intensity==case['cached_i19']
                if not zero_checked:
                    with plan_attention(span,row,support,kernels['plan'],0.):zero=memory.replay(row,past,reader,{},reference,zero=True)
                    assert torch.equal(model.get_output_embeddings()(zero).float()[:,0],all_native)
                    assert cache_digest()==fingerprint;zero_checked=True
                eligible=(native>=native.max()+math.log(.1)).nonzero().flatten()
                record=dict(case,native_logp=float(native.log_softmax(0)[token]),native_text=wrapper.tokenizer.decode([token]),
                    support=support.tolist(),problem=problem,eligible_ids=eligible.tolist(),native_logits=native[eligible].tolist(),
                    source_risk_absolute_error=risk_error,source_F_I_exact=True,trials=[])
                for name,kernel in kernels.items():
                    with plan_attention(span,row,support,kernel,problem['amount'])as calls:hidden=memory.replay(row,past,reader,{},reference)
                    assert cache_digest()==fingerprint and calls==plan['action_layers']
                    logits=model.get_output_embeddings()(hidden).float()[row,0];allowed=logits[eligible]
                    chosen=int(eligible[allowed.argmax()]);unmasked=int(logits.argmax())
                    record['trials'].append(dict(policy=name,new_token=chosen,new_token_text=wrapper.tokenizer.decode([chosen]),
                        unmasked_token=unmasked,unmasked_token_text=wrapper.tokenizer.decode([unmasked]),changed=chosen!=token,
                        eligible_logits=allowed.tolist(),target_logp=float(logits.log_softmax(0)[token]),all36_cache_unchanged=True,action_layers=calls))
                results.append(record);write(root/f'results{shard}.json',results)
            print(dict(shard=shard,step=step,completed=len(results),total=len(group['cases'])),flush=True)
    handle.remove();assert len(encoder)==1 and len(results)==len(group['cases']);check(plan)
    cached_rows=sorted(row for layer,row in reader.static if layer==19)
    cache=root/f'semantic19.shard{shard}.npz'
    np.savez_compressed(cache,image_ids=np.array([group['ids'][i]for i in cached_rows]),rows=np.array(cached_rows),
        log_s=np.stack([reader.static[19,i]['log_s'].float().cpu().numpy()for i in cached_rows]),
        token_ids=np.array(reader.tokens),batch_ids=np.array(group['ids']),span=np.array(span),prompt=prompt,
        hpre_sha256=np.array([digest([reader.visual[19][i]])for i in cached_rows]),protocol_sha256=sha(root/'protocol.json'))
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,encoder_calls=len(encoder),
        semantic_head_calls=reader.head_calls,cases=len(results),zero_exact=zero_checked,all_trials_rolled_back=True,
        original_risk_max_absolute_error=max(risk_errors),original_F_I_all_exact=True,
        semantic_cache_sha256=sha(cache),semantic_cache_bytes=cache.stat().st_size,cached_images=len(cached_rows),
        no_generation=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:
        if a.prepare:prepare(a.root)
        elif a.verify:verify(a.root)
        elif a.worker is not None:worker(a.root,a.worker)
        else:p.error('Select a stage.')
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
