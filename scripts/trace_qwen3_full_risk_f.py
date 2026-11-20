"""Original72-dimensional risk/F transfer, paired in a single native replay."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import trace_qwen3_risk_f_gate as compact
from scripts.verify_qwen3_full_risk_f import TRAIN

OUT=shared.OUT/'qwen3_full_risk_f_trace_20261118'
SOURCE=compact.OUT
CAPTIONS=compact.SOURCE
SEMANTICS=shared.OUT/'qwen3_sadi_pre_attention_i_el_trace'
FILES=['features/qwen3_full_risk_f.py','scripts/verify_qwen3_full_risk_f.py',
       'scripts/trace_qwen3_full_risk_f.py','scripts/analyze_qwen3_full_risk_f.py']


def prepare():
    assert not OUT.exists()
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    cpu=shared.OUT/'qwen3_full_risk_f_cpu/verification.json'
    assert shared.read(cpu)['passed']
    previous=shared.read(SOURCE/'protocol.json')
    hashes=dict(previous['source_sha256'])
    for name,expected in hashes.items():assert shared.sha(Path(name))==expected,name
    for path in [*(REPO/name for name in FILES),cpu,SOURCE/'protocol.json',SOURCE/'analysis_recovery.json',
            SOURCE/'labelled_positions_v2.json',SOURCE/'unlabelled_context_review.json',
            REPO/'features/qwen3_online_i_el.py',
            *(TRAIN/name/'seed43'/f for name in ('risk+F_log','I+E_l') for f in ('model.pt','input_scaler.npz'))]:
        hashes[str(path)]=shared.sha(path)
    for shard,groups in enumerate(previous['groups']):
        path=SOURCE/f'features.shard{shard}.jsonl';hashes[str(path)]=shared.sha(path)
        for index in range(len(groups)):
            path=SEMANTICS/f'semantics/shard{shard}.batch{index}.npz'
            hashes[str(path)]=shared.sha(path)
    OUT.mkdir();(OUT/'logs').mkdir()
    shared.write(OUT/'protocol.json',dict(source=str(SOURCE),captions=str(CAPTIONS),images=100,
        groups=previous['groups'],ids=previous['ids'],max_batch=16,max_new_tokens=512,
        source_sha256=hashes,fixed_at=time.time(),full_model=str(TRAIN/'risk+F_log/seed43'),
        i_el_control=str(TRAIN/'I+E_l/seed43'),threshold_p_real=.5,
        purpose='Read-only full original risk/F and I/El controls, native and local pre-SADI, one replay; no fitting or mitigation.',
        feature_order='risk0..35 then F_log0..35; original K32/tau.03 and both endpoint coefficients1.',
        cache='Shared309 semantic columns for both reads; original36-layer E_l sums reused and cross-checked.',
        eligibility='Exactly the209 previously validated online lexical/first-class positions, including18 unlabelled. All native tokens checked.',
        pre_readout='Only attention logits before local SADI edit on5--18; actual SADI hpre/FFN/values retained. Not a no-SADI model.',
        prospective_screen='At original.5 threshold, <=14 automatically-real positives on191 labelled positions and at least one image-confirmed unsupported object missed by old I/El. No threshold tuning.',
        followup='Qualifying readout only can proceed to a frozen attention-mitigation100, then independent500 if actual metrics and image review support it.',
        saved='Full72 scalar feature vectors/scores and timings, no hidden/attention/full-vocab tensor dump.'))


def preflight(wrapper,nouns):
    import numpy as np
    import torch
    from features.qwen3_full_risk_f import FullOnlineRiskF,EARLY
    from features.qwen3_online_risk_f import OnlineRiskF
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.generation_time_evidence_drift_qwen3 import MODEL,COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    (span,group),=same_span_batches(wrapper,[485532,558955],COCO_ROOT,16)
    inputs={k:torch.cat([v[k] for _,v in group]).to(wrapper.device) for k in group[0][1]}
    targets=[wrapper.tokenizer.encode(' person',add_special_tokens=False)[0],wrapper.tokenizer.encode(' dog',add_special_tokens=False)[0]]
    def run(enabled):
        logits=[];tails=[];checks=[];counts=dict(model=0,head=0,encoder=0)
        def count(name):
            def hook(*_):counts[name]+=1
            return hook
        small=OnlineRiskF(wrapper.model,span,targets) if enabled else None
        def callback(reader,out):
            if reader.step!=1:return
            result=reader.full_features([0,1],targets)
            reference,_=small.features([0,1],targets)
            np.testing.assert_array_equal(result['native'][:,[27,63,28,64]],reference)
            outside=[i for i in range(72) if i%36 not in EARLY]
            np.testing.assert_array_equal(result['native'][:,outside],result['pre'][:,outside])
            assert np.any(result['native']!=result['pre'])
            checks.append(dict(compact27_28_exact=True,only_early_attention_features_change=True,
                maximum_feature_difference=float(np.max(np.abs(result['native']-result['pre'])))))
        full=FullOnlineRiskF(wrapper.model,span,targets,callback=callback) if enabled else None
        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(wrapper.model,span,nouns,'sadi').capture())
            if small:stack.enter_context(small.capture());stack.enter_context(full.capture())
            for module,name in [(wrapper.model,'model'),(wrapper.model.get_output_embeddings(),'head'),(wrapper.model.model.visual,'encoder')]:
                stack.callback(module.register_forward_pre_hook(count(name)).remove)
            def observe(_m,_a,out):
                logits.append(out.logits[:,-1].cpu().clone())
                tails.append(tuple(t[:,:,-1:].cpu().clone() for layer in out.past_key_values.layers for t in (layer.keys,layer.values)))
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            tokens=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=12)).cpu()
        return tokens,logits,tails,checks,counts,full,small
    with torch.inference_mode():native,current=run(False),run(True)
    assert torch.equal(native[0],current[0])
    assert all(torch.equal(a,b) for a,b in zip(native[1],current[1],strict=True))
    assert all(torch.equal(a,b) for left,right in zip(native[2],current[2],strict=True) for a,b in zip(left,right,strict=True))
    assert len(current[3])==1 and current[4]['encoder']==1
    assert current[4]['head']==current[4]['model']+current[5].head_calls+current[6].head_calls
    return dict(passed=True,exact_native_logits_tokens_all36_current_KV=True,checks=current[3],counts=current[4],
        full_semantic_head_calls=current[5].head_calls,compact_reference_semantic_head_calls=current[6].head_calls)


def worker(shard):
    import numpy as np
    import torch
    from features.qwen3_full_risk_f import FullOnlineRiskF
    from features.qwen3_online_i_el import FullIElProbe
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import MODEL,COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    plan=shared.read(OUT/'protocol.json')
    for name,expected in plan['source_sha256'].items():assert shared.sha(Path(name))==expected,name
    wrapper=build_wrapper()
    allowed=frozenset(shared.read(SOURCE/'object_lexicon.json')['token_ids'])
    risk_probe=FullIElProbe(plan['full_model']).eval()
    iel_probe=FullIElProbe(plan['i_el_control']).eval()
    shared.write(OUT/f'preflight.shard{shard}.json',preflight(wrapper,allowed))
    previous=[json.loads(line) for line in (SOURCE/f'features.shard{shard}.jsonl').read_bytes().splitlines()]
    by_position={(r['image_id'],r['step']):r for r in previous}
    eos=wrapper.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    output=OUT/f'features.shard{shard}.jsonl';assert not output.exists()
    max_el_relative=0.
    for index,group in enumerate(shared.source_groups(CAPTIONS/f'coco100/sadi.shard{shard}.jsonl')):
        ids=[r['image_id'] for r in group]
        (span,batch),=same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
        assert [i for i,_ in batch]==ids
        inputs={k:torch.cat([v[k] for _,v in batch]).to(wrapper.device) for k in batch[0][1]}
        with np.load(SEMANTICS/f'semantics/shard{shard}.batch{index}.npz',allow_pickle=False) as data:
            assert data['image_ids'].tolist()==ids and data['span'].tolist()==list(span)
            semantic=data['E_l'].copy();columns={int(t):i for i,t in enumerate(data['token_ids'])}
        records=[]
        def callback(reader,value):
            nonlocal max_el_relative
            targets=value.logits[:,-1].argmax(-1).tolist();selected=[]
            for row,(target,cached) in enumerate(zip(targets,group,strict=True)):
                if reader.step>=len(cached['response_token_ids']):continue
                assert target==cached['response_token_ids'][reader.step],('Changed native token',ids[row],reader.step)
                if (ids[row],reader.step) in by_position:selected.append(row)
            if not selected:return
            requested=[targets[r] for r in selected]
            values=reader.full_features(selected,requested)
            old=[by_position[(ids[r],reader.step)] for r in selected]
            assert all(v['token_id']==token for v,token in zip(old,requested,strict=True))
            np.testing.assert_array_equal(values['native'][:,[27,63,28,64]],np.asarray([r['features'] for r in old],np.float32))
            reused_el=semantic[:,selected,[columns[t] for t in requested]].T.copy()
            np.testing.assert_allclose(values['E_l_from_log_s'],reused_el,rtol=3e-6,atol=1e-9)
            relative=np.max(np.abs(values['E_l_from_log_s']-reused_el)/np.maximum(np.abs(reused_el),1e-9))
            max_el_relative=max(max_el_relative,float(relative))
            scores={name:risk_probe(torch.from_numpy(values[name])).numpy() for name in ('native','pre')}
            iel={name:iel_probe(torch.from_numpy(np.concatenate([values[f'{name}_I'],reused_el],axis=1))).numpy() for name in ('native','pre')}
            for slot,(row,original) in enumerate(zip(selected,old,strict=True)):
                records.append(dict(image_id=ids[row],step=reader.step,token_id=targets[row],canonical=original['canonical'],
                    native_features=values['native'][slot].tolist(),pre_features=values['pre'][slot].tolist(),
                    native_I=values['native_I'][slot].tolist(),pre_I=values['pre_I'][slot].tolist(),E_l=reused_el[slot].tolist(),
                    native_p_real=float(scores['native'][slot]),pre_p_real=float(scores['pre'][slot]),
                    native_i_el_p_real=float(iel['native'][slot]),pre_i_el_p_real=float(iel['pre'][slot]),
                    compact_p_hall=original['p_hall']))
        reader=FullOnlineRiskF(wrapper.model,span,allowed,callback=callback)
        started=time.perf_counter()
        with torch.inference_mode(),Qwen3BatchLateAttention(wrapper.model,span,allowed,'sadi').capture(),reader.capture():
            result=wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=512))
        torch.cuda.synchronize(wrapper.device);seconds=time.perf_counter()-started
        tokens=[trim_eos(row,eos) for row in result[:,inputs['input_ids'].shape[1]:].tolist()]
        assert tokens==[r['response_token_ids'] for r in group]
        assert reader.risk_solver_calls==50*len(records)
        with output.open('a') as stream:
            for r in records:stream.write(json.dumps(r)+'\n')
        shared.write(OUT/f'timing.shard{shard}.batch{index}.json',dict(image_ids=ids,span=list(span),records=len(records),
            exact_native_full_tokens=True,compact_features_exact=True,E_l_max_relative=max_el_relative,
            wall_seconds=seconds,feature_seconds=reader.feature_seconds,static_seconds=reader.static_seconds,
            semantic_head_calls=reader.head_calls,risk_solver_calls=reader.risk_solver_calls,
            maximum_memory_allocated=torch.cuda.max_memory_allocated()))
        print(f'[{shard}] batch={index} images={len(ids)} positions={len(records)} seconds={seconds:.3f}',flush=True)
    assert sum(1 for _ in output.read_text().splitlines())==len(previous)
    shared.write(OUT/f'worker{shard}.json',dict(status='complete',finished=time.time()))


def run(prepared=False):
    import subprocess
    if not prepared:prepare()
    children=[];started=time.time()
    try:
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert max(map(int,used.splitlines()))<100,used
        for gpu in (0,1):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
                NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
            with (OUT/f'logs/gpu{gpu}.log').open('x') as log:
                children.append(subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(gpu)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT))
        shared.write(OUT/'status.json',dict(status='running',pid=os.getpid(),workers=[p.pid for p in children],started=started))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Full risk trace worker failed')
            time.sleep(3)
        assert all(p.returncode==0 for p in children)
        shared.write(OUT/'status.json',dict(status='complete',started=started,finished=time.time()))
        subprocess.run([sys.executable,str(REPO/'scripts/analyze_qwen3_full_risk_f.py'),str(OUT)],cwd=REPO,check=True)
    except BaseException as exc:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:p.wait()
        shared.write(OUT/'status.json',dict(status='failed',started=started,error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=int,choices=(0,1));parser.add_argument('--prepared',action='store_true')
    args=parser.parse_args();worker(args.worker) if args.worker is not None else run(args.prepared)
