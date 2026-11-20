"""Check a frozen four-feature exact-risk detector on native SADI captions."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_probe_first_class_pilot as base

OUT = shared.OUT / 'qwen3_risk_f_gate_trace_20261118'
INPUT = shared.OUT / 'qwen3_risk_f_gate_inputs_20261118'
SOURCE = base.OUT
FILES = ['features/qwen3_online_risk_f.py', 'scripts/verify_qwen3_online_risk_f.py',
         'scripts/trace_qwen3_risk_f_gate.py', 'scripts/analyze_qwen3_risk_f_gate.py']


def prepare():
    import numpy as np
    assert not OUT.exists()
    report = shared.read(INPUT / 'analysis.json')
    assert report['chosen'] == 'risk_F' and report['columns'] == [['risk',27],['F_log',27],['risk',28],['F_log',28]]
    compact = REPO.parent / 'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b/compact_features.npz'
    assert shared.sha(compact) == report['cache_sha256']
    with np.load(compact, allow_pickle=False) as data:
        old_images = set(data['image_id'].tolist())
    ids = shared.read(SOURCE / 'cohort.json')['coco100']
    assert len(ids) == 100 and not old_images.intersection(ids)
    hashes = {str(REPO / f): shared.sha(REPO / f) for f in FILES}
    for path in [*(REPO / f for f in shared.generation_sources()), compact,
            SOURCE / 'object_lexicon.json', SOURCE / 'protocol.json',
            REPO / 'features/coco_object_history.py', REPO / 'features/dgst_t.py',
            REPO / 'models/dgst_capture.py', REPO / 'scripts/run_qwen3_semantic_final_pilot.py',
            shared.OUT / 'qwen3_native_gate_coverage_20261118/positions.json',
            shared.OUT / 'qwen3_native_gate_coverage_20261118/coverage.json',
            REPO / 'coco-labeling/coco_chair.py', *(INPUT / f for f in ('gate.npz','analysis.json','cpu_verification.json'))]:
        hashes[str(path)] = shared.sha(path)
    cached = shared.read(SOURCE / 'protocol.json')['source_sha256']
    for f in shared.generation_sources():
        assert hashes[str(REPO / f)] == cached.get(str(REPO / f), cached.get(f)), f
    groups = []
    for shard in (0, 1):
        path = SOURCE / f'coco100/sadi.shard{shard}.jsonl'
        hashes[str(path)] = shared.sha(path)
        groups.append([dict(ids=[r['image_id'] for r in group], visual_tokens=group[0]['visual_token_count'])
                       for group in shared.source_groups(path)])
    OUT.mkdir()
    (OUT / 'logs').mkdir()
    (OUT / 'server_sources').mkdir()
    shutil.copyfile(SOURCE / 'object_lexicon.json', OUT / 'object_lexicon.json')
    shared.write(OUT / 'protocol.json', dict(source=str(SOURCE), groups=groups, ids=ids,
        images=100, max_new_tokens=512, max_batch=16, source_sha256=hashes,
        gate=str(INPUT / 'gate.npz'), threshold=report['threshold'], fixed_at=time.time(),
        purpose='Read-only native SADI trace; original exact K32/tau.03/w1 risk and F_log at27/28; no mitigation or refit.',
        eligibility='309 lexical token IDs, unseen token and canonical class in generated prefix; after prefill only.',
        prospective_followup='Only consider a mitigation run if the frozen threshold adds an image-confirmed unsupported object missed by old I/El, with automatic real positives<=14 (twice old7). No threshold tuning.',
        saved='Scalar4features/risk components, gate scores, target positions, timings. No full logits/hidden/attention tensors written.',
        precision='Native BF16 weighted values and output projection as original extractor; FP32 probability/cosine, CPU exact EMD; real-to-hall labels unchanged.'))
    for f in FILES:
        target = OUT / 'server_sources' / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / f, target)


def preflight(wrapper, nouns):
    import numpy as np
    import torch
    from features.qwen3_online_risk_f import OnlineRiskF
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from models.dgst_capture import apply_decoder_final_norm, target_probabilities_multi
    from scripts.generation_time_evidence_drift_qwen3 import MODEL, COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    from scripts.verify_qwen3_online_risk_f import reference_lp
    (span, group), = same_span_batches(wrapper, [485532,558955], COCO_ROOT, 16)
    inputs = {k: torch.cat([data[k] for _, data in group]).to(wrapper.device) for k in group[0][1]}
    targets = [wrapper.tokenizer.encode(' person', add_special_tokens=False)[0],
               wrapper.tokenizer.encode(' dog', add_special_tokens=False)[0]]
    def run(enabled):
        logits, tails, checks = [], [], []
        counts = dict(model=0, head=0, encoder=0)
        def count(name):
            def hook(*_):counts[name] += 1
            return hook
        def callback(reader, _value):
            if reader.step != 1:
                return
            features, details = reader.features([0,1], targets, reference_details=True)
            for row, record in enumerate(details):
                for layer, terms in record.items():
                    lp = reference_lp(terms['problem'], terms['penalties'])
                    risk = terms['transport']+terms['penalty']
                    assert abs(lp-risk) < 2e-6, (lp,risk)
                    normalized = apply_decoder_final_norm(reader.decoder.norm, terms['hpre'])
                    probs = target_probabilities_multi(output_layer=reader.head, states=normalized,
                        target_token_ids=[targets[row]], chunk_size=64)[:,0]
                    expected = torch.log1p(reader.head.weight.shape[0]*probs)
                    assert torch.equal(expected, terms['log_s'])
                    checks.append(dict(row=row, layer=int(layer), risk=features[row,0 if layer=='27' else 2].item(),
                        risk_joint_lp_error=abs(lp-risk), semantic_max_error=float((expected-terms['log_s']).abs().max())))
        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(wrapper.model, span, nouns, 'sadi').capture())
            reader = OnlineRiskF(wrapper.model, span, targets, callback=callback) if enabled else None
            if reader:stack.enter_context(reader.capture())
            for module, name in [(wrapper.model,'model'),(wrapper.model.get_output_embeddings(),'head'),(wrapper.model.model.visual,'encoder')]:
                stack.callback(module.register_forward_pre_hook(count(name)).remove)
            def observe(_m,_a,out):
                logits.append(out.logits[:, -1].cpu().clone())
                tails.append(tuple(t[:, :, -1:].cpu().clone() for layer in out.past_key_values.layers for t in (layer.keys,layer.values)))
            stack.callback(wrapper.model.register_forward_hook(observe).remove)
            tokens = wrapper.model.generate(**inputs, **dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=16)).cpu()
        return tokens, logits, tails, checks, counts, reader
    with torch.inference_mode():
        native, current = run(False), run(True)
    assert torch.equal(native[0],current[0])
    assert all(torch.equal(a,b) for a,b in zip(native[1],current[1],strict=True))
    assert all(torch.equal(a,b) for left,right in zip(native[2],current[2],strict=True) for a,b in zip(left,right,strict=True))
    assert len(current[3]) == 4 and current[4]['encoder'] == native[4]['encoder'] == 1
    assert current[4]['head'] == current[4]['model']+current[5].head_calls
    return dict(passed=True, exact_native_tokens_logits_current_KV_all36=True,
        reference_checks=current[3], counts=current[4], extra_semantic_head_calls=current[5].head_calls)


def worker(shard):
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_online_risk_f import OnlineRiskF, CompactRiskFGate
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import MODEL,COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches,trim_eos
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    plan = shared.read(OUT / 'protocol.json')
    for f, expected in plan['source_sha256'].items():assert shared.sha(Path(f)) == expected,f
    wrapper = build_wrapper()
    nouns = frozenset(shared.read(OUT / 'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py')
    classes = parser.token_classes(wrapper.tokenizer,nouns)
    gate = CompactRiskFGate(plan['gate'])
    shared.write(OUT / f'preflight.shard{shard}.json',preflight(wrapper,nouns))
    eos = wrapper.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    output = OUT / f'features.shard{shard}.jsonl'
    assert not output.exists()
    for index, group in enumerate(shared.source_groups(SOURCE / f'coco100/sadi.shard{shard}.jsonl')):
        ids = [r['image_id'] for r in group]
        (span, batch), = same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
        assert [i for i,_ in batch] == ids and span[1]-span[0] == group[0]['visual_token_count']
        inputs = {k:torch.cat([v[k] for _,v in batch]).to(wrapper.device) for k in batch[0][1]}
        records = []
        def callback(reader,value):
            targets = value.logits[:, -1].argmax(-1).tolist()
            selected = []
            for row,(target,cached) in enumerate(zip(targets,group,strict=True)):
                tokens = cached['response_token_ids']
                if reader.step >= len(tokens):continue
                assert target == tokens[reader.step],('Native trajectory changed',ids[row],reader.step)
                if reader.step == 0 or target not in nouns or target in tokens[:reader.step]:continue
                prefix = wrapper.tokenizer.decode(tokens[:reader.step],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                prior = parser.classes(prefix)
                if classes[target] not in prior:selected.append(row)
            if not selected:return
            features, details = reader.features(selected,[targets[r] for r in selected])
            probabilities = gate.predict(features)
            for row,vector,detail,probability in zip(selected,features,details,probabilities,strict=True):
                records.append(dict(image_id=ids[row],step=reader.step,token_id=targets[row],
                    canonical=classes[targets[row]],features=vector.tolist(),terms=detail,
                    p_hall=float(probability),positive=bool(probability>=gate.threshold)))
        reader = OnlineRiskF(wrapper.model,span,nouns,callback=callback)
        started = time.perf_counter()
        with torch.inference_mode(),Qwen3BatchLateAttention(wrapper.model,span,nouns,'sadi').capture(),reader.capture():
            result = wrapper.model.generate(**inputs,**dict(_generation_kwargs(wrapper,MODEL),max_new_tokens=512))
        torch.cuda.synchronize(wrapper.device)
        seconds = time.perf_counter()-started
        tokens = [trim_eos(row,eos) for row in result[:, inputs['input_ids'].shape[1]:].tolist()]
        assert tokens == [r['response_token_ids'] for r in group]
        with output.open('a') as stream:
            for r in records:stream.write(json.dumps(r)+'\n')
        shared.write(OUT / f'timing.shard{shard}.batch{index}.json',dict(image_ids=ids,span=list(span),
            exact_native_full_tokens=True,records=len(records),wall_seconds=seconds,
            feature_seconds=reader.feature_seconds,static_seconds=reader.static_seconds,
            semantic_head_calls=reader.head_calls,feature_calls=reader.feature_calls))
        print(f'[{shard}] batch={index} images={len(ids)} features={len(records)} seconds={seconds:.3f}',flush=True)
    shared.write(OUT / f'worker{shard}.json',dict(status='complete'))


def run(prepared=False):
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
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Risk trace worker failed')
            time.sleep(3)
        assert all(p.returncode==0 for p in children)
        shared.write(OUT/'status.json',dict(status='complete',started=started,finished=time.time()))
        subprocess.run([sys.executable,str(REPO/'scripts/analyze_qwen3_risk_f_gate.py'),str(OUT)],cwd=REPO,check=True)
    except BaseException as exc:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:p.wait()
        shared.write(OUT/'status.json',dict(status='failed',started=started,error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=int,choices=(0,1));parser.add_argument('--prepared',action='store_true')
    args=parser.parse_args()
    worker(args.worker) if args.worker is not None else run(args.prepared)
