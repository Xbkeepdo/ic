"""Validate parallel SADI source tracing before any probe adaptation.

Only archived source captions are teacher-forced. This is not a caption
generation experiment and does not use the development or reserved labels.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def verify(root):
    import numpy as np
    import torch
    from PIL import Image
    from features.llava_sadi import sadi_visual_logits
    from features.qwen3_teacher_sadi_capture import TeacherSadiCapture, sadi_query_slice
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from models.dgst_capture import apply_decoder_final_norm
    from models.qwen3_vl_wrapper import _append_prefix_token_ids
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, CONFIG

    assert not root.exists(), 'Inspect an existing preflight rather than overwrite it.'
    source = REPO / 'outputs/qwen3_sadi_source_semantic_cache_20261119_v2'
    provenance = read(source / 'verification.json')
    assert provenance['passed'] and provenance['original_feature_and_target_identity_verified']
    source_plan = read(source / 'protocol.json')
    train = REPO.parent / 'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'
    # Reuse the exact original108 model directory already recorded by the
    # consensus diagnosis. Do not discover a checkpoint by best test score.
    consensus = read(REPO / 'outputs/qwen3_semantic_consensus_gate_transfer_20261119/protocol.json')
    checkpoint = train/'risk+I+E_l/seed43'
    assert str(checkpoint/'model.pt') in consensus['source_sha256']
    assert sha(checkpoint/'model.pt')==consensus['source_sha256'][str(checkpoint/'model.pt')]
    paths = [Path(__file__), REPO/'features/qwen3_teacher_sadi_capture.py',
             REPO/'features/llava_sadi.py', REPO/'features/llava_decoding_evidence.py',
             REPO/'features/qwen3_sar_ems_attention_batch.py',
             REPO/'features/qwen3_online_risk_f.py', REPO/'features/dgst_t.py',
             REPO/'features/qwen3_risk_i_el_probe.py', REPO/'features/qwen3_online_i_el.py',
             REPO/'models/qwen3_vl_wrapper.py', REPO/'models/base_wrapper.py',
             REPO/'models/dgst_capture.py', REPO/'utils/config_utils.py', CONFIG,
             REPO/'scripts/run_qwen3_semantic_final_pilot.py',
             source/'protocol.json', source/'verification.json', source/'index.npz', source/'log_s.npy',
             train/'protocol.json', train/'compact_features.npz',
             checkpoint/'model.pt', checkpoint/'input_scaler.npz']
    generations = Path(read(train/'protocol.json')['source']).parent/'generations.json'
    paths.append(generations)
    hashes = {str(path): sha(path) for path in paths}
    root.mkdir()
    plan = dict(fixed_at=time.time(), source_sha256=hashes,
        source_cache=str(source), probe=str(checkpoint), images=source_plan['prototype_images'],
        selection=source_plan['prototype_selection'], layers=list(range(5,19)),
        no_new_captions=True, no_probe_training=True, no_current_or_reserved_labels=True,
        parallel='All prediction queries from last prompt position onward; earlier prompt and image queries unchanged.',
        sequential='Unchanged one-last-query SADI, real KV cache, forced archived source response IDs.',
        arithmetic='BF16 eager. Equal-shape collector controls require exact equality; parallel/sequential shape differences are explicitly measured.',
        limits=dict(parallel_risk_absolute=.02, parallel_I_relative=.01,
                    parallel_probe_absolute=.025, cache_log_s_absolute=.03,
                    cache_El_absolute_floor=2e-6, cache_El_relative=.02),
        cache_validation_layers=[0,18,35],
        future='Only a passing preflight permits source-domain feature extraction. No mitigation claim from this check.')
    write(root/'protocol.json',plan)
    torch.set_num_threads(1)
    torch.manual_seed(1701)
    # Different batches, heads, sequence lengths and a nonzero query offset.
    for dtype in (torch.float32,torch.bfloat16):
        values=torch.randn(2,4,11,19).to(dtype)
        actual=sadi_query_slice(values,2,8,5)
        expected=values.clone()
        for batch in range(2):
            for query in range(5,11):
                expected[batch,:,query,2:8]=sadi_visual_logits(values[batch:batch+1,:,query,2:8])[0]
        assert torch.equal(actual,expected)
        assert torch.equal(values[:,:,:5],actual[:,:,:5])
        assert torch.equal(values[:,:,:,8:],actual[:,:,:,8:])
    write(root/'cpu_check.json',dict(passed=True,exact_scalar_query_replay=True,
        untouched_earlier_and_nonvisual_entries=True,no_cuda_initialized=not torch.cuda.is_initialized()))
    old=np.load(train/'compact_features.npz',allow_pickle=False)
    index=np.load(source/'index.npz',allow_pickle=False)
    logs=np.load(source/'log_s.npy',mmap_mode='r',allow_pickle=False)
    captions=read(generations)
    wrapper=build_wrapper();model=wrapper.model;core=model.model
    probe=FullRiskIElProbe(checkpoint).eval()
    results=[];all_rows=[];all_parallel=[];all_sequential=[];all_raw=[]
    encoder_calls=[0]
    hook=model.model.visual.register_forward_hook(lambda *_:encoder_calls.__setitem__(0,encoder_calls[0]+1))
    started=time.time()

    def run_core(inputs,context=None,cache=False):
        core.rope_deltas=None
        with context if context is not None else nullcontext():
            return core(**inputs,use_cache=cache,return_dict=True)

    def sequential(inputs,response,stop,span,capture=None):
        core.rope_deltas=None
        length=inputs['input_ids'].shape[1];past=None
        features={};hidden=[]
        with capture.context() if capture is not None else Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture():
            for step in range(stop+1):
                if step==0:
                    current=dict(inputs);current['cache_position']=torch.arange(length,device=wrapper.device)
                else:
                    current=dict(input_ids=torch.tensor([[response[step-1]]],device=wrapper.device),
                        attention_mask=torch.ones((1,length+step),device=wrapper.device,dtype=inputs['attention_mask'].dtype),
                        cache_position=torch.tensor([length+step-1],device=wrapper.device),past_key_values=past)
                output=core(**current,use_cache=True,return_dict=True);past=output.past_key_values
                hidden.append(output.last_hidden_state[:,-1].cpu().clone())
                if capture is not None and step in wanted:
                    position=wanted[step]
                    features[step]=capture.features(raw_logs[position:position+1],semantics[position:position+1])[0]
        del past,output
        return hidden,features

    with torch.inference_mode():
        for image_number,image in enumerate(plan['images']):
            rows=np.flatnonzero(index['image_id']==image)
            assert rows.size and len(set(index['visual_tokens'][rows]))==1
            response=captions[str(image)]['response_token_ids']
            positions=index['response_token_idx'][rows].astype(int).tolist()
            assert len(set(positions))==len(positions)
            wanted={position:i for i,position in enumerate(positions)}
            stop=max(positions)
            path=COCO_ROOT/'val2014'/f'COCO_val2014_{image:012d}.jpg'
            with Image.open(path) as photo:
                inputs=wrapper._prompt_inputs(photo.convert('RGB'),'Describe this image.').to(wrapper.device)
            prompt=inputs['input_ids'].shape[1];span=wrapper._find_vision_token_range(inputs['input_ids'][0])
            assert span[1]-span[0]==int(index['visual_tokens'][rows[0]]),(image,span,int(index['visual_tokens'][rows[0]]))
            assert inputs['attention_mask'].all()
            sequence=_append_prefix_token_ids(inputs,prefix_token_ids=response[:stop],device=wrapper.device)
            absolute=[prompt+p-1 for p in positions]
            raw_logs=np.stack([logs[index['offsets'][row]:index['offsets'][row+1]].reshape(36,span[1]-span[0]) for row in rows])
            semantics=old['E_l'][rows]
            before=encoder_calls[0]
            parallel=TeacherSadiCapture(model,span,prompt,absolute)
            result=run_core(sequence,parallel.context())
            parallel_hidden=result.last_hidden_state.cpu();del result
            parallel_features=parallel.features(raw_logs,semantics)
            if image_number==0:
                disabled=TeacherSadiCapture(model,span,prompt,absolute,capture=False)
                result=run_core(sequence,disabled.context())
                assert torch.equal(parallel_hidden,result.last_hidden_state.cpu())
                del result
            raw=TeacherSadiCapture(model,span,prompt,absolute,sadi=False)
            result=run_core(sequence,raw.context())
            raw_hidden=result.last_hidden_state.cpu();del result
            raw_features=raw.features(raw_logs,semantics)
            if image_number==0:
                result=run_core(sequence)
                assert torch.equal(raw_hidden,result.last_hidden_state.cpu());del result
            # SADI changes only later causal queries, so visual hpre is exactly
            # identical between raw and SADI for the same input tensor shape.
            assert all(torch.equal(raw.visual[layer],parallel.visual[layer]) for layer in range(36))
            log_error=semantic_error=0.;semantic_ok=True
            target_ids=torch.tensor(index['target_token_id'][rows],device=wrapper.device)
            for layer in plan['cache_validation_layers']:
                visual=apply_decoder_final_norm(core.language_model.norm,raw.visual[layer])
                probabilities=[]
                for offset in range(0,len(visual),64):
                    logits=model.get_output_embeddings()(visual[offset:offset+64]).float()
                    probabilities.append((logits[:,target_ids]-logits.logsumexp(-1,keepdim=True)).clamp_max(0).exp())
                probability=torch.cat(probabilities).cpu().numpy().T
                log_error=max(log_error,float(np.max(np.abs(np.log1p(provenance['vocabulary_size']*probability)-raw_logs[:,layer]))))
                calculated=probability.sum(-1)
                semantic_error=max(semantic_error,float(np.max(np.abs(calculated-semantics[:,layer]))))
                semantic_ok=semantic_ok and bool(np.allclose(calculated,semantics[:,layer],atol=2e-6,rtol=.02))
            del raw,parallel,visual,logits,probability
            sequential_capture=TeacherSadiCapture(model,span,prompt,(),mode='last')
            sequential_hidden,by_position=sequential(inputs,response,stop,span,sequential_capture)
            sequential_features=np.stack([by_position[p] for p in positions])
            exact_native=True
            if image_number==0:
                native_hidden,_=sequential(inputs,response,stop,span)
                exact_native=all(torch.equal(a,b) for a,b in zip(sequential_hidden,native_hidden,strict=True))
                assert exact_native,'Last-query collector differs from unchanged native SADI.'
            risk_error=float(np.max(np.abs(parallel_features[:,:36]-sequential_features[:,:36])))
            intensity_relative=float(np.max(np.abs(parallel_features[:,36:72]-sequential_features[:,36:72])/np.maximum(np.abs(sequential_features[:,36:72]),1e-6)))
            a=probe(torch.from_numpy(parallel_features)).numpy();b=probe(torch.from_numpy(sequential_features)).numpy()
            probe_error=float(np.max(np.abs(a-b)))
            selected_hidden=torch.stack([parallel_hidden[0,prompt+p-1] for p in positions])
            reference_hidden=torch.cat([sequential_hidden[p] for p in positions])
            hidden_error=float((selected_hidden.float()-reference_hidden.float()).abs().max())
            passed=bool(risk_error<=.02 and intensity_relative<=.01 and probe_error<=.025 and log_error<=.03 and semantic_ok)
            item=dict(image_id=image,targets=len(rows),response_last_index=stop,prompt_length=prompt,visual_tokens=span[1]-span[0],
                parallel_risk_max_abs=risk_error,parallel_I_max_relative=intensity_relative,
                parallel_probe_max_abs=probe_error,parallel_final_hidden_max_abs=hidden_error,
                cache_log_s_max_abs=log_error,cache_El_max_abs=semantic_error,cache_El_within_limit=semantic_ok,
                source_raw_I_max_relative=float(np.max(np.abs(raw_features[:,36:72]-old['I'][rows])/np.maximum(np.abs(old['I'][rows]),1e-6))),
                source_raw_risk_max_abs=float(np.max(np.abs(raw_features[:,:36]-old['risk'][rows]))),
                exact_native_controller=exact_native if image_number==0 else 'checked_first_fixed_image',
                exact_raw_and_SADI_visual_prefix=True,encoder_calls=encoder_calls[0]-before,
                passed=passed,seconds=time.time()-started)
            results.append(item);all_rows.extend(map(int,rows))
            all_parallel.extend(parallel_features);all_sequential.extend(sequential_features);all_raw.extend(raw_features)
            write(root/f'image{image}.json',item);print(json.dumps(item),flush=True)
            del sequential_capture,sequential_hidden,parallel_hidden,raw_hidden
    hook.remove()
    np.savez_compressed(root/'features.npz',row=np.asarray(all_rows),parallel=np.asarray(all_parallel),
                        sequential=np.asarray(all_sequential),raw=np.asarray(all_raw))
    for name,expected in hashes.items():assert sha(Path(name))==expected,name
    result=dict(passed=all(row['passed'] for row in results),images=len(results),targets=len(all_rows),
        protocol_sha256=sha(root/'protocol.json'),features_sha256=sha(root/'features.npz'),results=results,
        exact_scalar_query_cpu=True,exact_equal_shape_raw_and_parallel_collector_controls=True,
        source_labels_not_used=True,maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.time()-started,no_mitigation_claim=True,
        followup='Source adaptation permitted only if passed; failure requires investigation rather than relaxing limits.')
    write(root/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args()
    try:
        verify(args.root)
    except Exception:
        if args.root.exists() and not (args.root/'failure.json').exists():
            write(args.root/'failure.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
