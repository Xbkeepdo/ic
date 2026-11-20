"""Locate source-cache discrepancies without fitting or generating captions."""

import argparse
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read,write,sha


def run(root):
    import numpy as np
    import torch
    from PIL import Image
    from features.qwen3_teacher_sadi_capture import TeacherSadiCapture
    from models.dgst_capture import apply_decoder_final_norm
    from models.qwen3_vl_wrapper import _append_prefix_token_ids
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    assert not root.exists()
    cache=REPO/'outputs/qwen3_sadi_source_semantic_cache_20261119_v2'
    previous=REPO/'outputs/qwen3_teacher_sadi_preflight_20261119'
    assert read(previous/'verification.json')['passed'] is False
    vocabulary=read(cache/'verification.json')['vocabulary_size']
    train=REPO.parent/'ic/outputs/log_s_hpre_k32_tau003_w1_two_model_probe/qwen3_vl_8b'
    generation=Path(read(train/'protocol.json')['source']).parent/'generations.json'
    paths=[Path(__file__),REPO/'features/qwen3_teacher_sadi_capture.py',
           REPO/'scripts/verify_qwen3_teacher_sadi_capture.py',
           previous/'verification.json',cache/'protocol.json',cache/'verification.json',cache/'index.npz',cache/'log_s.npy',
           train/'compact_features.npz',generation]
    hashes={str(path):sha(path) for path in paths}
    image=read(cache/'protocol.json')['prototype_images'][0]
    root.mkdir();write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,
        image_id=image,selection='First source prototype fixed before the preceding preflight.',
        variants=['prompt_only','to_last_source_target','full_source_response'],
        intervention='None. Same image, prompt, BF16 eager model and target IDs.',
        purpose='Compare visual hpre and exact final-norm probabilities under different causal input lengths.',
        no_training=True,no_generated_captions=True,no_threshold_fitting=True))
    index=np.load(cache/'index.npz',allow_pickle=False)
    old=np.load(train/'compact_features.npz',allow_pickle=False)
    stored=np.load(cache/'log_s.npy',mmap_mode='r',allow_pickle=False)
    rows=np.flatnonzero(index['image_id']==image)
    response=read(generation)[str(image)]['response_token_ids']
    lengths=[0,int(index['response_token_idx'][rows].max()),len(response)]
    wrapper=build_wrapper();model=wrapper.model;core=model.model
    with Image.open(COCO_ROOT/'val2014'/f'COCO_val2014_{image:012d}.jpg') as photo:
        inputs=wrapper._prompt_inputs(photo.convert('RGB'),'Describe this image.').to(wrapper.device)
    prompt=inputs['input_ids'].shape[1];span=wrapper._find_vision_token_range(inputs['input_ids'][0])
    assert span[1]-span[0]==int(index['visual_tokens'][rows[0]])
    cached=np.stack([stored[index['offsets'][row]:index['offsets'][row+1]].reshape(36,span[1]-span[0]) for row in rows])
    tokens=torch.tensor(index['target_token_id'][rows],device=wrapper.device)
    results=[];arrays={};reference=None
    started=time.time()
    with torch.inference_mode():
        for name,length in zip(['prompt_only','to_last_source_target','full_source_response'],lengths,strict=True):
            current=_append_prefix_token_ids(inputs,prefix_token_ids=response[:length],device=wrapper.device)
            positions=[prompt+int(p)-1 for p in index['response_token_idx'][rows]] if length else [prompt-1]
            capture=TeacherSadiCapture(model,span,prompt,positions,sadi=False)
            core.rope_deltas=None
            with capture.context():
                output=core(**current,use_cache=False,return_dict=True)
            del output
            if reference is None:reference={layer:value.clone() for layer,value in capture.visual.items()}
            computed=[];prefix=[]
            for layer in range(36):
                value=capture.visual[layer];original=reference[layer]
                prefix.append(dict(layer=layer,exact=bool(torch.equal(value,original)),
                    max_absolute=float((value.float()-original.float()).abs().max()),
                    relative_l2=float((value.float()-original.float()).norm()/original.float().norm().clamp_min(1e-20))))
                normalized=apply_decoder_final_norm(core.language_model.norm,value)
                probability=[]
                for offset in range(0,len(value),64):
                    logits=model.get_output_embeddings()(normalized[offset:offset+64]).float()
                    probability.append((logits[:,tokens]-logits.logsumexp(-1,keepdim=True)).clamp_max(0).exp())
                computed.append(torch.cat(probability).cpu().numpy().T)
            probability=np.stack(computed,axis=1)
            logs=np.log1p(vocabulary*probability).astype(np.float32);semantic=probability.sum(-1)
            arrays[name+'_log_s']=logs;arrays[name+'_E_l']=semantic
            item=dict(name=name,response_prefix_tokens=length,prompt_length=prompt,
                visual_prefix_vs_prompt=prefix,
                cache_log_s_max_abs=float(np.max(np.abs(logs-cached))),
                cache_El_max_abs=float(np.max(np.abs(semantic-old['E_l'][rows]))))
            if length:
                features=capture.features(cached,old['E_l'][rows]);arrays[name+'_features']=features
                item.update(source_raw_I_max_relative=float(np.max(np.abs(features[:,36:72]-old['I'][rows])/np.maximum(np.abs(old['I'][rows]),1e-6))),
                    source_raw_risk_max_abs=float(np.max(np.abs(features[:,:36]-old['risk'][rows]))))
            results.append(item);write(root/(name+'.json'),item)
            print({k:v for k,v in item.items() if k!='visual_prefix_vs_prompt'},flush=True)
    arrays.update(row=rows,cached_log_s=cached,cached_E_l=old['E_l'][rows])
    np.savez_compressed(root/'arrays.npz',**arrays)
    for name,expected in hashes.items():assert sha(Path(name))==expected,name
    write(root/'analysis.json',dict(completed=True,source_hashes_verified=True,
        protocol_sha256=sha(root/'protocol.json'),arrays_sha256=sha(root/'arrays.npz'),
        image_id=image,results=results,seconds=time.time()-started,
        interpretation='Different lengths are a numerical-input-shape diagnosis, not three mitigation methods.',
        no_mitigation_claim=True,no_training=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
