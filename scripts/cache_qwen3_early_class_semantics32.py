"""Cache early visual class evidence with one native prefill per original B16."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.probe_qwen3_signal_erasure_contrast import check

SOURCE=REPO/'outputs/qwen3_risk_plan_breadth32_20261120'
CLASS_CACHE=REPO/'outputs/qwen3_visual_semantics_cache_20261120/image353411.npz'
LAYERS=[3,4]


def prepare(root):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    assert not root.exists();old=read(SOURCE/'protocol.json');check(old)
    assert read(SOURCE/'pipeline_complete.json')['passed']
    config=Path('/root/rivermind-data/models/qwen3_vl_8b/config.json')
    assert read(config)['vision_config']['deepstack_visual_indexes']==[8,16,24]
    text=Path(qwen.__file__).read_text()
    assert 'layer_idx in range(len(deepstack_visual_embeds))'in text and 'deepstack_visual_embeds[layer_idx]'in text
    files=[Path(__file__),Path(qwen.__file__),config,CLASS_CACHE,SOURCE/'protocol.json',SOURCE/'pipeline_complete.json']
    files += [SOURCE/f'{name}{s}{ext}'for s in [0,1]for name,ext in [('semantic19.shard','.npz'),('worker','.json')]]
    plan=dict(old);plan['source_sha256']={**old['source_sha256'],**{str(p):sha(p)for p in files}}
    plan.update(fixed_at=time.time(),previous=str(SOURCE),layers=LAYERS,
        selection='Same27 images with existing19-layer cache inside original32 B16 groups; no label or score selection.',
        architecture='Vision blocks8/16/24 outputs injected after text layers0/1/2; hpre3 is after all three injections. hpre4 is adjacent control,19 cached reference. Indexes0-based.',
        motivation='Early maps may retain different object evidence than19. Old2-image layer comparisons already existed; this is exploratory, not a blind layer choice or independent validation.',
        compute='Exactly one full original B16 native SADI prefill per GPU. No teacher continuation or tail proposals. Read309 final-norm probabilities at3/4 with original64-position chunks; sum62 lexical classes on CPU.',
        reuse='Do not reproject layer19. Verify every selected visual_hpre19 hash against previous cache and model file fingerprints; retain old19 cache as reference.',
        saved='Only Float64 class mass=sum_alias expm1(log_s)=V*sum_alias p, layers3/4,27images. No hidden/attention/full-vocabulary outputs.',
        scope='Readout diagnostic only, no new captions/CHAIR/F1/500. Next-stage analysis compares all3/4/19, not best-layer-only.')
    check(plan);root.mkdir();(root/'source').mkdir()
    (root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',plan)


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from models.dgst_capture import apply_decoder_final_norm
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch,digest
    plan=read(root/'protocol.json');check(plan)
    for name,stat in plan['model_files'].items():assert Path(name).stat().st_size==stat['size']and Path(name).stat().st_mtime_ns==stat['mtime_ns']
    with np.load(CLASS_CACHE,allow_pickle=False)as d:
        tokens=d['token_ids'].copy();classes=d['class_names'].copy();mapping=d['token_classes'].copy()
    columns=[np.flatnonzero(mapping==c)for c in classes]
    with np.load(SOURCE/f'semantic19.shard{shard}.npz',allow_pickle=False)as d:
        ids=d['batch_ids'].tolist();rows=d['rows'].tolist();image_ids=d['image_ids'].copy()
        old_span=d['span'].tolist();old_prompt=int(d['prompt']);old_hashes=d['hpre_sha256'].tolist()
        assert np.array_equal(tokens,d['token_ids'])
    assert ids==plan['groups'][shard]['ids'];wrapper=build_wrapper();model=wrapper.model;core=model.model
    span,inputs,_=batch(wrapper,ids);assert list(span)==old_span and inputs['input_ids'].shape[1]==old_prompt
    captured={};encoder=[];core.rope_deltas=None;started=time.time()
    def capture(index):
        def hook(_module,args,kwargs):
            value=args[0]if args else kwargs['hidden_states']
            assert value.ndim==3 and value.shape[0]==16 and value.shape[1]==old_prompt
            assert index not in captured
            captured[index]=value[rows,span[0]:span[1]].detach().clone()
        return hook
    with torch.inference_mode(),ExitStack()as stack:
        stack.enter_context(Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture())
        for index in [3,4,19]:stack.callback(model.model.language_model.layers[index].register_forward_pre_hook(capture(index),with_kwargs=True).remove)
        stack.callback(core.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1)).remove)
        output=core(**inputs,use_cache=True,return_dict=True)
        assert len(encoder)==1 and set(captured)=={3,4,19}
        assert all(output.past_key_values.get_seq_length(i)==old_prompt for i in range(36))
        for index,wanted in enumerate(old_hashes):assert digest([captured[19][index]])==wanted
        del output
    assert len(encoder)==1
    head=model.get_output_embeddings();norm=model.model.language_model.norm;head_calls=0;class_values=[];hpre_hashes=[]
    with torch.inference_mode():
        selected=torch.tensor(tokens,device=wrapper.device)
        for layer in LAYERS:
            values=[];hashes=[]
            for row in range(len(rows)):
                hidden=captured[layer][row];hashes.append(digest([hidden]));normalized=apply_decoder_final_norm(norm,hidden)
                pieces=[]
                for offset in range(0,len(hidden),64):
                    logits=head(normalized[offset:offset+64]).float();probability=(logits[:,selected]-logits.logsumexp(-1,keepdim=True)).clamp_max(0).exp()
                    pieces.append(torch.log1p(head.weight.shape[0]*probability).cpu().numpy());head_calls+=1
                log_s=np.concatenate(pieces).astype(np.float64);scaled=np.expm1(log_s)
                values.append(np.stack([scaled[:,c].sum(1)for c in columns],axis=1))
            class_values.append(np.stack(values));hpre_hashes.append(hashes)
    mass=np.stack(class_values);assert np.isfinite(mass).all()and(mass>=0).all()
    path=root/f'classes34.shard{shard}.npz'
    np.savez_compressed(path,class_mass=mass,layers=np.array(LAYERS),class_names=classes,token_ids=tokens,token_classes=mapping,
        image_ids=image_ids,rows=np.array(rows),batch_ids=np.array(ids),span=np.array(span),prompt=old_prompt,
        hpre_sha256=np.array(hpre_hashes),source19_hpre_sha256=np.array(old_hashes),vocabulary_size=head.weight.shape[0],
        source19_cache_sha256=sha(SOURCE/f'semantic19.shard{shard}.npz'),protocol_sha256=sha(root/'protocol.json'))
    check(plan)
    for name,stat in plan['model_files'].items():assert Path(name).stat().st_size==stat['size']and Path(name).stat().st_mtime_ns==stat['mtime_ns']
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,images=len(rows),full_batch=16,
        native_prefills=1,encoder_calls=len(encoder),original19_hpre_all_exact=True,source19_head_calls=0,
        new_head_calls=head_calls,layers=LAYERS,vocabulary_size=head.weight.shape[0],class_shape=list(mass.shape),
        raw_class_bytes=mass.nbytes,compressed_bytes=path.stat().st_size,cache_sha256=sha(path),
        teacher_continuation_steps=0,new_caption_images=0,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1]);a=parser.parse_args()
    try:
        if a.prepare:prepare(a.root)
        elif a.worker is not None:worker(a.root,a.worker)
        else:parser.error('Select prepare or worker.')
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
