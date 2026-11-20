"""Cache native36-layer selected309 semantics with frozen-prefix checks."""

import argparse
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha

SOURCE=REPO/'outputs/qwen3_supported_rivals_probe_20261120'
COARSE=REPO/'outputs/qwen3_region_rivals_probe_20261120'


def prepare(root):
    assert not root.exists() and read(SOURCE/'pipeline_complete.json')['passed']
    old=read(SOURCE/'protocol.json');hashes=dict(old['source_sha256'])
    files=[Path(__file__),SOURCE/'protocol.json',REPO/'features/qwen3_native_risk_i_el.py',
        REPO/'features/coco_object_history.py',REPO/'scripts/run_qwen3_region_guidance.py']
    files.extend(SOURCE.glob('features108.*.npz'));files.extend(COARSE.glob('visual19.*.npz'))
    hashes.update({str(p):sha(p) for p in files})
    for n,v in hashes.items():assert sha(REPO/n)==v,n
    root.mkdir();(root/'source').mkdir();(root/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source_sha256=hashes,groups=old['groups'],
        scope='Two previously used image prefixes, original full B/P and ordered groups; no generation or new quality evaluation.',
        selected_tokens=309,layers=36,base='Unchanged full project SADI5--18',
        save='Raw probabilities, log_s and exact E_l for selected309 columns only; class alias sums. No hidden, full vocabulary logits or attention arrays.',
        proof='All matching saved108 candidate E_l columns must be bit-exact; whole19-layer roundtrip probabilities bit-exact to the previous cache.',
        use='Reusable CPU class-level cross-layer comparison before layer or mitigation selection. No claim of general layer usefulness from two images.'))


def worker(root,shard):
    import numpy as np
    import torch
    from features.qwen3_native_risk_i_el import NativeRiskIEl
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch
    plan=read(root/'protocol.json')
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    group=plan['groups'][shard];wrapper=build_wrapper();model=wrapper.model
    allowed=read(REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119/object_lexicon.json')['token_ids']
    classes=ObjectClassHistory(REPO/'coco-labeling/coco_chair.py').token_classes(wrapper.tokenizer,allowed)
    names=sorted(set(classes.values()));span,inputs,_=batch(wrapper,group['ids'])
    reader=NativeRiskIEl(model,span,allowed);encoder=[]
    handle=model.model.visual.blocks[0].register_forward_hook(lambda *_:encoder.append(1))
    records=[];started=time.time()
    with torch.inference_mode(),Qwen3BatchLateAttention(model,span,frozenset(),'sadi').capture(),reader.capture():
        out=model(**inputs,use_cache=True,return_dict=True,logits_to_keep=1)
        assert reader.step==0 and len(encoder)==1
        for row in sorted({c['row'] for c in group['cases']}):
            image=group['ids'][row];probabilities=[];log_s=[];el=[];mass=[];roundtrip_mass=[]
            columns={c:[reader.columns[t] for t in reader.tokens if classes[t]==c] for c in names}
            for layer in range(36):
                entry=reader._static(layer,row)
                probabilities.append(entry['probabilities'].cpu().numpy());log_s.append(entry['log_s'].cpu().numpy())
                el.append(entry['E_l'].cpu().numpy())
                mass.append([float(entry['probabilities'][:,cols].sum(-1).sum()) for cols in columns.values()])
                roundtrip_mass.append([float((entry['log_s'][:,cols].expm1()/reader.head.weight.shape[0]).sum(-1).sum()) for cols in columns.values()])
            el=np.stack(el);checks=0
            for path in sorted(SOURCE.glob(f'features108.{image}.*.npz')):
                with np.load(path) as previous:
                    index=[reader.columns[int(t)] for t in previous['tokens']]
                    assert np.array_equal(el[:,index].T,previous['features'][:,72:]),str(path)
                    checks+=len(index)*36
            with np.load(COARSE/f'visual19.shard{shard}.row{row}.npz') as previous:
                actual=(reader._static(19,row)['log_s'].expm1()/reader.head.weight.shape[0]).cpu().numpy()
                assert np.array_equal(actual,previous['probabilities'])
                assert previous['batch_ids'].tolist()==group['ids'] and previous['prompt']==inputs['input_ids'].shape[1]
            path=root/f'image{image}.npz'
            np.savez_compressed(path,probabilities=np.stack(probabilities),log_s=np.stack(log_s),E_l=el,
                class_E_l=np.asarray(mass,dtype=np.float32),class_E_l_roundtrip=np.asarray(roundtrip_mass,dtype=np.float32),
                token_ids=np.asarray(reader.tokens,dtype=np.int64),class_names=np.asarray(names),
                token_classes=np.asarray([classes[t] for t in reader.tokens]),image_id=image,batch_ids=np.asarray(group['ids']),
                span=np.asarray(span),prompt_length=inputs['input_ids'].shape[1])
            records.append(dict(image_id=image,rows=36,visual_tokens=span[1]-span[0],columns=len(reader.tokens),classes=len(names),
                exact_El_cells_verified=checks,whole_layer19_cache_exact=True,bytes=path.stat().st_size,sha256=sha(path)))
            print(records[-1],flush=True)
    handle.remove();assert len(encoder)==1
    for n,v in plan['source_sha256'].items():assert sha(REPO/n)==v,n
    write(root/f'worker{shard}.json',dict(passed=True,records=records,encoder_calls=1,seconds=time.time()-started,
        selected_head_calls=reader.head_calls,no_generation=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:prepare(a.root) if a.prepare else worker(a.root,a.worker)
    except BaseException:
        if a.root.exists():write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
