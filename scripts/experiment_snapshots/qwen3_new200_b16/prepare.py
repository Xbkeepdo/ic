"""Freeze 200 new COCO images before generation, excluding prior caption cohorts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

REPO=Path('/root/rivermind-data/ic-ac')
ROOT=REPO/'outputs/generation_time_evidence_drift_qwen3_new200_b16'
COCO=REPO.parent/'dataset/coco'
sys.path.insert(0,str(REPO))
from data.qa_benchmark import atomic_write_json

def digest_ids(ids):
    return hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()

def record_ids(value):
    result=set()
    if isinstance(value,dict):
        if 'image_id' in value:
            result.add(int(value['image_id']))
        for key in ('image','image_path','file_name'):
            text=value.get(key)
            if isinstance(text,str):
                match=re.search(r'(?:COCO_val2014_)?(\d+)\.(?:jpg|jpeg|png)$',text)
                if match: result.add(int(match.group(1)))
        for child in value.values():
            if isinstance(child,(dict,list)): result.update(record_ids(child))
    elif isinstance(value,list):
        for child in value: result.update(record_ids(child))
    return result

def cohort_ids(value):
    if isinstance(value,list):
        if all(isinstance(item,int) or (isinstance(item,str) and item.isdigit()) for item in value):
            return set(map(int,value))
        result=set()
        for child in value: result.update(cohort_ids(child))
        return result
    if isinstance(value,dict):
        result=set()
        for child in value.values(): result.update(cohort_ids(child))
        return result
    # Cohort manifests also contain scalar metadata such as seed and description.
    return set()

def main():
    if (ROOT/'cohort.json').exists():
        raise FileExistsError('Frozen cohort already exists; reuse it without resampling')
    all_ids={int(path.stem.rsplit('_',1)[-1]) for path in (COCO/'val2014').glob('COCO_val2014_*.jpg')}
    sources=[]
    excluded=set()
    def add(path,kind):
        data=path.read_bytes()
        value=[json.loads(line) for line in data.decode().splitlines() if line.strip()] if path.suffix=='.jsonl' else json.loads(data)
        if kind=='coco4000':
            if not isinstance(value,dict) or not all(str(key).isdigit() for key in value):
                raise ValueError('Expected generation JSON keyed by numeric COCO image ID')
            ids=set(map(int,value))
        else:
            ids=cohort_ids(value) if kind=='cohort' else record_ids(value)
        if not ids: raise ValueError(f'No image IDs parsed from {path}')
        if kind=='coco4000' and len(ids)!=4000: raise ValueError(f'Expected COCO4000, found {len(ids)}')
        excluded.update(ids)
        sources.append({'path':str(path),'kind':kind,'file_sha256':hashlib.sha256(data).hexdigest(),'unique_ids':len(ids),'ids_sha256':digest_ids(ids),'ids':sorted(ids)})
    add(REPO.parent/'ic/outputs/qwen3_vl_8b/COCO4000-512-ENDAC-BASELINES-V2/generations.json','coco4000')
    add(REPO.parent/'dataset/qa_sam_svar_base/pope/questions.jsonl','pope')
    for path in sorted((REPO/'outputs').glob('generation_time_*/cohort.json')):
        if path.parent!=ROOT: add(path,'cohort')
    pool=sorted(all_ids-excluded)
    draw=np.random.default_rng(43).choice(pool,size=200,replace=False).tolist()
    selected=sorted(map(int,draw))
    assert len(set(selected))==200 and not set(selected)&excluded
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_write_json(ROOT/'cohort.json',{'coco200':selected})
    atomic_write_json(ROOT/'sampling_order.json',{'seed':43,'rng':'numpy default_rng / PCG64','draw_ids':draw})
    selection={'image_count':len(all_ids),'excluded_available_images':len(all_ids&excluded),'candidate_images':len(pool),'selected_images':200,'selected_sha256':digest_ids(selected),'ids_hash_encoding':'sorted JSON integer list, compact separators','overlap_with_any_excluded_source':0,'excluded_ids':sorted(excluded),'sources':sources}
    atomic_write_json(ROOT/'selection.json',selection)
    files=['features/qwen3_sar_ems_attention.py','features/qwen3_sar_ems_attention_batch.py','features/llava_sar_ems_attention.py','features/llava_decoding_evidence.py','scripts/generation_time_sar_ems_seed43.py','scripts/generation_time_evidence_drift_qwen3_batched.py','models/qwen3_vl_wrapper.py','configs/model_configs_coco4000_512_endac_baselines_v2.yaml']
    protocol={'model':'Qwen3-VL-8B-Instruct','variants':['svar','svar_sar_ems_attention'],'prompt':'Describe this image.','dtype':'bfloat16','do_sample':False,'max_new_tokens':512,'batch_size_limit':16,'batch_policy':'same visual span and prompt length; retain smaller tail groups; paired OOM split is recorded','shards':2,'images_per_shard':100,'generation_order':'sorted image IDs, stride-2 shard assignment','recipe':{'base_layers':[5,18],'svar_strength':0.5,'write_layer':18,'ffn_layer':20,'late_layers':[21,22],'strength':5,'tau':0.03},'selected_sha256':digest_ids(selected),'source_sha256':{rel:hashlib.sha256((REPO/rel).read_bytes()).hexdigest() for rel in files}}
    atomic_write_json(ROOT/'protocol.json',protocol)
    print(json.dumps({k:v for k,v in selection.items() if k not in ('sources','excluded_ids')},ensure_ascii=False))
    print(json.dumps([{'kind':s['kind'],'path':s['path'],'unique_ids':s['unique_ids']} for s in sources],ensure_ascii=False))

if __name__=='__main__': main()
