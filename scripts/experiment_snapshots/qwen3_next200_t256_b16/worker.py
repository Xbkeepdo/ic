"""Generate two paired 256-token arms using the already verified controller."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from features.llava_decoding_evidence import lexical_nouns
from models import build_model
from scripts import generation_time_evidence_drift_qwen3_batched as shared
from scripts.generation_time_qwen3_layer_feedback import generate_group
from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
from utils.config_utils import get_extraction_model_cfg, load_config

NAMES=('sadi','sadi_sar_ems_next_19_20')

@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    ids=json.loads((args.output_root/'cohort.json').read_text())['coco200']
    if len(ids)!=200 or len(set(ids))!=200: raise ValueError('Expected 200 frozen unique images')
    protocol=json.loads((args.output_root/'protocol.json').read_text())
    if tuple(protocol['variants'])!=NAMES or protocol['max_new_tokens']!=256:
        raise ValueError('Unexpected frozen arm or token limit')
    folder=args.output_root/'coco200'
    folder.mkdir(parents=True,exist_ok=True)
    paths={name:folder/f'{name}.shard{args.shard}.jsonl' for name in NAMES}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Existing output requires paired batch audit before resume')
    config=dict(get_extraction_model_cfg(load_config(str(CONFIG)),MODEL))
    config['hf_name']=str(MODEL_ROOT)
    wrapper=build_model(MODEL,config,device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    nouns=lexical_nouns(wrapper.tokenizer,word_start='Ġ')
    if not nouns: raise RuntimeError('Empty entity vocabulary')
    existing={name:set() for name in NAMES}
    with patch.object(shared,'generate_group',generate_group):
        for span,group in shared.same_span_batches(wrapper,ids[args.shard::2],COCO_ROOT,16):
            # Explicit limit overrides the model wrapper's historical 512 default,
            # for baseline and feedback through the same generate_group API.
            shared.write_group(wrapper,span,group,nouns,existing,paths,256,args.shard)
    if any(len(done)!=100 for done in existing.values()): raise RuntimeError('Incomplete shard')
    print('NEXT200_T256_SHARD_COMPLETE',args.shard,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    run(parser.parse_args())
