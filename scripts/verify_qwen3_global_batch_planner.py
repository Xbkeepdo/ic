"""Processor-only shape planning; reproduce old batches before future scheduling."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_ids(metadata, ids, maximum=16):
    pending, groups = {}, []
    for image in ids:
        row = metadata[image]; key = (*row['span'], row['sequence_length'])
        pending.setdefault(key, []).append(image)
        if len(pending[key]) == maximum:
            groups.append(pending.pop(key))
    groups.extend(pending.values())
    return groups


def assign_global_groups(metadata, ids):
    groups = group_ids(metadata, ids)
    groups.sort(key=lambda g: (-len(g), -metadata[g[0]]['visual_tokens'], g[0]))
    assigned, weights = [[], []], [0, 0]
    for group in groups:
        gpu = min(range(2), key=lambda i: (len(assigned[i]), weights[i], i))
        assigned[gpu].append(group)
        weights[gpu] += len(group)*metadata[group[0]]['sequence_length']
    assert abs(len(assigned[0])-len(assigned[1])) <= 1
    assert Counter(image for shard in assigned for group in shard for image in group) == Counter(ids)
    return assigned


def verify(root, output):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3_vl_wrapper import Qwen3VLWrapper
    from models.base_wrapper import configure_image_processor_limits
    from scripts.generation_time_evidence_drift_qwen3 import MODEL, MODEL_ROOT, CONFIG, COCO_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config
    assert not output.exists()
    assert read(root/'final_provenance.json')['passed']
    assert read(root/'protocol.json')['images'] == 100
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    cfg = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    cfg['hf_name'] = str(MODEL_ROOT)
    processor = AutoProcessor.from_pretrained(str(MODEL_ROOT), trust_remote_code=True, local_files_only=True)
    configure_image_processor_limits(processor, cfg)
    # The two unbound methods only use processor and the image-pad ID. No
    # wrapper constructor, model weights, or vision/text encoder is loaded.
    wrapper = SimpleNamespace(processor=processor,
        _image_token_id=int(processor.tokenizer.convert_tokens_to_ids(cfg.get('image_token', '<|image_pad|>'))))
    ids = read(root/'cohort.json')['coco100']; metadata = {}; started = time.perf_counter()
    for image in ids:
        path = COCO_ROOT/'val2014'/f'COCO_val2014_{image:012d}.jpg'
        with Image.open(path) as opened:
            inputs = Qwen3VLWrapper._prompt_inputs(wrapper, opened.convert('RGB'), 'Describe this image.')
        span = Qwen3VLWrapper._find_vision_token_range(wrapper, inputs['input_ids'][0])
        assert inputs['attention_mask'].all() and span[1] < inputs['input_ids'].shape[1]
        metadata[image] = dict(span=list(span), sequence_length=int(inputs['input_ids'].shape[1]),
            visual_tokens=span[1]-span[0], image_sha256=sha(path), image_grid=inputs['image_grid_thw'].tolist(),
            field_shapes={name:list(value.shape) for name,value in inputs.items()})
    seconds = time.perf_counter()-started
    recorded = []
    inputs_hash = {str(root/'protocol.json'): sha(root/'protocol.json')}
    for shard in (0, 1):
        path = root/f'coco100/sadi.shard{shard}.jsonl'; rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        inputs_hash[str(path)] = sha(path)
        groups, offset = [], 0
        while offset < len(rows):
            size = rows[offset]['batch_size']; group = rows[offset:offset+size]
            assert len(group) == size
            groups.append([r['image_id'] for r in group]); offset += size
            for row in group:
                assert row['visual_token_count'] == metadata[row['image_id']]['visual_tokens']
        assert groups == group_ids(metadata, ids[shard::2])
        recorded.extend(groups)
    assigned = assign_global_groups(metadata, ids)
    for shard in assigned:
        for group in shard:
            assert 1 <= len(group) <= 16
            assert len({(*metadata[i]['span'],metadata[i]['sequence_length']) for i in group}) == 1
            assert len({tuple(sorted(metadata[i]['field_shapes'])) for i in group}) == 1
    sources = [Path(__file__), REPO/'models/qwen3_vl_wrapper.py', REPO/'models/base_wrapper.py', CONFIG,
        Path(inspect.getfile(processor.__class__)), Path(inspect.getfile(processor.image_processor.__class__))]
    result = dict(passed=True, source_sha256={str(path):sha(path) for path in sources}, input_sha256=inputs_hash,
        cpu_preprocessing_seconds=seconds, images=100, original_stride2_batches=len(recorded),
        old_batch_boundaries_and_visual_counts_exact=True, global_batches=sum(map(len,assigned)),
        per_worker_global_batches=list(map(len,assigned)), global_plan=assigned, metadata=metadata,
        cuda_initialized=torch.cuda.is_initialized(), model_weights_or_encoders_loaded=False,
        scope='Shape/scheduling validation only, no generation or score comparison.',
        future_rule='Use a frozen global plan only when every compared method is generated fresh with those same groups. Retain existing old100 groups for cached controls.',
        audit_rule='A future globally grouped run needs an explicit group-manifest audit; do not reuse a stride2 membership assertion.')
    assert not result['cuda_initialized']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('passed','cpu_preprocessing_seconds','original_stride2_batches','global_batches','per_worker_global_batches','cuda_initialized','model_weights_or_encoders_loaded')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();verify(args.root,args.output)
