"""Freeze processor-only prospective batches without consuming a reserved test."""

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_qwen3_global_batch_planner import read, sha, group_ids, assign_global_groups


def plan(root):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3_vl_wrapper import Qwen3VLWrapper
    from models.base_wrapper import configure_image_processor_limits
    from scripts.generation_time_evidence_drift_qwen3 import MODEL, MODEL_ROOT, CONFIG, COCO_ROOT
    from utils.config_utils import get_extraction_model_cfg, load_config
    from scripts.run_qwen3_block_acg_fresh500 import ids_sha

    output = root/'batch_plan.json'
    assert not output.exists() and not (root/'consumed.json').exists()
    reservation = read(root/'reservation.json'); ids = read(root/'cohort.json')['coco500']
    assert reservation['reserve_only'] and not reservation['neural_generation_started']
    assert ids_sha(ids) == reservation['sample_sha256'] and len(set(ids)) == 500
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    cfg = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    cfg['hf_name'] = str(MODEL_ROOT)
    processor = AutoProcessor.from_pretrained(str(MODEL_ROOT), trust_remote_code=True, local_files_only=True)
    configure_image_processor_limits(processor, cfg)
    wrapper = SimpleNamespace(processor=processor,
        _image_token_id=int(processor.tokenizer.convert_tokens_to_ids(cfg.get('image_token', '<|image_pad|>'))))
    started = time.perf_counter(); metadata = {}
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
    assigned = assign_global_groups(metadata, ids)
    assert Counter(image for shard in assigned for group in shard for image in group) == Counter(ids)
    for shard in assigned:
        for group in shard:
            assert 1 <= len(group) <= 16
            assert len({(*metadata[i]['span'], metadata[i]['sequence_length']) for i in group}) == 1
            assert len({tuple(sorted(metadata[i]['field_shapes'])) for i in group}) == 1
    sources = [Path(__file__), REPO/'scripts/verify_qwen3_global_batch_planner.py',
        REPO/'models/qwen3_vl_wrapper.py', REPO/'models/base_wrapper.py', CONFIG,
        Path(inspect.getfile(processor.__class__)), Path(inspect.getfile(processor.image_processor.__class__))]
    result = dict(passed=True, planned_at=time.time(), sample_sha256=ids_sha(ids), images=500,
        max_batch=16, prompt='Describe this image.', global_plan=assigned, metadata=metadata,
        global_batches=sum(map(len, assigned)), per_worker_global_batches=list(map(len, assigned)),
        old_stride2_batches=sum(len(group_ids(metadata, ids[s::2])) for s in (0, 1)),
        cpu_preprocessing_seconds=seconds, cuda_initialized=torch.cuda.is_initialized(),
        model_weights_or_encoders_loaded=False, generation_started=False,
        input_sha256={str(root/n):sha(root/n) for n in ('cohort.json', 'reservation.json', 'selection.json')},
        source_sha256={str(path):sha(path) for path in sources},
        rule='Prospective test only. Generate every method fresh with this identical ordered manifest. No use for cached old100 controls. This does not consume or authorize neural validation.',
        audit_rule='Verify exact per-worker ordered groups against this manifest, not stride2 membership.')
    assert not result['cuda_initialized']
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps({k:result[k] for k in ('passed', 'images', 'global_batches', 'per_worker_global_batches',
        'old_stride2_batches', 'cpu_preprocessing_seconds', 'generation_started')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    plan(parser.parse_args().root)
