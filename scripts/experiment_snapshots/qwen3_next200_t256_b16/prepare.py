"""Freeze the new 200-image, two-arm next-layer feedback experiment before generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

REPO = Path('/root/rivermind-data/ic-ac')
ROOT = REPO / 'outputs/generation_time_evidence_drift_qwen3_next200_t256_b16'
COCO = REPO.parent / 'dataset/coco'
SNAPSHOT = 'scripts/experiment_snapshots/qwen3_next200_t256_b16'
NAMES = ('sadi', 'sadi_sar_ems_next_19_20')
VERIFIED_ROOT = REPO / 'outputs/generation_time_evidence_drift_qwen3_layer_feedback100_b16'
SOURCE_FILES = (
    'features/qwen3_sar_ems_layer_feedback.py',
    'features/qwen3_sar_ems_attention.py',
    'features/qwen3_sar_ems_attention_batch.py',
    'features/llava_sar_ems_attention.py', 'features/llava_sadi.py',
    'features/llava_decoding_evidence.py',
    'scripts/generation_time_qwen3_layer_feedback.py',
    'scripts/verify_qwen3_layer_feedback.py',
    'scripts/generation_time_evidence_drift_qwen3_batched.py',
    'scripts/generation_time_evidence_drift_qwen3.py',
    'scripts/generation_time_evidence_drift_llava.py',
    'scripts/generation_time_sar_ems_seed43.py',
    'scripts/evaluate_generation_time_evidence_drift_llava.py', 'coco-labeling/coco_chair.py',
    'scripts/experiment_snapshots/qwen3_new200_b16/prepare.py',
    'models/qwen3_vl_wrapper.py',
    'configs/model_configs_coco4000_512_endac_baselines_v2.yaml',
    *(f'{SNAPSHOT}/{name}' for name in ('prepare.py', 'evaluate.py', 'worker.py', 'run.py', 'launch.py', 'README.md')),
)
sys.path.insert(0, str(REPO))
from scripts.experiment_snapshots.qwen3_new200_b16.prepare import cohort_ids, record_ids, digest_ids


def verification_reuse(source_sha):
    """Require the exact previously validated inference implementation and config."""
    verification_path = VERIFIED_ROOT / 'verification.json'
    protocol_path = VERIFIED_ROOT / 'protocol.json'
    data, protocol_data = verification_path.read_bytes(), protocol_path.read_bytes()
    verification, previous = json.loads(data), json.loads(protocol_data)
    if verification.get('passed') is not True:
        raise ValueError('Previous GPU verification did not pass')
    for mode in ('next', 'same'):
        for control in ('zero_strength', 'empty_nouns'):
            if verification['controls'][f'{mode}_{control}'].get('exact_baseline_logits_and_tokens') is not True:
                raise ValueError('Incomplete previous no-op verification')
    if sum(s['guided_attention_calls'] for s in verification['live']['next']['statistics']) <= 0:
        raise ValueError('Previous verification did not exercise live next-layer guidance')
    relevant = (
        'features/qwen3_sar_ems_layer_feedback.py', 'features/qwen3_sar_ems_attention.py',
        'features/qwen3_sar_ems_attention_batch.py', 'features/llava_sar_ems_attention.py',
        'features/llava_sadi.py', 'features/llava_decoding_evidence.py',
        'scripts/generation_time_qwen3_layer_feedback.py', 'scripts/verify_qwen3_layer_feedback.py',
        'scripts/generation_time_evidence_drift_qwen3_batched.py',
        'scripts/generation_time_evidence_drift_qwen3.py',
        'scripts/generation_time_sar_ems_seed43.py', 'models/qwen3_vl_wrapper.py',
        'configs/model_configs_coco4000_512_endac_baselines_v2.yaml',
    )
    for rel in relevant:
        if previous['source_sha256'].get(rel) != source_sha[rel]:
            raise ValueError(f'Inference source differs from validated version: {rel}')
    if previous['recipe']['next'] != [
        {'evidence_layer': 18, 'ffn_layer': 18, 'target_layer': 19},
        {'evidence_layer': 19, 'ffn_layer': 19, 'target_layer': 20},
    ] or previous['recipe']['strength'] != 5.0 or previous['recipe']['tau'] != 0.03:
        raise ValueError('Previous verification protocol does not match the unchanged next-layer recipe')
    return {'path': str(verification_path), 'sha256': hashlib.sha256(data).hexdigest(),
            'protocol_path': str(protocol_path), 'protocol_sha256': hashlib.sha256(protocol_data).hexdigest(),
            'matched_source_sha256': {rel: source_sha[rel] for rel in relevant},
            'gpu_verification_rerun': False,
            'scope': 'Reuses source/config-level inference verification; no claim of a full model-weight hash audit.'}


def main():
    if ROOT.exists():
        raise FileExistsError('Output root already exists; never resample or overwrite a frozen experiment')
    # Resolve every dependency before creating any manifest. Missing source files
    # must fail before sampling is published or a GPU worker can start.
    source_sha = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in SOURCE_FILES}
    reused_verification = verification_reuse(source_sha)
    all_ids = {int(path.stem.rsplit('_', 1)[-1]) for path in (COCO / 'val2014').glob('COCO_val2014_*.jpg')}
    sources, excluded = [], set()

    def add(path, kind):
        data = path.read_bytes()
        value = ([json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
                 if path.suffix == '.jsonl' else json.loads(data))
        if kind == 'coco4000':
            if not isinstance(value, dict) or not all(str(key).isdigit() for key in value):
                raise ValueError(f'Expected COCO generation keyed by numeric image IDs: {path}')
            ids = set(map(int, value))
            if len(ids) != 4000:
                raise ValueError(f'Expected COCO4000, found {len(ids)}')
        else:
            ids = cohort_ids(value) if kind == 'cohort' else record_ids(value)
        if not ids:
            raise ValueError(f'No image IDs found: {path}')
        excluded.update(ids)
        sources.append({'path': str(path), 'kind': kind,
                        'file_sha256': hashlib.sha256(data).hexdigest(),
                        'unique_ids': len(ids), 'ids_sha256': digest_ids(ids), 'ids': sorted(ids)})

    add(REPO.parent / 'ic/outputs/qwen3_vl_8b/COCO4000-512-ENDAC-BASELINES-V2/generations.json', 'coco4000')
    add(REPO.parent / 'dataset/qa_sam_svar_base/pope/questions.jsonl', 'pope')
    previous_new200 = REPO / 'outputs/generation_time_evidence_drift_qwen3_new200_b16/cohort.json'
    if not previous_new200.is_file():
        raise FileNotFoundError('Previous new200 cohort is required for the exclusion audit')
    if not (VERIFIED_ROOT / 'cohort.json').is_file():
        raise FileNotFoundError('Previous 100-image layer-feedback cohort is required for exclusion')
    for path in sorted((REPO / 'outputs').glob('generation_time_*/cohort.json')):
        if path.parent != ROOT:
            add(path, 'cohort')
    pool = sorted(all_ids - excluded)
    if len(pool) < 200:
        raise ValueError('Not enough unused COCO images')
    draw = np.random.default_rng(43).choice(pool, size=200, replace=False).tolist()
    selected = sorted(map(int, draw))
    if len(set(selected)) != 200 or set(selected) & excluded:
        raise ValueError('Invalid or overlapping sample')
    selected_sha = digest_ids(selected)
    selection = {
        'image_count': len(all_ids), 'excluded_available_images': len(all_ids & excluded),
        'candidate_images': len(pool), 'selected_images': 200, 'selected_sha256': selected_sha,
        'ids_hash_encoding': 'sorted JSON integer list, compact separators',
        'overlap_with_any_excluded_source': 0, 'excluded_ids': sorted(excluded), 'sources': sources,
    }
    protocol = {
        'model': 'Qwen3-VL-8B-Instruct', 'variants': list(NAMES), 'cohort': 'coco200',
        'prompt': 'Describe this image.', 'dtype': 'bfloat16', 'do_sample': False,
        'max_new_tokens': 256, 'batch_size_limit': 16,
        'batch_policy': 'same visual span and prompt length; retain smaller tail groups; common two-arm OOM split',
        'shards': 2, 'images_per_shard': 100,
        'generation_order': 'sorted image IDs, stride-2 shard assignment; identical ordered batches in all arms',
        'recipe': {
            'base': 'SADI', 'base_layers_inclusive': [5, 18], 'layer_numbering': 'zero-based',
            'target_layers': [19, 20], 'strength': 5.0, 'tau': 0.03,
            'gate': 'WordNet entity first-subtoken and positive FFN target gain',
            'next': [{'evidence_layer': 18, 'ffn_layer': 18, 'target_layer': 19},
                     {'evidence_layer': 19, 'ffn_layer': 19, 'target_layer': 20}],
            'budget': '5 * (1 - sum(sqrt(Q*P))) * tanh(gain/2) * relu(Q-P) / max(max(relu(Q-P)),1e-6)',
            'refresh': 'Fresh target and budget per source layer; no shared budget between target layers.',
        },
        'evaluation': {'primary': 'all 200 frozen images', 'bootstrap_draws': 5000, 'bootstrap_seed': 43,
                       'secondary': ['actual batch16 (preprocessing-defined)', 'common uncapped (output-conditioned)', 'largest single-image influence', 'denominator-growth top1/top3/top5 and canonical-object repetition']},
        'timing': 'Per-method generate time includes its prefill and controller caches; excludes model loading, CPU image preprocessing and verification.',
        'verification_reuse': reused_verification,
        'selected_sha256': selected_sha, 'source_sha256': source_sha,
    }
    payloads = {'cohort.json': {'coco200': selected},
                'sampling_order.json': {'seed': 43, 'rng': 'numpy default_rng / PCG64', 'draw_ids': draw},
                'selection.json': selection, 'protocol.json': protocol}
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{ROOT.name}.freeze-', dir=ROOT.parent))
    try:
        for name, payload in payloads.items():
            (staging / name).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        # Publish all four manifests together; ROOT must still be absent.
        if ROOT.exists():
            raise FileExistsError('Output root appeared during preparation')
        staging.rename(ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({k: v for k, v in selection.items() if k not in ('sources', 'excluded_ids')}, indent=2))
    print(f'FROZEN {ROOT}: {len(source_sha)} source hashes, {len(sources)} exclusion sources')


if __name__ == '__main__':
    main()
