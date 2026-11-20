"""Add missing timing metadata in derived inputs; never rewrite generations."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha

VARIANTS = ['sadi', 'svar', 'semantic_crop', 'semantic_crop_route', 'shifted_crop']


def prepare(root):
    assert read(root/'generation_complete.json')['passed']
    assert (root/'driver_failure.json').exists() and not (root/'evaluation_recovery_request.json').exists()
    paths = [root/f'coco10/{name}.shard{s}.jsonl' for name in VARIANTS for s in [0, 1]]
    request = dict(fixed_at=time.time(), raw_sha256={str(p): sha(p) for p in paths},
        source_sha256={str(p): sha(p) for p in [Path(__file__), REPO/'scripts/evaluate_caption_chair.py',
            REPO/'scripts/evaluate_generation_time_evidence_drift_llava.py', root/'protocol.json', root/'cohort.json']},
        correction='Only derived candidate inputs receive seconds=batch_wall_seconds/batch_size; text, tokens, methods and all existing fields unchanged. Original caption files immutable.',
        evaluators=['sentence', 'legacy'], variants=VARIANTS)
    write(root/'evaluation_recovery_request.json', request)
    stage = root/'evaluation_seconds_inputs'; assert not stage.exists()
    stage.mkdir(); (stage/'coco10').mkdir(); shutil.copyfile(root/'cohort.json', stage/'cohort.json')
    rows = []
    for source in paths:
        target = stage/'coco10'/source.name
        original = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        corrected = []
        for item in original:
            derived = dict(item)
            if 'seconds' not in derived:
                derived['seconds'] = derived['batch_wall_seconds']/derived['batch_size']
                restored = dict(derived); del restored['seconds']; assert restored == item
            else: assert derived == item
            assert derived['generated_text'] == item['generated_text']
            assert derived['response_token_ids'] == item['response_token_ids']
            corrected.append(derived)
        if all('seconds' in item for item in original): shutil.copyfile(source, target)
        else:
            target.write_text(''.join(json.dumps(item, ensure_ascii=False)+'\n' for item in corrected), encoding='utf-8')
        rows.append(dict(raw=str(source), raw_sha256=sha(source), derived=str(target),
            derived_sha256=sha(target), images=len(original), text_and_token_ids_unchanged=True))
    write(root/'evaluation_derived_inputs.json', dict(passed=True, files=rows))
    print('RECOVERY_INPUTS_PREPARED', flush=True)


def run(root):
    request = read(root/'evaluation_recovery_request.json')
    assert not (root/'evaluation_recovery_v2_request.json').exists()
    write(root/'evaluation_recovery_v2_request.json',dict(fixed_at=time.time(),source_sha256=sha(Path(__file__)),parent_request_sha256=sha(root/'evaluation_recovery_request.json'),derived_inputs_sha256=sha(root/'evaluation_derived_inputs.json'),correction='Reuse unchanged derived inputs; preserve evaluator provenance when pre-seeding GT cache.'))
    for name, wanted in {**request['raw_sha256'], **request['source_sha256']}.items(): assert sha(Path(name)) == wanted
    assert read(root/'evaluation_derived_inputs.json')['passed']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NLTK_DATA=str(REPO/'outputs/generation_time_evidence_drift_llava/nltk_data'))
    results = []
    for version in request['evaluators']:
        destination = root/f'evaluations/recovery_v2_{version}_v1'; assert not destination.exists()
        old = root/'evaluations/sentence_v1' if version == 'sentence' else REPO/'outputs/qwen3_live_visual_ffn_contrast10_20261120/evaluations/legacy_v1'
        cache_copies = []
        if old.exists() and (old/'evaluator.json').is_file():
            destination.mkdir(parents=True)
            shutil.copyfile(old/'evaluator.json',destination/'evaluator.json')
            for source in old.glob('chair_ground_truth.*.json'):
                target = destination/source.name; shutil.copyfile(source, target)
                cache_copies.append(dict(source=str(source), sha256=sha(source)))
        with (root/f'evaluation_recovery_v2_{version}.log').open('xb') as log:
            result = subprocess.run([sys.executable, '-u', 'scripts/evaluate_caption_chair.py',
                '--evaluator', version, '--generation-root', str(root/'evaluation_seconds_inputs'),
                '--evaluation-root', str(destination), '--cohort', 'coco10', '--variants', *VARIANTS],
                cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        results.append(dict(version=version, returncode=result.returncode, reused_gt_caches=cache_copies))
        assert result.returncode == 0, results
        print('RECOVERY_EVALUATED', version, flush=True)
    for name, wanted in request['raw_sha256'].items(): assert sha(Path(name)) == wanted
    write(root/'evaluation_recovery_v2_complete.json', dict(passed=True, results=results,
        original_captions_unchanged=True, completed_at=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path)
    p.add_argument('--prepare', action='store_true'); a = p.parse_args()
    try: prepare(a.root) if a.prepare else run(a.root)
    except BaseException:
        import traceback
        write(a.root/'evaluation_recovery_v2_failure.json', dict(traceback=traceback.format_exc(), failed_at=time.time()))
        raise
