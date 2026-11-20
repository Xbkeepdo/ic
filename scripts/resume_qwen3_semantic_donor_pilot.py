"""Resume complete batches with explicit collection of detached reader cycles."""

import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_semantic_donor_pilot as previous
from scripts import run_qwen3_signed_feedback as shared

SOURCE = previous.OUT
OUT = SOURCE.with_name(SOURCE.name+'_memory_recovery')


def prepare():
    assert not OUT.exists()
    old = shared.read(SOURCE/'protocol.json')
    assert shared.read(SOURCE/'run_status.json')['status'] == 'failed'
    hashes = dict(old['source_sha256'])
    for name, expected in hashes.items():
        assert shared.sha(Path(name)) == expected, name
    for path in [Path(__file__), REPO/'scripts/analyze_qwen3_semantic_donor_pilot.py',
                 SOURCE/'protocol.json', SOURCE/'run_status.json', *(SOURCE/'logs').glob('*.log')]:
        hashes[str(path)] = shared.sha(path)
    OUT.mkdir(); (OUT/'logs').mkdir(); (OUT/'coco100').mkdir()
    for name in ('cohort.json', 'object_lexicon.json'):
        shutil.copyfile(SOURCE/name, OUT/name)
    prefixes = {}
    for path in (SOURCE/'coco100').glob('*.jsonl'):
        groups = shared.source_groups(path)
        if any(path.name.startswith(n+'.') for n in old['candidates']):
            prefixes[path.name] = dict(groups=len(groups), images=sum(map(len, groups)), sha256=shared.sha(path))
        hashes[str(path)] = shared.sha(path)
        shutil.copyfile(path, OUT/'coco100'/path.name)
    plan = dict(old, source_sha256=hashes, shards=2, recovery_fixed_at=time.time(),
        failed_run=str(SOURCE), retained_completed_prefixes=prefixes,
        recovery='Identical math, batches/order/settings; collect unreachable owner-reader cycles and empty unused allocator blocks after every generation. Keep completed output bytes; no automatic rebatching.')
    shared.write(OUT/'protocol.json', plan)
    return plan


def worker(gpu):
    import json
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_semantic_donor_feedback import Qwen3BatchSemanticDonor
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    plan = shared.read(OUT/'protocol.json')
    for name, expected in plan['source_sha256'].items():
        assert shared.sha(Path(name)) == expected, name
    wrapper = previous.build_wrapper()
    allowed = frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    probe = FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    groups = shared.source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    paths = {name: OUT/f'coco100/{name}.shard{gpu}.jsonl' for name in previous.MODES}
    completed = {name: shared.source_groups(p) if p.exists() else [] for name, p in paths.items()}
    for name, done in completed.items():
        assert shared.group_keys(done) == shared.group_keys(groups[:len(done)])
    verified = False
    memory = []
    with torch.inference_mode():
        for index, group in enumerate(groups):
            pending = [n for n in paths if len(completed[n]) <= index]
            if not pending:
                continue
            ids = [r['image_id'] for r in group]
            (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [i for i, _ in inputs] == ids and span[1]-span[0] == group[0]['visual_token_count']
            if not verified:
                tokens, _, _ = original.generate_group(wrapper, span, inputs, allowed, 'sadi', 32)
                assert tokens == [r['response_token_ids'][:32] for r in group]
                shared.write(OUT/f'verification.gpu{gpu}.json', dict(passed=True, first_resumed_group=index,
                    image_ids=ids, exact_native_prefix_tokens=32, original_batch_size=len(ids)))
                gc.collect(); torch.cuda.empty_cache()
                verified = True
            static = {}
            for name in pending:
                policy = previous.MODES[name]
                def factory(model, current_span, nouns, mode, strength):
                    return Qwen3BatchSemanticDonor(model, current_span, allowed, probe=probe,
                        semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer, history_parser=parser,
                        transport_mode=policy, shared_static=static, real_threshold=plan['real_threshold'])
                torch.cuda.reset_peak_memory_stats(wrapper.device)
                before = torch.cuda.memory_allocated(wrapper.device)
                with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                    tokens, stats, seconds = original.generate_group(wrapper, span, inputs, allowed, shared.POSITIVE, 512)
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                with paths[name].open('a', encoding='utf-8') as stream:
                    for image, current, text, info in zip(ids, tokens, texts, stats, strict=True):
                        assert 0 < len(current) <= 512 and info['maximum_probe_hidden_difference'] == 0
                        stream.write(json.dumps(dict(image_id=image, model=MODEL, variant=name,
                            generated_text=text.strip(), response_token_ids=current, seconds=seconds/len(ids),
                            batch_size=len(ids), batch_wall_seconds=seconds, visual_token_count=span[1]-span[0],
                            sar_ems_statistics=info, decoder='native_qwen3_generate_same_span_batch'), ensure_ascii=False)+'\n')
                peak = torch.cuda.max_memory_allocated(wrapper.device)
                before_gc = torch.cuda.memory_allocated(wrapper.device)
                collected = gc.collect()
                torch.cuda.empty_cache()
                memory.append(dict(group=index, policy=policy, images=ids, before=before, before_gc=before_gc,
                    after_gc=torch.cuda.memory_allocated(wrapper.device), peak=peak, collected_objects=collected))
                shared.write(OUT/f'memory.gpu{gpu}.json', memory)
                print(f'[{gpu}] resumed {name} group={index} batch={len(ids)} seconds={seconds:.3f}', flush=True)
            del static
            gc.collect(); torch.cuda.empty_cache()
            memory[-1]['after_group_gc'] = torch.cuda.memory_allocated(wrapper.device)
            shared.write(OUT/f'memory.gpu{gpu}.json', memory)
    for name, path in paths.items():
        assert shared.group_keys(shared.source_groups(path)) == shared.group_keys(groups)
        origin = SOURCE/'coco100'/path.name
        if origin.exists():
            assert path.read_bytes().startswith(origin.read_bytes())


def run():
    plan = prepare()
    children = []
    started = time.time()
    try:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        assert len(used.splitlines()) == 2 and max(map(int, used.splitlines())) < 100
        for gpu in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       NLTK_DATA=str(shared.OUT/'generation_time_evidence_drift_llava/nltk_data'))
            with (OUT/f'logs/gpu{gpu}.log').open('w') as log:
                children.append(subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(gpu)],
                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        shared.write(OUT/'run_status.json', dict(status='generating', pid=os.getpid(), pids=[p.pid for p in children], started=started))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Memory recovery worker failed')
            time.sleep(3)
        assert all(p.returncode == 0 for p in children)
        shared.write(OUT/'run_status.json', dict(status='evaluating', started=started))
        shared.evaluate(OUT, plan)
        shared.write(OUT/'run_status.json', dict(status='complete', started=started, finished=time.time()))
    except BaseException as exc:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()
        shared.write(OUT/'run_status.json', dict(status='failed', error=repr(exc), started=started))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int, choices=(0, 1))
    args = p.parse_args()
    run() if args.worker is None else worker(args.worker)
