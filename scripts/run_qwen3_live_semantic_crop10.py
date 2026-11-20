"""Complete-caption development screen of early semantic crop memory."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.probe_qwen3_decoupled_semantic_route import CACHE, PRIORS, check
from scripts.qwen3_caption_groups_with_cache import source_groups

PARENT = REPO/'outputs/qwen3_semantic_crop_route_probe_20261120'
CONTROLS = REPO/'outputs/qwen3_clva_signal_preflight10_v2_20261120'
GRAPH = REPO/'outputs/qwen3_relative_evidence_probe_20261120/head_class_neighbors.json'
MODES = ['semantic_crop', 'semantic_crop_route', 'shifted_crop']


def prepare(root):
    import numpy as np
    assert not root.exists()
    plan = dict(read(PARENT/'protocol.json')); check(plan)
    assert read(PARENT/'pipeline_complete.json')['passed'] and read(CONTROLS/'pipeline_complete.json')['passed']
    files = [Path(__file__), REPO/'features/qwen3_live_semantic_crop.py',
        REPO/'features/qwen3_decoupled_semantic_route.py',
        REPO/'scripts/run_qwen3_region_guidance.py', REPO/'scripts/evaluate_caption_chair.py',
        REPO/'features/coco_object_history.py', REPO/'coco-labeling/coco_chair.py',
        GRAPH, PRIORS, CONTROLS/'cohort.json', PARENT/'protocol.json', PARENT/'pipeline_complete.json']
    files += [CONTROLS/f'coco10/{name}.shard{s}.jsonl' for name in ['sadi', 'svar'] for s in [0, 1]]
    plan['source_sha256'].update({str(p): sha(p) for p in files})
    ids = [i for g in plan['groups'] for i in g['ids']]
    with np.load(PRIORS, allow_pickle=False) as d:
        assert not set(ids)&set(d['image_ids'].tolist())
    plan.update(fixed_at=time.time(), source=str(CONTROLS), cohort='coco10', images=10,
        candidates=MODES, variants=['sadi', 'svar', *MODES], copied_controls=['sadi', 'svar'], max_new_tokens=512,
        eligibility='All frozen309 object tokens; active rows; class absent from generated prefix; native source19 target gain>0; nearest1 head-class rival rho>max(1,rho_y). No learned detector, GT or hand-selected trigger positions.',
        action='Full SADI5--18. Early3 no-I positive rival difference selects3x3 crop448. Blend original image/DeepStack features by native D; replace visualKV20--35 only for current-query replay. Optional fixed19 attention routing; no FFN/logit editing.',
        acceptance='Greedy token changes and original-target logp decreases. Commit accepted final hidden and all20--35 currentKV after evaluating every row against original native cache; restore visual slots and history.',
        sharing='Exact same B/P groups share original vision, early3 class mass, source19 semantics, crops and refined prefixKV keyed by row/rectangle/exactD. Crop encoding once per unique rectangle. Old2 static matrices reused after ID/B/P/span checks.',
        preflight='GPU1 native128 matches copied SADI; zero/reject tokens/logits/current36/full36 KV exact; live crop commits checked by exact readback. No first-word flip is an efficacy claim.',
        scope='Complete10 known development screen; inspect actual object errors and recall/F1. Promising result advances to complete100, then independent500. Original WCOST detector unchanged; D is a Hellinger proxy.',
        development_screen='Vs SADI lower Cs/Ci and hallucinated count, F1 loss<=1.5pp and recall loss<=2pp, changes beyond one loop/denominator or unsupported object-word substitution.',
        timing='Warm shared-cache timings; not independent cold starts.')
    model = REPO.parent/'models/qwen3_vl_8b'
    plan['model_files'] = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in model.iterdir() if p.is_file() and p.suffix in ['.json', '.safetensors']}
    check(plan); root.mkdir(); (root/'coco10').mkdir(); (root/'crops').mkdir(); (root/'source').mkdir()
    for p in [Path(__file__), REPO/'features/qwen3_live_semantic_crop.py']:
        (root/'source'/p.name).write_bytes(p.read_bytes())
    (root/'cohort.json').write_bytes((CONTROLS/'cohort.json').read_bytes())
    assert set(read(root/'cohort.json')['coco10']) == set(ids)
    copied = []
    for name in ['sadi', 'svar']:
        for shard in [0, 1]:
            source = CONTROLS/f'coco10/{name}.shard{shard}.jsonl'
            target = root/f'coco10/{name}.shard{shard}.jsonl'; target.write_bytes(source.read_bytes())
            copied.append(dict(source=str(source), target=str(target), sha256=sha(source)))
    plan['copied_control_receipts'] = copied
    write(root/'protocol.json', plan)


def setup(root, shard):
    import numpy as np
    import torch
    from features.coco_object_history import ObjectClassHistory
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch
    plan = read(root/'protocol.json'); check(plan)
    for p, stat in plan['model_files'].items():
        assert Path(p).stat().st_size == stat['size'] and Path(p).stat().st_mtime_ns == stat['mtime_ns']
    wrapper = build_wrapper(); group = plan['groups'][shard]
    span, inputs, photos = batch(wrapper, group['ids'])
    with np.load(PRIORS, allow_pickle=False) as d:
        names = d['class_names'].tolist(); tokens = d['token_ids'].tolist(); mapping = d['token_classes'].tolist()
        prior = torch.from_numpy(d['class_prior'][d['layers'].tolist().index(3)].copy())
    graph = read(GRAPH); assert graph['classes'] == names and graph['tokens'] == tokens
    neighbors = {name: graph['neighbors'][name][0]['name'] for name in names}
    cached = {}
    for row, image in enumerate(group['ids']):
        path = CACHE/f'image{image}.npz'
        if not path.exists(): continue
        with np.load(path, allow_pickle=False) as d:
            assert d['token_ids'].tolist() == tokens and d['class_names'].tolist() == names
            assert d['batch_ids'].tolist() == group['ids'] and d['span'].tolist() == list(span)
            scaled = np.expm1(d['log_s'][3].astype(np.float64)); labels = np.array(mapping)
            mass = np.stack([scaled[:, labels == c].sum(1) for c in names], axis=1)
            cached[row] = dict(batch_ids=d['batch_ids'].tolist(), span=d['span'].tolist(),
                prompt=int(d['prompt_length']), mass3=mass, log_s19=torch.from_numpy(d['log_s'][19].copy()))
    args = (wrapper, span, group['ids'], photos, names, tokens, mapping, prior, neighbors,
        ObjectClassHistory(REPO/'coco-labeling/coco_chair.py'), root/'crops'/f'shard{shard}')
    return plan, wrapper, inputs, args, cached


def preflight(root):
    import torch
    from features.qwen3_live_semantic_crop import LiveSemanticCrop
    from scripts.run_qwen3_region_guidance import generate
    plan, wrapper, inputs, args, cached = setup(root, 1)
    shared = {}; reports = []; native = None; cap = 128
    with torch.inference_mode():
        for mode in ['native', 'zero', 'reject', 'live']:
            controller = LiveSemanticCrop(*args, shared=shared, cached=cached,
                disabled=mode == 'native', zero=mode == 'zero', reject=mode == 'reject')
            current = generate(wrapper, inputs, controller, cap, True)
            stats = controller.statistics(current[0]); events = [e for s in stats for e in s['full_events']]
            if mode == 'native':
                native = current
                records = {r['image_id']: r for g in source_groups(root/'coco10/sadi.shard1.jsonl') for r in g}
                assert current[0] == [records[i]['response_token_ids'][:cap] for i in plan['groups'][1]['ids']]
            if mode in ['zero', 'reject']:
                assert current[:3] == native[:3], mode
                assert events and not any(e['accepted'] for e in events), mode
            if mode == 'zero': assert all(e['current_kv_difference'] == 0 for e in events)
            if mode == 'reject':
                case, = [e for e in stats[0]['full_events'] if e['step'] == 97]
                assert case['token'] == 22435 and case['proposed_token'] == 25859
                assert abs(case['drift']-.3897426724433899) < 1e-7 and case['current_kv_difference'] > 0
            if mode == 'live':
                accepted = [e for e in events if e['accepted']]
                assert accepted and all(e['positive_commit_readback_exact'] for e in accepted)
            reports.append(dict(mode=mode, seconds=current[3], events=len(events), accepted=sum(e['accepted'] for e in events),
                native_exact=mode != 'live', stats=stats, tokens=current[0]))
            write(root/'preflight_progress.json', reports)
            print(dict(mode=mode, seconds=current[3], events=len(events), accepted=reports[-1]['accepted']), flush=True)
            controller.close(); del controller, current; gc.collect(); torch.cuda.empty_cache()
    check(plan)
    write(root/'preflight.json', dict(passed=True, cap=cap, positive_commit_tested=True,
        native_zero_reject_all36_kv_logits_tokens_exact=True, cases=reports, protocol_sha256=sha(root/'protocol.json')))


def worker(root, shard):
    import torch
    from features.qwen3_live_semantic_crop import LiveSemanticCrop
    from scripts.run_qwen3_region_guidance import generate
    plan, wrapper, inputs, args, cached = setup(root, shard)
    assert read(root/'preflight.json')['passed']
    paths = {mode: root/f'coco10/{mode}.shard{shard}.jsonl' for mode in MODES}
    assert not any(p.exists() for p in paths.values())
    shared = {}; progress = []; started = time.time()
    with torch.inference_mode():
        for mode in MODES:
            controller = LiveSemanticCrop(*args, policy=mode, shared=shared, cached=cached)
            tokens, _, _, seconds = generate(wrapper, inputs, controller, 512)
            stats = controller.statistics(tokens)
            texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            with paths[mode].open('x', encoding='utf-8') as f:
                for image, ids, text, record in zip(plan['groups'][shard]['ids'], tokens, texts, stats, strict=True):
                    f.write(json.dumps(dict(image_id=image, model='qwen3_vl_8b', variant=mode,
                        response_token_ids=ids, generated_text=text.strip(), batch_size=len(tokens),
                        visual_token_count=args[1][1]-args[1][0], batch_wall_seconds=seconds,
                        sar_ems_statistics=record, generation_reused=False), ensure_ascii=False)+'\n')
            progress.append(dict(mode=mode, seconds=seconds, images=len(tokens),
                accepted=sum(s['accepted'] for s in stats), proposals=sum(len(s['full_events']) for s in stats),
                encoder_calls=controller.encoder_calls, source19_head_calls=controller.reader.head_calls,
                early3_head_calls=controller.early_head_calls, peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            write(root/f'progress{shard}.json', progress); print(progress[-1], flush=True)
            controller.close(); del controller, tokens, stats; gc.collect(); torch.cuda.empty_cache()
    check(plan)
    for p, stat in plan['model_files'].items():
        assert Path(p).stat().st_size == stat['size'] and Path(p).stat().st_mtime_ns == stat['mtime_ns']
    write(root/f'worker{shard}.json', dict(passed=True, seconds=time.time()-started, progress=progress))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--preflight', action='store_true')
    p.add_argument('--worker', type=int, choices=[0, 1]); a = p.parse_args()
    try:
        if a.prepare: prepare(a.root)
        elif a.preflight: preflight(a.root)
        elif a.worker is not None: worker(a.root, a.worker)
        else: p.error('Select stage.')
    except BaseException:
        if a.root.exists(): write(a.root/f'failure_{"preflight" if a.preflight else a.worker}.json',
            dict(traceback=traceback.format_exc(), failed_at=time.time()))
        raise
