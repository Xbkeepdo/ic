"""Fixed-prefix mechanism test of SAR-selected late visual-memory rereading.

This is an exploratory test on two previously reviewed development images,
not new caption generation. Every trial is rolled back before continuing.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_qwen3_teacher_sadi_capture import read, write, sha


def prepare(root):
    import torch
    from features.qwen3_region_memory import rectangle, image_box, replace_region
    from scripts.run_qwen3_signed_feedback import source_groups
    assert not root.exists()
    source = REPO / 'outputs/qwen3_mass_matched_visual_pilot_20261119'
    diagnostic = REPO / 'outputs/qwen3_native_caption_gate_transfer_recovery_20261120'
    reserved = REPO / 'outputs/qwen3_mass_matched_visual_reserved500_20261119'
    assert read(source / 'final_provenance.json')['passed']
    assert read(diagnostic / 'analysis.json')['passed']
    assert not (reserved / 'consumed.json').exists()
    selected = [243882, 353411]
    positions = [r for r in read(diagnostic / 'predictions.json') if r['image_id'] in selected]
    groups = []
    for shard in [0, 1]:
        for batch in source_groups(source / f'coco100/sadi.shard{shard}.jsonl'):
            ids = [r['image_id'] for r in batch]
            if not set(ids) & set(selected): continue
            wanted = []
            for case in positions:
                if case['image_id'] not in ids: continue
                item = {key: case[key] for key in ['image_id', 'step', 'token_id', 'canonical', 'automatic_hallucination']}
                item.update(row=ids.index(case['image_id']), cached_p_real=case['scores']['reference'])
                item['reference_gate_positive'] = item['cached_p_real'] < .418672114610672
                wanted.append(item)
            assert wanted
            groups.append(dict(ids=ids, source_shard=shard,
                visual_tokens=batch[0]['visual_token_count'], cases=sorted(wanted, key=lambda r: (r['step'], r['row']))))
    assert set(r['image_id'] for g in groups for r in g['cases']) == set(selected)
    assert len(groups) == 2, 'Preserve and review actual groups rather than silently changing assignment.'
    # Geometry/row-preservation checks before expensive model work.
    for h, w in [(2, 4), (4, 5), (5, 4)]:
        for peak in [0, h*w//2, h*w-1]:
            q = torch.zeros(h*w); q[peak] = 1
            for mode in ['evidence', 'shifted']:
                rect = rectangle(q, h, w, mode); assert (rect[2]-rect[0])*(rect[3]-rect[1]) <= 9
                image_box(rect, (h, w), (640, 480))
    original = torch.arange(2*20*3, dtype=torch.bfloat16).reshape(40, 3)
    fine = torch.full((64, 3), 2., dtype=torch.bfloat16); rect = (1, 1, 4, 4)
    assert replace_region(original, fine, 1, (4, 5), (8, 8), rect, 0) is original
    modified = replace_region(original, fine, 1, (4, 5), (8, 8), rect, 1)
    assert torch.equal(modified[:20], original[:20]) and (modified[20:].view(4, 5, 3)[1:4, 1:4] == 2).all()
    files = [Path(__file__), REPO / 'features/qwen3_region_memory.py',
        *[REPO / name for name in ['features/qwen3_online_risk_f.py', 'features/qwen3_bounded_visual_tilt.py',
            'features/qwen3_acg_signal_feedback.py', 'features/qwen3_sar_ems_final_target.py',
            'features/qwen3_sar_ems_attention_batch.py', 'scripts/run_qwen3_semantic_final_pilot.py',
            'scripts/generation_time_evidence_drift_qwen3_batched.py', 'models/qwen3_vl_wrapper.py',
            'models/base_wrapper.py', 'configs/model_configs_coco4000_512_endac_baselines_v2.yaml']],
        source / 'protocol.json', source / 'cohort.json', source / 'final_provenance.json',
        diagnostic / 'protocol.json', diagnostic / 'analysis.json', diagnostic / 'predictions.json',
        reserved / 'cohort.json', reserved / 'reservation.json', reserved / 'batch_plan.json',
        *[source / f'coco100/sadi.shard{s}.jsonl' for s in [0, 1]]]
    hashes = dict(read(source / 'protocol.json')['source_sha256'])
    hashes.update({str(p): sha(p) for p in files})
    for filename, expected in hashes.items(): assert sha(REPO / filename) == expected, filename
    root.mkdir(); (root / 'crops').mkdir(); (root / 'source').mkdir()
    for path in files:
        if path.is_relative_to(REPO) and path.suffix in ['.py', '.yaml']:
            target = root / 'source' / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    write(root / 'protocol.json', dict(fixed_at=time.time(), source_sha256=hashes, source=str(source),
        selected_images=selected, groups=groups, shards=2, full_sadi_layers=list(range(5, 19)),
        source_layer=19, intervention_layers=list(range(20, 36)),
        components=['full_sar', 'semantic_increment'], location_controls=['evidence', 'shifted'],
        evidence='Q=normalize(write_norm*(1+log_s)) or normalize(write_norm*log_s); native layer19 BF16 writes, final-norm visual logits/full-vocabulary denominator.',
        ffn='P=softmax(cos(visual_hpre19,FFN_update19)/.03); D=clamp(1-sum(sqrt(QP)),0,1).',
        crop='3x3 merged-patch window around argmax Q, resized448x448. Control shifts peak by half grid height/width, with identical window, D and encoder settings. No GT boxes.',
        action='Bilinearly map crop image and all DeepStack features into original ROI, blend by D, reconstruct with cached original vision outputs and exact original full B/P prompt. Substitute only visual KV20--35 for one current-query replay.',
        rollback='All temporary visual and current KV restored after every trial; native forced trajectory continues. Zero reconstruction compares full36-layer prompt cache and final hidden exactly; first case per group hashes whole cache before/after all trials.',
        gate='Cached original108 real probability at fixed unchanged native prefix, tau .418672114610672; all selected slots receive forced mechanism trials, but only gate-positive ones are eligible online. No new detector claims.',
        scope='Two already reviewed development images, all their existing209-pool slots, no new captions, no ground-truth input to model/crop, no independent efficacy claim.',
        cost='Original vision encoder once per source group; crop features once per unique rectangle; cached image/deepstack reused for all prompt reconstructions; layer19 semantic readout only.',
        no_reserved_consumption=True, cpu_geometry_checks_passed=True))
    print(dict(prepared=True, groups=len(groups), cases=sum(len(g['cases']) for g in groups),
               identities=[[(r['image_id'], r['step'], r['canonical']) for r in g['cases']] for g in groups]), flush=True)


def worker(root, shard):
    import json
    import torch
    from PIL import Image
    from features.qwen3_region_memory import RegionEvidenceReader, RegionMemory, rectangle
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_signed_feedback import source_groups
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    plan = read(root / 'protocol.json')
    for filename, expected in plan['source_sha256'].items(): assert sha(REPO / filename) == expected, filename
    group = plan['groups'][shard]
    archived = {r['image_id']: r for s in [0, 1] for batch in source_groups(Path(plan['source']) / f'coco100/sadi.shard{s}.jsonl') for r in batch}
    wrapper = build_wrapper(); model = wrapper.model; core = model.model
    (span, items), = same_span_batches(wrapper, group['ids'], COCO_ROOT, len(group['ids']))
    assert [i for i, _ in items] == group['ids'] and span[1]-span[0] == group['visual_tokens']
    inputs = {key: torch.cat([item[key] for _, item in items], dim=0).to(wrapper.device) for key in items[0][1]}
    cases = {}; photos = {}
    for case in group['cases']: cases.setdefault(case['step'], []).append(case)
    for image in {case['image_id'] for case in group['cases']}:
        with Image.open(COCO_ROOT / 'val2014' / f'COCO_val2014_{image:012d}.jpg') as image_file: photos[image] = image_file.convert('RGB')
    reader = RegionEvidenceReader(model, span, sorted({r['token_id'] for r in group['cases']}))
    vision = []; encoder = [0]
    encoder_hook = core.visual.blocks[0].register_forward_hook(lambda *_: encoder.__setitem__(0, encoder[0]+1))
    capture_hook = core.visual.register_forward_hook(lambda _m, _a, output: vision.append(output))
    responses = [archived[image]['response_token_ids'] for image in group['ids']]
    prompt = inputs['input_ids'].shape[1]; core.rope_deltas = None; past = None; results = []
    started = time.time(); zero_checked = False

    def cache_digest(cache):
        digest = hashlib.sha256()
        for layer in cache.layers:
            for tensor in [layer.keys, layer.values]: digest.update(tensor.contiguous().view(torch.int16).cpu().numpy().tobytes())
        return digest.hexdigest()

    def top(logits):
        ids = logits.topk(5).indices.tolist()
        return dict(ids=ids, text=[wrapper.tokenizer.decode([i]) for i in ids])

    with torch.inference_mode(), Qwen3BatchLateAttention(model, span, frozenset(), 'sadi').capture(), reader.capture():
        for step in range(max(cases)+1):
            if step == 0: current = dict(inputs, cache_position=torch.arange(prompt, device=wrapper.device))
            else:
                previous = [r[step-1] if step <= len(r) else wrapper.tokenizer.eos_token_id for r in responses]
                current = dict(input_ids=torch.tensor(previous, device=wrapper.device)[:, None],
                    attention_mask=torch.ones((len(responses), prompt+step), device=wrapper.device, dtype=inputs['attention_mask'].dtype),
                    past_key_values=past, cache_position=torch.tensor([prompt+step-1], device=wrapper.device))
            output = core(**current, use_cache=True, return_dict=True); past = output.past_key_values
            assert reader.step == step
            if step == 0:
                assert len(vision) == 1; capture_hook.remove()
                memory = RegionMemory(wrapper, inputs, span, vision.pop(), output.last_hidden_state[:, -1:])
            if step not in cases: continue
            reference = output.last_hidden_state[:, -1:].clone()
            logits = model.get_output_embeddings()(reference).float()[:, 0]
            for case in cases[step]:
                row, token = case['row'], case['token_id']
                assert responses[row][step] == token and int(logits[row].argmax()) == token
                before_digest = cache_digest(past) if not zero_checked else None
                if not zero_checked:
                    zero = memory.keys(row, past, reader)
                    proposal = memory.replay(row, past, reader, zero, reference, zero=True)
                    assert torch.equal(model.get_output_embeddings()(proposal).float()[:, 0], logits)
                record = dict(case, native=top(logits[row]), native_logp=float(logits[row].log_softmax(0)[token]), trials=[])
                for component in plan['components']:
                    q, p, drift = reader.distributions(row, token, component)
                    grid = memory.grid(row)
                    for location in plan['location_controls']:
                        rect = rectangle(q, *grid, location)
                        destination = root / 'crops' / f"{case['image_id']}_{step}_{component}_{location}.jpg"
                        with reader.paused(): crop = memory.encode_crop(photos[case['image_id']], case['image_id'], row, rect, destination)
                        refined = memory.keys(row, past, reader, crop=crop, rect=rect, amount=drift)
                        hidden = memory.replay(row, past, reader, refined, reference)
                        proposed = model.get_output_embeddings()(hidden).float()[row, 0]
                        logp = float(proposed.log_softmax(0)[token]); proposed_token = int(proposed.argmax())
                        top_y = record['native']['ids']
                        record['trials'].append(dict(component=component, location=location, drift=drift,
                            rect=rect, grid=grid, pixel_box=crop['pixel_box'], crop_file=crop['image_file'],
                            crop_grid=crop['grid'], Q=q.tolist(), P=p.tolist(), top5=top(proposed),
                            original_logp=logp, original_logp_delta=logp-record['native_logp'],
                            token_changed=proposed_token != token,
                            would_accept=proposed_token != token and logp < record['native_logp'],
                            original_top5_logp=logits[row].log_softmax(0)[top_y].tolist(),
                            refined_original_top5_logp=proposed.log_softmax(0)[top_y].tolist()))
                if not zero_checked:
                    assert cache_digest(past) == before_digest
                    zero_checked = True
                results.append(record)
                print(dict(image=case['image_id'], step=step, native=record['native']['text'][0],
                    proposals=[(t['component'], t['location'], t['top5']['text'][0], t['original_logp_delta']) for t in record['trials']]), flush=True)
    encoder_hook.remove()
    assert len(results) == len(group['cases']) and encoder[0] == 1+memory.crop_encoder_calls
    for filename, expected in plan['source_sha256'].items(): assert sha(REPO / filename) == expected, filename
    write(root / f'results{shard}.json', results)
    write(root / f'worker{shard}.json', dict(passed=True, shard=shard, cases=len(results),
        original_encoder_calls=1, crop_encoder_calls=memory.crop_encoder_calls,
        actual_encoder_calls=encoder[0], cached_crop_hits=memory.vision_cache_hits,
        cached_vision_prompt_reconstructions=memory.reconstruction_calls,
        original_B_and_P=[len(group['ids']), prompt], full36_cache_restore_checked=zero_checked,
        seconds=time.time()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        no_new_captions=True, all_trials_rolled_back=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    try:
        if args.prepare: prepare(args.root)
        elif args.worker is not None: worker(args.root, args.worker)
        else: parser.error('Choose prepare or worker.')
    except Exception:
        if args.root.exists():
            target = args.root / (f'failure.worker{args.worker}.json' if args.worker is not None else 'failure.json')
            if not target.exists(): write(target, dict(failed_at=time.time(), traceback=traceback.format_exc()))
        raise
