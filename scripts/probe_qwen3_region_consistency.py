"""Test finer-source consistency and FFN-guided region attenuation on fixed prefixes."""

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
    assert not root.exists()
    previous = REPO / 'outputs/qwen3_region_memory_probe_20261120'
    assert read(previous / 'pipeline_complete.json')['passed']
    old = read(previous / 'protocol.json')
    for filename, expected in old['source_sha256'].items(): assert sha(REPO / filename) == expected, filename
    files = [Path(__file__), REPO / 'features/qwen3_region_consistency.py',
        REPO / 'features/qwen3_region_memory.py', REPO / 'features/coco_object_history.py',
        previous / 'protocol.json', previous / 'pipeline_complete.json',
        *[previous / f'{stem}{s}.json' for stem in ['results', 'worker'] for s in [0, 1]],
        Path(old['source']) / 'object_lexicon.json']
    hashes = dict(old['source_sha256']); hashes.update({str(p): sha(p) for p in files})
    root.mkdir(); (root / 'crops').mkdir(); (root / 'source').mkdir()
    for p in files:
        if p.suffix == '.py' and p.is_relative_to(REPO):
            dest = root / 'source' / p.relative_to(REPO); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    plan = dict(old, source_sha256=hashes, fixed_at=time.time(), previous=str(previous),
        locators=['full_sar', 'semantic_peak'], policies=['verified', 'unverified', 'shifted'],
        location='Full SAR argmax, or argmax of summed logit-lens probabilities of the current canonical-class aliases; fixed3x3 window, no GT boxes.',
        verification='Fine crop is448x448; inspect layer19 visual hpre at the same ROI coordinates after original-shape cached-vision prefill with full ROI replacement. Sum current-class alias probabilities, averaged over ROI.',
        reliability_deficit='r=relu(p_coarse-p_fine)/(p_coarse+p_fine), with denominator floor1e-30; [0,1].',
        action='Native visual attention values retained. Penalize ROI logits by5*D*r at20--35, then exactly renormalize visual logsumexp before full softmax. FFN changes only D, never its activation.',
        controls='Unverified uses5*D; shifted uses the same verified5*D*r budget on a half-grid-shifted ROI, sharing the evidence verification (no separate control crop).',
        rollback='All trials restore visual/current KV. Zero strength calls native attention exactly. Existing original-prefix36-layer reconstruction and fresh zero-tail/full-cache checks retained.',
        scope='Same6 previously reviewed development slots; forced mechanism trials at all6, online eligibility still cached original108 gate. No new captions or independent effect claim.',
        no_reproduction_claim='New project region-suppression adaptation with crop consistency and SAR/FFN guidance.')
    write(root / 'protocol.json', plan)
    print(dict(prepared=True, cases=6, locators=plan['locators'], policies=plan['policies']), flush=True)


def worker(root, shard):
    import torch
    from PIL import Image
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_region_memory import RegionEvidenceReader, rectangle
    from features.qwen3_region_consistency import RegionConsistency, category_probabilities
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
    allowed = read(Path(plan['source']) / 'object_lexicon.json')['token_ids']
    classes = ObjectClassHistory(REPO / 'coco-labeling/coco_chair.py').token_classes(wrapper.tokenizer, allowed)
    (span, items), = same_span_batches(wrapper, group['ids'], COCO_ROOT, len(group['ids']))
    assert [i for i, _ in items] == group['ids'] and span[1]-span[0] == group['visual_tokens']
    inputs = {key: torch.cat([item[key] for _, item in items], dim=0).to(wrapper.device) for key in items[0][1]}
    cases = {}; photos = {}
    for case in group['cases']: cases.setdefault(case['step'], []).append(case)
    for image in {case['image_id'] for case in group['cases']}:
        with Image.open(COCO_ROOT / 'val2014' / f'COCO_val2014_{image:012d}.jpg') as f: photos[image] = f.convert('RGB')
    reader = RegionEvidenceReader(model, span, allowed)
    vision = []; encoder = [0]
    encoder_hook = core.visual.blocks[0].register_forward_hook(lambda *_: encoder.__setitem__(0, encoder[0]+1))
    capture_hook = core.visual.register_forward_hook(lambda _m, _a, output: vision.append(output))
    responses = [archived[image]['response_token_ids'] for image in group['ids']]
    prompt = inputs['input_ids'].shape[1]; core.rope_deltas = None; past = None
    results = []; zero_checked = False; started = time.time()
    old_cases = {(r['image_id'], r['step']): r for r in read(Path(plan['previous']) / f'results{shard}.json')}

    def cache_digest(cache):
        h = hashlib.sha256()
        for layer in cache.layers:
            for tensor in [layer.keys, layer.values]: h.update(tensor.contiguous().view(torch.int16).cpu().numpy().tobytes())
        return h.hexdigest()

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
                memory = RegionConsistency(wrapper, inputs, span, vision.pop(), output.last_hidden_state[:, -1:])
            if step not in cases: continue
            reference = output.last_hidden_state[:, -1:].clone()
            logits = model.get_output_embeddings()(reference).float()[:, 0]
            for case in cases[step]:
                row, token = case['row'], case['token_id']
                assert responses[row][step] == token and int(logits[row].argmax()) == token
                aliases = [t for t in allowed if classes[t] == classes[token]]
                before_digest = cache_digest(past) if not zero_checked else None
                if not zero_checked:
                    zero_inspection = memory.inspect(row, past, reader, None, (0, 0, 3, 3), aliases, amount=0.)
                    assert zero_inspection['reliability_deficit'] == 0
                    hidden, _ = memory.attenuate(row, past, reader, (0, 0, 3, 3), 0., reference, zero=True)
                    assert torch.equal(model.get_output_embeddings()(hidden).float()[:, 0], logits)
                q, p, drift = reader.distributions(row, token)
                prior_q = old_cases[case['image_id'], step]['trials'][0]['Q']
                assert float((q-torch.tensor(prior_q, device=q.device)).abs().max()) < 1e-6
                semantic = category_probabilities(reader, row, aliases)
                record = dict(case, native=top(logits[row]), native_logp=float(logits[row].log_softmax(0)[token]), trials=[])
                for locator in plan['locators']:
                    heat = q if locator == 'full_sar' else semantic
                    rect = rectangle(heat, *memory.grid(row), 'evidence')
                    dest = root / 'crops' / f"{case['image_id']}_{step}_{locator}.jpg"
                    with reader.paused(): crop = memory.encode_crop(photos[case['image_id']], case['image_id'], row, rect, dest)
                    inspection = memory.inspect(row, past, reader, crop, rect, aliases)
                    for policy in plan['policies']:
                        strength = 5*drift*(1. if policy == 'unverified' else inspection['reliability_deficit'])
                        applied = rectangle(heat, *memory.grid(row), 'shifted') if policy == 'shifted' else rect
                        hidden, mass_error = memory.attenuate(row, past, reader, applied, strength, reference)
                        proposed = model.get_output_embeddings()(hidden).float()[row, 0]
                        logp = float(proposed.log_softmax(0)[token]); new = int(proposed.argmax())
                        record['trials'].append(dict(locator=locator, policy=policy, rect=rect, applied_rect=applied,
                            grid=memory.grid(row), crop_file=crop['image_file'], pixel_box=crop['pixel_box'],
                            drift=drift, strength=strength, inspection=inspection, top5=top(proposed),
                            original_logp=logp, original_logp_delta=logp-record['native_logp'],
                            token_changed=new != token, would_accept=new != token and logp < record['native_logp'],
                            visual_mass_relative_error_fp32=mass_error))
                if not zero_checked: assert cache_digest(past) == before_digest; zero_checked = True
                results.append(record)
                print(dict(image=case['image_id'], step=step, native=record['native']['text'][0],
                    proposals=[(t['locator'], t['policy'], t['top5']['text'][0], t['inspection']['reliability_deficit']) for t in record['trials']]), flush=True)
    encoder_hook.remove()
    assert len(results) == len(group['cases']) and encoder[0] == 1+memory.crop_encoder_calls
    for filename, expected in plan['source_sha256'].items(): assert sha(REPO / filename) == expected, filename
    write(root / f'results{shard}.json', results)
    write(root / f'worker{shard}.json', dict(passed=True, shard=shard, cases=len(results),
        original_encoder_calls=1, crop_encoder_calls=memory.crop_encoder_calls, actual_encoder_calls=encoder[0],
        crop_cache_hits=memory.vision_cache_hits, inspection_cache_hits=memory.verification_cache_hits,
        cached_vision_prompt_reconstructions=memory.reconstruction_calls,
        zero_tail_and_full36_cache_rollback_checked=zero_checked, seconds=time.time()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), no_new_captions=True, all_trials_rolled_back=True))


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
