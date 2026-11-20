"""Fixed/guided late ACG adaptations; preserve complete SADI and cached controls."""

import argparse
import gc
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_qwen3_signed_feedback as shared
from scripts import run_qwen3_history_route_pilot as driver
from scripts.run_qwen3_semantic_final_pilot import build_wrapper

OUT = shared.OUT/'qwen3_acg_signal_pilot_20261118'
SOURCE = shared.OUT/'qwen3_semantic_donor_pilot_20261118_memory_recovery'
PREFLIGHT = shared.OUT/'qwen3_acg_signal_preflight_v2/verification.json'
CPU = shared.OUT/'qwen3_acg_signal_cpu_v2/verification.json'
MODES = {'sadi_sam_acg_fixed_31': False, 'sadi_sam_acg_guided_31': True}
FILES = ['features/qwen3_acg_signal_feedback.py', 'scripts/verify_qwen3_acg_signal.py',
         'scripts/verify_qwen3_acg_signal_gpu.py', 'scripts/run_qwen3_acg_signal_pilot.py',
         'scripts/audit_qwen3_acg_signal_pilot.py', 'scripts/audit_qwen3_cached_caption_pilot.py',
         'scripts/audit_qwen3_caption_denominator.py']


def prepare():
    assert not OUT.exists(), 'Inspect existing job before relaunching'
    assert shared.read(SOURCE/'final_provenance.json')['passed']
    old = shared.read(SOURCE/'protocol.json')
    hashes = dict(old['source_sha256'])
    assert shared.read(CPU)['passed'] and shared.read(PREFLIGHT)['passed']
    for name, expected in shared.read(PREFLIGHT)['source_sha256'].items():
        assert shared.sha(REPO/name) == expected, name
    for path in [CPU, PREFLIGHT, SOURCE/'protocol.json', SOURCE/'human_review.json',
                 *(REPO/n for n in FILES)]:
        hashes[str(path)] = shared.sha(path)
    for name, expected in hashes.items():
        assert shared.sha(Path(name)) == expected, name
    OUT.mkdir(); (OUT/'logs').mkdir(); (OUT/'coco100').mkdir()
    for name in ('cohort.json', 'object_lexicon.json'):
        shutil.copyfile(SOURCE/name, OUT/name)
    controls = old['variants']
    for shard in (0, 1):
        for name in controls:
            path = SOURCE/f'coco100/{name}.shard{shard}.jsonl'
            hashes[str(path)] = shared.sha(path)
            shutil.copyfile(path, OUT/'coco100'/path.name)
    plan = dict(stage='late-acg-signal-pilot', images=100, shards=2, cohort='coco100', max_batch=16,
        max_new_tokens=512, variants=[*controls, *MODES], candidates=list(MODES), source=str(SOURCE),
        copied_controls=controls, cached_controls=['sadi'], source_sha256=hashes, fixed_at=time.time(),
        probe=old['probe'], real_threshold=old['real_threshold'],
        recipe=dict(base_layers=[5, 18], source_layer=31, intervention_layer=31, target_read_layer=35,
            gate='Same original108 native SAM-V2 p_real<source_real5th_percentile, first object token/class.',
            borrowed='ACG Algorithm1 equations5--8: masked-visual attention path with reused QKV; remove text-parallel contrast component.',
            scope='Both methods are gated layer31 ACG adaptations after SADI, not all-layer original-paper ACG results.',
            gamma=1.4, gamma_source='Fixed from ACG Qwen-VL setting before experiments; not claimed optimal for Qwen3.',
            fixed='Same1.4 scale at every head.',
            guided='Q=Norm(native visual-write norms*(1+log_s)); P=softmax(cos(uFFN,hprevis)/.03); D=HellingerSquared(Q,P).',
            head_scale='1.4*D*relu(-gA_WO_dot_ACGcontrast_h)/max_heads(relu(-gA_WO_dot_ACGcontrast)); all nonnegative effects -> no action.',
            attention_output='Add scaled orthogonal contrast to native attention context, before o_proj and FFN. This is not probability-mass transport.',
            acceptance='Original target actual final log probability strictly decreases; else exact hidden/current32--35 KV rollback.',
            costs='Fixed arm no backward. Guided one31FFN+32--35 backward per selected step; no extra encoder or QKV. FullB replay. Cache309 fully normalized probabilities shared per original group after exact image-prefix equality.',
            memory='After every generation collect owner-reader cycles and free unused allocator blocks, log allocated memory; group shared semantics kept until both methods finish.',
            saved='Captions and scalar/head32 diagnostics; no hidden/attention/full-vocabulary tensor dumps.'),
        reference='https://arxiv.org/html/2601.13707v2',
        development_screen='Vs SADI both Cs/Ci and false mentions decrease, F1 loss<=1.5pp, recall loss<=2pp, leave-largest Ci no worse and image-confirmed identity correction; guided must improve fixed adaptation.',
        followup='If supported: matched-head weighting control, proper ACG scope ablation, then independent500. Reused development100 not independent evidence.')
    shared.write(OUT/'protocol.json', plan)
    return plan


def worker(gpu):
    import json
    import torch
    from features.coco_object_history import ObjectClassHistory
    from features.qwen3_risk_i_el_probe import FullRiskIElProbe
    from features.qwen3_acg_signal_feedback import Qwen3BatchACGSignal
    from scripts import generation_time_evidence_drift_qwen3_batched as common
    from scripts import generation_time_qwen3_layer_feedback as original
    from scripts.generation_time_evidence_drift_qwen3 import COCO_ROOT, MODEL
    plan = shared.read(OUT/'protocol.json')
    for name, expected in plan['source_sha256'].items():
        assert shared.sha(Path(name)) == expected, name
    wrapper = build_wrapper()
    allowed = frozenset(shared.read(OUT/'object_lexicon.json')['token_ids'])
    parser = ObjectClassHistory(REPO/'coco-labeling/coco_chair.py')
    probe = FullRiskIElProbe(plan['probe']).to(wrapper.device).eval()
    groups = shared.source_groups(OUT/f'coco100/sadi.shard{gpu}.jsonl')
    paths = {name: OUT/f'coco100/{name}.shard{gpu}.jsonl' for name in MODES}
    assert not any(p.exists() for p in paths.values())
    memory = []
    with torch.inference_mode():
        for index, group in enumerate(groups):
            ids = [r['image_id'] for r in group]
            (span, inputs), = common.same_span_batches(wrapper, ids, COCO_ROOT, len(ids))
            assert [i for i, _ in inputs] == ids and span[1]-span[0] == group[0]['visual_token_count']
            if index == 0:
                tokens, _, _ = original.generate_group(wrapper, span, inputs, allowed, 'sadi', 32)
                assert tokens == [r['response_token_ids'][:32] for r in group]
                shared.write(OUT/f'verification.gpu{gpu}.json', dict(passed=True, image_ids=ids,
                    actual_batch=len(ids), exact_native_prefix_tokens=32))
                gc.collect(); torch.cuda.empty_cache()
            static = {}
            for name, guided in MODES.items():
                def factory(model, current_span, nouns, mode, strength):
                    return Qwen3BatchACGSignal(model, current_span, allowed, probe=probe,
                        semantic_tokens=sorted(allowed), tokenizer=wrapper.tokenizer, history_parser=parser,
                        shared_static=static, real_threshold=plan['real_threshold'], signal_guided=guided, gamma=1.4)
                torch.cuda.reset_peak_memory_stats(wrapper.device)
                with patch.object(original, 'Qwen3BatchLayerFeedback', factory):
                    tokens, stats, seconds = original.generate_group(wrapper, span, inputs, allowed, shared.POSITIVE, 512)
                texts = wrapper.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                with paths[name].open('a', encoding='utf-8') as stream:
                    for image_id, current, text, info in zip(ids, tokens, texts, stats, strict=True):
                        assert 0 < len(current) <= 512 and info['maximum_probe_hidden_difference'] == 0
                        stream.write(json.dumps(dict(image_id=image_id, model=MODEL, variant=name,
                            generated_text=text.strip(), response_token_ids=current, seconds=seconds/len(ids),
                            batch_size=len(ids), batch_wall_seconds=seconds, visual_token_count=span[1]-span[0],
                            sar_ems_statistics=info, decoder='native_qwen3_generate_same_span_batch'), ensure_ascii=False)+'\n')
                peak = torch.cuda.max_memory_allocated(wrapper.device)
                before = torch.cuda.memory_allocated(wrapper.device)
                collected = gc.collect(); torch.cuda.empty_cache()
                memory.append(dict(group=index, name=name, before_gc=before,
                    after_gc=torch.cuda.memory_allocated(wrapper.device), peak=peak, collected=collected))
                shared.write(OUT/f'memory.gpu{gpu}.json', memory)
                print(f'[{gpu}] {name} group={index} batch={len(ids)} seconds={seconds:.3f}', flush=True)
            del static
            gc.collect(); torch.cuda.empty_cache()
            memory[-1]['after_group_gc'] = torch.cuda.memory_allocated(wrapper.device)
            shared.write(OUT/f'memory.gpu{gpu}.json', memory)
    assert all(shared.group_keys(shared.source_groups(p)) == shared.group_keys(groups) for p in paths.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--worker', type=int, choices=(0, 1))
    args = p.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        with patch.object(driver, 'OUT', OUT), patch.object(driver, '__file__', __file__), patch.object(driver, 'prepare', prepare):
            driver.run()
