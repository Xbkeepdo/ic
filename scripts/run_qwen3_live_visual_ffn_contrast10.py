"""Complete-caption screen for visual FFN-output preference and matched controls."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_qwen3_signed_feedback import read, write, sha
from scripts.qwen3_caption_groups_with_cache import source_groups

PREVIOUS = REPO/'outputs/qwen3_visual_ffn_preference_probe_20261120'
CONTROLS = REPO/'outputs/qwen3_clva_signal_preflight10_v2_20261120'
MODES = ['uv_excess', 'hpre_excess', 'uniform', 'rolled_uv_excess']


def check(plan):
    for name, wanted in plan['source_sha256'].items(): assert sha(REPO/name) == wanted, name


def prepare(root):
    assert not root.exists()
    old = read(PREVIOUS/'protocol.json'); assert read(PREVIOUS/'pipeline_complete.json')['passed']
    assert read(CONTROLS/'pipeline_complete.json')['passed']
    assert read(PREVIOUS/'math_proof.json')['passed']
    for shard in [0,1]:
        assert read(PREVIOUS/f'worker{shard}.json')['passed'] and read(PREVIOUS/f'output_preference_audit{shard}.json')['passed']
        assert sha(PREVIOUS/f'ffn_outputs{shard}.npz')==read(PREVIOUS/f'worker{shard}.json')['cache_sha256']
    files = [Path(__file__), REPO/'features/qwen3_live_preference_contrast.py',
        REPO/'scripts/run_qwen3_full_sadi_next6.py', REPO/'scripts/run_qwen3_region_guidance.py',
        REPO/'scripts/analyze_qwen3_clva_quality.py', REPO/'scripts/evaluate_caption_chair.py',
        REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py', PREVIOUS/'protocol.json',
        PREVIOUS/'math_proof.json', PREVIOUS/'pipeline_complete.json', CONTROLS/'protocol.json',
        CONTROLS/'pipeline_complete.json', CONTROLS/'cohort.json', CONTROLS/'object_lexicon.json']
    files += [CONTROLS/f'coco10/{n}.shard{s}.jsonl' for n in ['sadi', 'svar'] for s in [0,1]]
    files += [REPO/'features/qwen3_live_visual_ffn_contrast.py',PREVIOUS/'math_proof.json']
    files += [PREVIOUS/f'{name}{shard}{suffix}' for name,suffix in [('worker','.json'),('output_preference_audit','.json'),('ffn_outputs','.npz')] for shard in [0,1]]
    hashes = {**old['source_sha256'], **{str(p): sha(p) for p in files}}
    plan = dict(fixed_at=time.time(), source_sha256=hashes, source=str(CONTROLS),
        previous=str(PREVIOUS), groups=old['groups'], cohort='coco10', images=10, shards=2,
        max_batch=16,max_new_tokens=512,variants=['sadi','svar',*MODES],candidates=MODES,copied_controls=['sadi','svar'],
        base_layers=list(range(5,19)), source_layer=19, action_layers=list(range(20,36)),
        eligibility='Frozen309 COCO object word-start token IDs; all native final targets including repeats, skip prefill. No MLP or history-class filter.',
        action='Read actual visual FFN19 outputs without additional sourceFFN forward. P_uv from cosine(u_visual,u_query)/.03; negative attention prior is uv-excess, old hpre-excess, uniform or rolled uv-excess. NativeSADI positive; negative20--35 KV always restored.',
        strengths={n:'.5*HellingerSquared(Q,P_uv)' for n in MODES},
        decoding='2*native_logits-negative_logits, native plausibility beta.1, alpha1. No target-probability acceptance filter.',
        sharing='Same original B/P groups only: vision, source19 semantics, and actual visual FFN outputs shared after exact tensor/weight identity checks. Query updates always read native current states.',
        preflight='Native128 matches cached controls; disabled128 and four actual nonzero branches with alpha0 plus uv-excess eta0 replay exact tokens/logits/all36current/fullKV. Disabled native visual FFN tensors also exactly match earlier saved BF16 cache.',
        scope='Complete10 exploratory screen, not independent validation. No500 selection from a single changed word; if useful inspect full captions and proceed to full100.',
        timing='Warm shared static/vision between variants; times are not independent cold-start comparisons.')
    assert plan['copied_controls']==['sadi','svar']
    check(plan); root.mkdir(); (root/'coco10').mkdir(); (root/'source').mkdir()
    for p in [Path(__file__),REPO/'features/qwen3_live_visual_ffn_contrast.py',REPO/'features/qwen3_visual_ffn_preference.py']:
        (root/'source'/p.name).write_bytes(p.read_bytes())
    for name in ['cohort.json', 'object_lexicon.json']: (root/name).write_bytes((CONTROLS/name).read_bytes())
    assert set(read(root/'cohort.json')['coco10']) == {i for g in plan['groups'] for i in g['ids']}
    assert len(read(root/'object_lexicon.json')['token_ids']) == 309
    for n in ['sadi','svar']:
        for s in [0,1]: (root/f'coco10/{n}.shard{s}.jsonl').write_bytes((CONTROLS/f'coco10/{n}.shard{s}.jsonl').read_bytes())
    write(root/'protocol.json', plan)
    write(root/'analysis_request.json', dict(fixed_at=time.time(), protocol_sha256=sha(root/'protocol.json'),
        primary='Full10 sentence CHAIRs/Ci, false/total mentions, image recall/F1. Legacy secondary, paired5000 seed43.',
        review='Inspect whole descriptions and changed false classes versus actual images, repetitions and denominators. No efficacy claim from token flip.',
        before_new_caption_metrics=True))
    from scripts.analyze_qwen3_clva_quality import prepare as quality_prepare
    quality_prepare(root)


def worker(root, shard):
    import torch
    from features.qwen3_live_visual_ffn_contrast import Qwen3LiveVisualFFNContrast
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    from scripts.run_qwen3_region_guidance import batch, generate
    from scripts.run_qwen3_full_sadi_next6 import VisionCache
    plan = read(root/'protocol.json'); check(plan)
    wrapper = build_wrapper(); ids = plan['groups'][shard]['ids']; span, inputs, _ = batch(wrapper, ids)
    nouns = frozenset(read(root/'object_lexicon.json')['token_ids']); shared = {}; reports = []
    baseline = [r['response_token_ids'] for g in source_groups(root/f'coco10/sadi.shard{shard}.jsonl') for r in g]
    vision = VisionCache(wrapper, inputs); started = time.time()
    def make(mode='uv_excess', **kwargs): return Qwen3LiveVisualFFNContrast(wrapper,span,nouns,policy=mode,shared=shared,**kwargs)
    with torch.inference_mode(), vision.capture():
        native = generate(wrapper, inputs, Qwen3BatchLateAttention(wrapper.model, span, nouns, 'sadi'), 128, True)
        assert native[0] == [t[:128] for t in baseline]
        diagnostics=[('disabled','uv_excess',dict(disabled=True))]
        diagnostics += [('alpha0_'+n,n,dict(zero_alpha=True)) for n in MODES]
        diagnostics += [('zero_amount','uv_excess',dict(zero_amount=True))]
        for name,mode,kwargs in diagnostics:
            controller = make(mode,**kwargs); result = generate(wrapper, inputs, controller, 128, True)
            assert result[:3] == native[:3], name
            events = [e for s in controller.statistics(result[0]) for e in s['full_events']]
            assert not any(e['applied'] for e in events)
            if name != 'disabled': assert events
            else:
                import numpy as np
                with np.load(PREVIOUS/f'ffn_outputs{shard}.npz',allow_pickle=False) as saved:
                    assert saved['batch_ids'].tolist()==ids and saved['span'].tolist()==list(span)
                    old_visual=torch.from_numpy(saved['visual_update_bf16'].copy()).view(torch.bfloat16).to(wrapper.device)
                row=plan['groups'][shard]['cases'][0]['row']
                assert torch.equal(controller.reader.visual_updates[19,row],old_visual)
            reports.append(dict(mode=name, passed=True, exact_sadi=True, events=len(events), seconds=result[3]))
            write(root/f'preflight_progress{shard}.json', reports); controller.close(); del controller,result
        write(root/f'preflight{shard}.json', dict(passed=True, checks=reports, native_tokens=native[0],
            native_per_step=native[1], full_cache_sha256=native[2], token_and_current_full36_cache_exact=True))
        while not all((root/f'preflight{s}.json').exists() for s in [0,1]):
            assert not any((root/f'failure{s}.json').exists() for s in [0,1]); time.sleep(1)
        assert all(read(root/f'preflight{s}.json')['passed'] for s in [0,1]); del native
        for mode in MODES:
            controller = make(mode); trace = [0]; prompt = inputs['input_ids'].shape[1]
            def lengths(_m, _a, out):
                assert all(out.past_key_values.get_seq_length(i) == prompt+trace[0] for i in range(36)); trace[0] += 1
            with ExitStack() as stack:
                stack.callback(wrapper.model.register_forward_hook(lengths).remove)
                result = generate(wrapper, inputs, controller, 512)
            stats = controller.statistics(result[0]); texts = wrapper.processor.batch_decode(result[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert trace[0] == controller.step+1
            rows = [dict(image_id=image,model='qwen3_vl_8b',variant=mode,response_token_ids=tokens,generated_text=text.strip(),
                seconds=result[3]/len(ids),batch_wall_seconds=result[3],batch_size=len(ids),visual_token_count=span[1]-span[0],
                sar_ems_statistics=s,decoder='native_qwen3_generate_same_span_batch',generation_reused=False,
                actual_model_calls_in_batch=controller.step+1) for image,tokens,text,s in zip(ids,result[0],texts,stats,strict=True)]
            with (root/f'coco10/{mode}.shard{shard}.jsonl').open('x',encoding='utf-8') as stream:
                for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            events = [e for s in stats for e in s['full_events']]
            report = dict(mode=mode,passed=True,seconds=result[3],row_events=len(events),
                immediate_changes=sum(e['immediate_change'] for e in events),replay_model_steps=controller.replay_steps,
                visual_ffn_capture_calls=controller.reader.visual_capture_calls,visual_ffn_cache_hits=controller.visual_ffn_cache_hits,
                semantic_head_calls=controller.reader.head_calls,model_steps=controller.step+1)
            reports.append(report); write(root/f'progress{shard}.json',reports); print(report,flush=True)
            controller.close(); del controller,result,stats; gc.collect(); torch.cuda.empty_cache()
    check(plan); assert vision.calls == 1
    write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,groups=1,reports=reports,
        original_encoder_calls=vision.calls,vision_hits=vision.hits))


def analyze(root):
    plan = read(root/'protocol.json'); check(plan)
    request = read(root/'analysis_request.json'); assert request['protocol_sha256'] == sha(root/'protocol.json')
    assert read(root/'pipeline_complete.json')['passed']
    for shard in [0,1]: assert read(root/f'worker{shard}.json')['passed'] and read(root/f'preflight{shard}.json')['passed']
    for mode in MODES:
        for shard in [0,1]:
            for group in source_groups(root/f'coco10/{mode}.shard{shard}.jsonl'):
                for row in group:
                    stats=row['sar_ems_statistics'];assert stats['eligibility_tokens']==309 and stats['mode']==mode
                    assert stats['source_preference']=='actual_visual_ffn_outputs19' and stats['common_uv_drift_strength']
                    assert stats['additional_source19_mlp_forwards']==0 and stats['visual_ffn_capture_calls']==1
                    assert stats['read_only_counterfactual'] and not any(stats[k] for k in ['disabled','zero_alpha','zero_amount'])
                    for e in stats['full_events']:
                        assert row['response_token_ids'][e['step']] == e['new_token']
                        assert not e['counterfactual_kv_committed'] and 0 <= e['drift'] <= 1
                        assert e['amount']==.5*e['drift']
    for version in ['sentence','legacy']:
        with (root/f'evaluate_{version}.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/evaluate_caption_chair.py','--generation-root',str(root),
                '--cohort','coco10','--limit','10','--variants',*plan['variants'],'--evaluator',version],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'run_status.json',dict(status='complete',finished=time.time()))
    with (root/'independent_audit.log').open('x') as log:
        subprocess.run([sys.executable,'scripts/audit_qwen3_caption_with_native_cache_v2.py',str(root)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(root/'independent_summary.json')['passed']; check(plan)
    write(root/'analysis_provenance.json',dict(passed=True,request_sha256=sha(root/'analysis_request.json'),
        protocol_sha256=sha(root/'protocol.json'),physical_review='pending',finished=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--prepare',action='store_true');p.add_argument('--analyze',action='store_true');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    try:
        if a.prepare: prepare(a.root)
        elif a.analyze: analyze(a.root)
        elif a.worker is not None: worker(a.root,a.worker)
        else: p.error('Select a stage.')
    except BaseException:
        if a.root.exists(): write(a.root/f'failure{a.worker}.json',dict(traceback=traceback.format_exc(),failed_at=time.time()))
        raise
