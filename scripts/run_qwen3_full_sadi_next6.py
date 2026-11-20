"""Original next2 versus next6 after complete SADI, with nominal-strength control."""

import argparse
from contextlib import contextmanager,ExitStack
import gc
import json
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_qwen3_signed_feedback import read,write,sha
from scripts.run_qwen3_region_guidance import batch,generate
from scripts.qwen3_caption_groups_with_cache import source_groups

SOURCE=REPO/'outputs/generation_time_evidence_drift_qwen3_layer_feedback100_b16'
SVAR=REPO/'outputs/qwen3_mass_matched_visual_pilot_20261119'
PROOF=REPO/'outputs/qwen3_full_sadi_next6_math_20261120.json'
MODES=dict(next6_matched=5/3,next6_per_layer=5.)
OLD_NEXT='sadi_sar_ems_next_19_20'


def check(plan):
    for name,wanted in plan['source_sha256'].items():assert sha(REPO/name)==wanted,name


def prepare(root):
    assert not root.exists(),'Inspect existing state instead of overwriting or relaunching.'
    old=read(SOURCE/'protocol.json');proof=read(PROOF)
    assert read(SOURCE/'run_status.json')['status']=='complete' and proof['passed'] and not proof['cuda_initialized']
    for name,wanted in old['source_sha256'].items():assert sha(REPO/name)==wanted,name
    for name,wanted in proof['source_sha256'].items():assert sha(Path(name))==wanted,name
    from transformers import AutoTokenizer
    from nltk.corpus import wordnet
    from features.llava_decoding_evidence import lexical_nouns
    from scripts.generation_time_evidence_drift_qwen3 import MODEL_ROOT
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_ROOT),local_files_only=True,trust_remote_code=True)
    nouns=sorted(lexical_nouns(tokenizer,word_start='Ġ'));assert nouns
    paths=[Path(__file__),REPO/'features/qwen3_full_sadi_expanded_next.py',REPO/'scripts/verify_qwen3_full_sadi_expanded_next.py',
        REPO/'scripts/analyze_qwen3_full_sadi_next6.py',REPO/'scripts/analyze_qwen3_clva_quality.py',
        REPO/'scripts/run_qwen3_semantic_final_pilot.py',REPO/'scripts/run_qwen3_region_guidance.py',
        REPO/'scripts/qwen3_caption_groups_with_cache.py',REPO/'scripts/audit_qwen3_caption_with_native_cache_v2.py',
        REPO/'scripts/evaluate_qwen3_region_caption_chair.py',REPO/'scripts/evaluate_caption_chair.py',
        REPO/'scripts/evaluate_qwen3_next200_sentence_tokenized.py',REPO/'coco-labeling/coco_chair.py',
        REPO/'models/__init__.py',REPO/'utils/config_utils.py',SOURCE/'protocol.json',SOURCE/'cohort.json',
        SOURCE/'run_status.json',SOURCE/'environment.json',SVAR/'protocol.json',PROOF]
    environment=read(SOURCE/'environment.json');qwen_source=Path(environment['qwen_source'])
    assert sha(qwen_source)==environment['qwen_source_sha256'];paths.append(qwen_source)
    for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
        p=Path(MODEL_ROOT)/name
        if p.exists():paths.append(p)
    pointer=wordnet.root
    if hasattr(pointer,'zipfile'):paths.append(Path(pointer.zipfile.filename))
    else:paths.extend(p for p in Path(str(pointer)).iterdir() if p.is_file())
    groups={};other=[]
    for shard in [0,1]:
        original=source_groups(SOURCE/f'coco100/sadi.shard{shard}.jsonl')
        reference=source_groups(SOURCE/f'coco100/{OLD_NEXT}.shard{shard}.jsonl')
        svar=source_groups(SVAR/f'coco100/svar.shard{shard}.jsonl')
        key=lambda g:(tuple(r['image_id'] for r in g),g[0]['visual_token_count'])
        assert list(map(key,original))==list(map(key,reference))==list(map(key,svar))
        assert len(original[0])==16
        groups[str(shard)]=[dict(ids=list(key(g)[0]),visual_tokens=key(g)[1]) for g in original]
        paths.extend([SOURCE/f'coco100/{name}.shard{shard}.jsonl' for name in ['sadi',OLD_NEXT]])
        paths.append(SVAR/f'coco100/svar.shard{shard}.jsonl')
        other.append(dict(variant='svar',shard=shard,source=str(paths[-1]),source_sha256=sha(paths[-1])))
    hashes={**old['source_sha256'],**{str(p):sha(p) for p in paths}}
    root.mkdir();(root/'coco100').mkdir();(root/'source').mkdir()
    (root/'cohort.json').write_bytes((SOURCE/'cohort.json').read_bytes())
    write(root/'wordnet_nouns.json',dict(token_ids=nouns,word_start='Ġ',source='Unchanged original WordNet physical-noun senses; no309 restriction.'))
    for shard in [0,1]:
        for name in ['sadi',OLD_NEXT]:
            p=SOURCE/f'coco100/{name}.shard{shard}.jsonl';(root/'coco100'/p.name).write_bytes(p.read_bytes())
        p=SVAR/f'coco100/svar.shard{shard}.jsonl';(root/'coco100'/p.name).write_bytes(p.read_bytes())
    hashes.update({str(root/n):sha(root/n) for n in ['cohort.json','wordnet_nouns.json']})
    for p in paths:
        if p.suffix=='.py':(root/'source'/p.name).write_bytes(p.read_bytes())
    write(root/'protocol.json',dict(fixed_at=time.time(),source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),
        source_sha256=hashes,other_controls=other,groups=groups,cohort='coco100',images=100,shards=2,max_batch=16,max_new_tokens=512,
        variants=['sadi',OLD_NEXT,'svar',*MODES],candidates=list(MODES),copied_controls=['sadi',OLD_NEXT],
        base_layers=list(range(5,19)),source_layers=list(range(18,24)),target_layers=list(range(19,25)),
        strengths=MODES,reference_strength=5.,reference_targets=[19,20],nominal_sums=dict(reference=10.,next6_matched=10.,next6_per_layer=30.),
        gate='Unchanged original WordNet physical nouns and positive local FFN gain, refreshed at each source layer. No108 MLP or final309 trigger.',
        budget='strength*HellingerSquared(Q_SAR,P_FFN)*tanh(gain/2)*relu(Q-P)/max(maxdeficit,1e-6); tau=.03. Not original K32 OT risk.',
        interpretation='Match nominal coefficient sums only. Actual budgets and trigger decisions differ with the trajectory and source layer.',
        compression='Original whole-M full-vocabulary head and logsumexp once; retain ALL original WordNet columns and normalizer/Gram. Same softplus math. No64-chunk head shape or vocabulary narrowing.',
        preflight='First originalB16 on each GPU: native32 and old-next32 cached-prefix check; seeded full adapter2 exactold-next32; disabled6 exactnative32; forcedlive6 eight steps with one call per attention/MLP and one KV advance; forcedzero6 exactnative32; compactadapter2 exactold-next32.',
        reuse='All200 SADI/old-next and100 SVAR controls copied unchanged. Same original22 ordered groups. Shared vision/DeepStack and strict static terms across new arms; no duplicate model loads between preflight and generation.',
        scope='Reused development100; full cohort and physical/repetition review before any reserved500. No validation tuning.'))
    check(read(root/'protocol.json'));print(dict(prepared=True,nouns=len(nouns),groups={s:len(g) for s,g in groups.items()}),flush=True)


class VisionCache:
    def __init__(self,wrapper,inputs):self.wrapper=wrapper;self.inputs=inputs;self.entry=None;self.calls=self.hits=0
    @contextmanager
    def capture(self):
        import torch
        visual=self.wrapper.model.model.visual;original=visual.forward
        def forward(pixels,grid_thw,**kwargs):
            if self.entry is None:
                result=original(pixels,grid_thw,**kwargs);self.calls+=1
                self.entry=dict(pixels=pixels.detach().clone(),grid=grid_thw.detach().clone(),
                    output=(result[0].detach().clone(),[v.detach().clone() for v in result[1]]))
                return result
            assert torch.equal(self.entry['pixels'],pixels) and torch.equal(self.entry['grid'],grid_thw)
            self.hits+=1;return self.entry['output']
        with patch.object(visual,'forward',forward):yield self


def preflight(root,shard,wrapper,nouns,ids,span,inputs,shared):
    import torch
    from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
    from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback
    from features.qwen3_full_sadi_expanded_next import Qwen3FullSadiExpandedNext,seed_from_original,compact_shared
    reports=[]
    native=generate(wrapper,inputs,Qwen3BatchLateAttention(wrapper.model,span,nouns,'sadi'),32,True)
    controls=source_groups(root/f'coco100/sadi.shard{shard}.jsonl')[0]
    assert native[0]==[r['response_token_ids'][:32] for r in controls]
    old=Qwen3BatchLayerFeedback(wrapper.model,span,nouns,mode='next',strength=5.)
    reference=generate(wrapper,inputs,old,32,True)
    controls=source_groups(root/f'coco100/{OLD_NEXT}.shard{shard}.jsonl')[0]
    assert reference[0]==[r['response_token_ids'][:32] for r in controls]
    seed_from_original(old,shared);del old;gc.collect()
    for mode,kwargs,cap in [('adapter2_full',dict(targets=(19,20),compact=False),32),
        ('disabled6',dict(compact=False,disable_guidance=True),32),
        ('forced_live6',dict(compact=False,force_guidance=True),8),
        ('forced_zero6',dict(compact=False,force_guidance=True,zero_budget=True),32)]:
        controller=Qwen3FullSadiExpandedNext(wrapper.model,span,nouns,shared=shared,**kwargs)
        counters={i:dict(attention=0,mlp=0) for i in range(36)}
        def increment(index,key):
            def hook(*_):counters[index][key]+=1
            return hook
        with ExitStack() as stack:
            for i,layer in enumerate(wrapper.model.model.language_model.layers):
                stack.callback(layer.self_attn.register_forward_hook(increment(i,'attention')).remove)
                stack.callback(layer.mlp.register_forward_hook(increment(i,'mlp')).remove)
            result=generate(wrapper,inputs,controller,cap,True)
        assert all(v['attention']==v['mlp']==len(result[1]) for v in counters.values())
        if mode=='adapter2_full':assert result[:3]==reference[:3]
        elif mode!='forced_live6':assert result[:3]==native[:3]
        stats=controller.statistics(result[0]);events=[e for s in stats for e in s['full_events']]
        if mode.startswith('forced'):
            assert set(e['target_layer'] for e in events)==set(range(19,25))
            assert all(e['target_layer']==e['source_layer']+1 for e in events)
            if mode=='forced_zero6':assert all(e['budget_max']==0 for e in events)
        reports.append(dict(mode=mode,passed=True,exact_reference=mode!='forced_live6',events=len(events),
            seconds=result[3],model_steps=len(result[1]),calls_per_layer=counters))
        write(root/f'preflight_progress{shard}.json',reports);print(dict(shard=shard,mode=mode,passed=True),flush=True)
        controller.close();del controller,result,stats;gc.collect();torch.cuda.empty_cache()
    compression=compact_shared(shared,nouns);gc.collect();torch.cuda.empty_cache()
    controller=Qwen3FullSadiExpandedNext(wrapper.model,span,nouns,targets=(19,20),shared=shared,compact=True)
    current=generate(wrapper,inputs,controller,32,True);assert current[:3]==reference[:3]
    reports.append(dict(mode='adapter2_compact',passed=True,exact_old_next=True,seconds=current[3]))
    controller.close();del controller,current;gc.collect();torch.cuda.empty_cache()
    write(root/f'preflight{shard}.json',dict(passed=True,ids=ids,cases=reports,compression=compression,
        native_tokens=native[0],next_tokens=reference[0],native_full_cache_sha256=native[2],next_full_cache_sha256=reference[2]))


def worker(root,shard):
    import torch
    from features.llava_decoding_evidence import lexical_nouns
    from features.qwen3_full_sadi_expanded_next import Qwen3FullSadiExpandedNext
    from scripts.run_qwen3_semantic_final_pilot import build_wrapper
    plan=read(root/'protocol.json');check(plan);wrapper=build_wrapper();started=time.time()
    nouns=lexical_nouns(wrapper.tokenizer,word_start='Ġ');assert sorted(nouns)==read(root/'wordnet_nouns.json')['token_ids']
    groups=source_groups(root/f'coco100/sadi.shard{shard}.jsonl');reports=[]
    paths={m:root/f'coco100/{m}.shard{shard}.jsonl' for m in MODES};assert not any(p.exists() for p in paths.values())
    with torch.inference_mode():
        for index,group in enumerate(groups):
            ids=[r['image_id'] for r in group];span,inputs,photos=batch(wrapper,ids);del photos
            assert span[1]-span[0]==group[0]['visual_token_count']
            shared={};vision=VisionCache(wrapper,inputs)
            with vision.capture():
                if index==0:
                    preflight(root,shard,wrapper,nouns,ids,span,inputs,shared)
                    while not all((root/f'preflight{s}.json').exists() for s in [0,1]):
                        assert not any((root/f'failure{s}.json').exists() for s in [0,1]),'Other preflight failed.'
                        time.sleep(2)
                    assert all(read(root/f'preflight{s}.json')['passed'] for s in [0,1])
                for mode,strength in MODES.items():
                    controller=Qwen3FullSadiExpandedNext(wrapper.model,span,nouns,strength=strength,shared=shared,compact=True)
                    prompt=inputs['input_ids'].shape[1];calls=[]
                    def verify_cache(_m,_a,out):
                        assert all(out.past_key_values.get_seq_length(i)==prompt+len(calls) for i in range(36))
                        calls.append(1)
                    handle=wrapper.model.register_forward_hook(verify_cache)
                    old_calls,old_hits=vision.calls,vision.hits
                    try:result=generate(wrapper,inputs,controller,512,False)
                    finally:handle.remove()
                    assert len(calls)==controller.steps
                    stats=controller.statistics(result[0]);texts=wrapper.processor.batch_decode(result[0],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    for s in stats:s.update(original_encoder_calls=vision.calls-old_calls,vision_cache_hits=vision.hits-old_hits)
                    with paths[mode].open('a',encoding='utf-8') as f:
                        for image,tokens,text,s in zip(ids,result[0],texts,stats,strict=True):
                            f.write(json.dumps(dict(image_id=image,model='qwen3_vl_8b',variant=mode,response_token_ids=tokens,
                                generated_text=text.strip(),seconds=result[3]/len(ids),batch_wall_seconds=result[3],batch_size=len(ids),
                                visual_token_count=span[1]-span[0],sar_ems_statistics=s,decoder='native_qwen3_generate_same_span_batch',
                                generation_reused=False,actual_model_calls_in_batch=controller.steps),ensure_ascii=False)+'\n')
                    report=dict(group=index,mode=mode,images=len(ids),seconds=result[3],events=sum(s['guided_attention_calls'] for s in stats),
                        encoder_calls=vision.calls-old_calls,vision_cache_hits=vision.hits-old_hits,new_terms=controller.new_terms,
                        shared_term_hits=controller.cache_hits,logit_cache_bytes=sum(t.visual_logits.numel()*t.visual_logits.element_size() for t in shared['terms'].values()))
                    reports.append(report);write(root/f'progress{shard}.json',reports);print(dict(shard=shard,**report),flush=True)
                    controller.close();del controller,result,stats;gc.collect();torch.cuda.empty_cache()
            assert vision.calls==1
            shared.clear();del shared,vision,inputs;gc.collect();torch.cuda.empty_cache()
    check(plan);write(root/f'worker{shard}.json',dict(passed=True,seconds=time.time()-started,groups=len(groups),progress=reports))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1]);args=parser.parse_args()
    try:prepare(args.root) if args.prepare else worker(args.root,args.worker)
    except BaseException:
        if args.root.exists():write(args.root/f'failure{args.worker}.json',dict(failed_at=time.time(),traceback=traceback.format_exc()))
        raise
