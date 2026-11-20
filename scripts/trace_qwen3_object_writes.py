"""Measure token-conditioned writes and FFN drift on saved SADI caption prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / 'outputs'
SOURCE = OUT / 'generation_time_evidence_drift_qwen3_fresh_alternative500'
RUN = OUT / 'qwen3_sadi_first_object_trace'
LAYERS = (18, 24, 25, 28, 31)
CAP = 128


def load(path):
    return json.loads(path.read_bytes())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def groups(shard):
    path = SOURCE / f'coco500/sadi.shard{shard}.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert len(rows) == 250
    cursor = 0
    while cursor < len(rows):
        n = rows[cursor]['batch_size']
        batch = rows[cursor:cursor+n]
        assert len(batch) == n and len({r['batch_wall_seconds'] for r in batch}) == 1
        cursor += n
        # All the historical maximal batches are used. Keep exact GPU shards,
        # companion images and order so replay remains comparable.
        if n == 16:
            yield batch


def prepare():
    assert load(SOURCE / 'independent_summary.json')['status'] == 'passed'
    if RUN.exists():
        raise FileExistsError(RUN)
    batches = [[dict(ids=[r['image_id'] for r in batch],
                     fold=(i + shard) % 2) for i, batch in enumerate(groups(shard))] for shard in (0, 1)]
    RUN.mkdir()
    plan = {'layers': LAYERS, 'steps': CAP, 'source': str(SOURCE), 'batches': batches,
            'split': 'alternate original B16 batches within each GPU shard',
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                SOURCE / 'protocol.json', SOURCE / 'batch_audit.json',
                SOURCE / 'coco500/sadi.shard0.jsonl', SOURCE / 'coco500/sadi.shard1.jsonl', Path(__file__))}}
    write(RUN / 'protocol.json', plan)
    # Register the diagnostic sampling scope for any future unused-image audit.
    write(RUN / 'cohort.json', {'trace': sorted(i for shard in batches for b in shard for i in b['ids'])})
    print('FROZEN', sum(len(b['ids']) for shard in batches for b in shard), 'images', flush=True)


def worker(shard):
    import nltk
    import torch
    from torch.nn import functional as F
    from transformers import LogitsProcessor, StoppingCriteria

    from features.llava_decoding_evidence import TAU, lexical_nouns
    from features.qwen3_sar_ems_layer_feedback import Qwen3BatchLayerFeedback
    from models import build_model
    from scripts.evaluate_generation_time_evidence_drift_llava import load_chair
    from scripts.evaluate_qwen3_next200_sentence_tokenized import load_original_chair_module, make_sentence_chair
    from scripts.generation_time_evidence_drift_qwen3 import CONFIG, COCO_ROOT, MODEL, MODEL_ROOT
    from scripts.generation_time_evidence_drift_qwen3_batched import same_span_batches
    from scripts.generation_time_sar_ems_seed43 import _generation_kwargs
    from unittest.mock import patch
    from utils.config_utils import get_extraction_model_cfg, load_config
    from utils.token_alignment import build_response_token_offsets, token_indices_for_char_span

    plan = load(RUN / 'protocol.json')
    slots = plan['batches'][shard]
    base = load_original_chair_module()
    sentence = make_sentence_chair(base.CocoChairEvaluator, nltk.data.load('tokenizers/punkt/english.pickle'))
    eval_root = SOURCE / 'evaluations/sentence_v1'
    with patch.object(base, 'CocoChairEvaluator', sentence):
        chair, cache = load_chair(COCO_ROOT, eval_root, load(SOURCE/'cohort.json')['coco500'], None)
    assert load(cache)['evaluator_provenance']['evaluator'] == 'sentence'

    torch.set_num_threads(1)
    config = dict(get_extraction_model_cfg(load_config(str(CONFIG)), MODEL))
    config['hf_name'] = str(MODEL_ROOT)
    wrapper = build_model(MODEL, config, device='cuda:0')
    wrapper.model.eval().requires_grad_(False)
    tokenizer = wrapper.tokenizer
    nouns = lexical_nouns(tokenizer, word_start='Ġ')

    class Recorder(Qwen3BatchLayerFeedback):
        def __init__(self, span, wanted, records):
            super().__init__(wrapper.model, span, nouns)
            self.sources = LAYERS
            self.targets = tuple(i+1 for i in LAYERS)
            self.wanted = wanted
            self.records = records
            self.projections = {}

        def _budget(self, layer):
            active = self.wanted.get(self.step, [])
            if not active:
                return [], None, []
            indices = [row for row in active]
            requested = [self.records[row]['response_token_ids'][self.step] for row in indices]
            state = self.scratch[layer]
            r = state['ffn_residual'][indices]
            u = state['update'][indices]
            norm = self.decoder.norm
            before = norm(r).float()
            after = norm(r+u).float()
            weights = self.head.weight[requested].float()
            gain = (weights * (after-before)).sum(-1)
            logits = self.head(after.to(self.head.weight.dtype)).float()
            top_scores, top_ids = logits.topk(8, dim=-1)
            matches = (top_ids == top_ids.new_tensor(requested)[:, None]).any(-1)
            final_score = logits[torch.arange(len(indices),device=logits.device),requested]
            readout = weights * norm.weight.float()
            r = r.float()
            readout = readout - (readout*r).mean(-1,keepdim=True)*r/(r.square().mean(-1,keepdim=True)+norm.variance_epsilon)
            if layer not in self.projections:
                self.projections[layer] = self.decoder.layers[layer].self_attn.o_proj.weight.float()
            values = self.visual_values[layer][indices].float()
            direction = (readout @ self.projections[layer]).reshape(len(indices), values.shape[1], -1)
            attention = state['attention'][indices].float()
            write = torch.einsum('bhm,bhmd,bhd->bm',attention,values,direction)
            positive = write.clamp_min(0)
            total = positive.sum(-1)
            total_abs = write.abs().sum(-1)
            evidence = positive / total.clamp_min(1e-12)[:,None]
            ffn_sim = (self.visual_normalized[layer][indices] @ F.normalize(u.float(), dim=-1).unsqueeze(-1)).squeeze(-1)
            preference = (ffn_sim / TAU).softmax(dim=-1)
            drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0,1)
            numeric = torch.stack((write.sum(-1)/total_abs.clamp_min(1e-12), drift, gain,
                                   top_scores[:,0]-final_score, matches, attention.sum(-1).mean(-1),
                                   (top_ids[:,0] == top_ids.new_tensor(requested))),dim=-1).tolist()
            for row,data,token,source_top in zip(indices,numeric,requested,top_ids[:,0].tolist(),strict=True):
                event = active[row]
                event['features'][str(layer)] = dict(zip(
                    ('signed_support','drift','ffn_gain','logit_gap','in_top8','visual_mass','is_source_top'),data,strict=True))
                event['token_is_noun'] = token in nouns
                event['source_top_is_noun_'+str(layer)] = source_top in nouns
            return [], None, []

    class Replay(LogitsProcessor):
        def __init__(self, record, prompt_length, eos, wanted):
            self.records, self.prompt_length, self.eos, self.wanted = record, prompt_length, eos, wanted

        def __call__(self, input_ids, scores):
            step = input_ids.shape[1]-self.prompt_length
            for row in self.wanted.get(step, []):
                self.wanted[step][row]['replay_matches_target'] = int(scores[row].argmax()) == self.records[row]['response_token_ids'][step]
            chosen = [r['response_token_ids'][step] if step < len(r['response_token_ids']) else self.eos for r in self.records]
            return torch.full_like(scores, -float('inf')).scatter_(1,scores.new_tensor(chosen,dtype=torch.long)[:,None],0)

    class Stop(StoppingCriteria):
        def __init__(self,length):
            self.length = length

        def __call__(self,input_ids,scores,**_):
            return input_ids.new_full((len(input_ids),),input_ids.shape[1]>=self.length,dtype=torch.bool)

    eos_id = wrapper.model.generation_config.eos_token_id
    eos = eos_id[0] if isinstance(eos_id,list) else eos_id
    log = RUN / f'trace.shard{shard}.jsonl'
    if log.exists():
        raise FileExistsError(log)
    with torch.inference_mode(), log.open('x',encoding='utf-8') as stream:
        for index,(batch,slot) in enumerate(zip(groups(shard),slots,strict=True)):
            wanted = {}
            ids = [r['image_id'] for r in batch]
            assert ids == slot['ids']
            for row,r in enumerate(batch):
                text = r['generated_text']
                offsets = build_response_token_offsets(tokenizer,r['response_token_ids'],text)
                mentions = chair.compute_chair_token(r['image_id'], text)['object_mentions']
                seen = set()
                for mention in mentions:
                    canonical = mention['canonical_name']
                    repeat = canonical in seen
                    seen.add(canonical)
                    steps = token_indices_for_char_span(offsets,mention['char_start'],mention['char_end'])
                    if not steps:
                        raise ValueError(('Unmatched CHAIR object mention',r['image_id'],mention))
                    step = steps[0]
                    if step >= CAP:
                        continue
                    step_events = wanted.setdefault(step,{})
                    event = step_events.setdefault(row, {'image_id':r['image_id'],'step':step,'features':{},
                                                         'fold':slot['fold'],'batch':index,'shard':shard,
                                                         'target_token_id':r['response_token_ids'][step],'mentions':[]})
                    event['mentions'].append({'class':canonical,'surface':mention['surface'],
                                              'hallucinated':bool(mention['hallucinated']),'repeat':repeat})
            if not wanted:
                continue
            (span,inputs), = same_span_batches(wrapper,ids,COCO_ROOT,len(ids))
            assert [i for i,_ in inputs] == ids and span[1]-span[0] == batch[0]['visual_token_count']
            model_inputs={k:torch.cat([item[k] for _,item in inputs],dim=0).to(wrapper.device) for k in inputs[0][1]}
            prompt_length=model_inputs['input_ids'].shape[1]
            kwargs=_generation_kwargs(wrapper,MODEL)
            kwargs.update(max_new_tokens=256,logits_processor=[Replay(batch,prompt_length,eos,wanted)],
                          stopping_criteria=[Stop(prompt_length+max(wanted)+1)])
            controller=Recorder(span,wanted,batch)
            with controller.capture():
                wrapper.model.generate(**model_inputs,**kwargs)
            events=[event for step in sorted(wanted) for event in wanted[step].values()]
            for event in events:
                assert set(event['features'])=={str(i) for i in LAYERS}
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')
            stream.flush()
            print('BATCH',shard,index,'positions',len(events),'last_step',max(wanted),flush=True)


def main():
    prepare()
    processes=[]
    try:
        for shard in (0,1):
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                     NLTK_DATA=str(OUT/'generation_time_evidence_drift_llava/nltk_data'))
            with (RUN/f'gpu{shard}.log').open('w') as log:
                processes.append(subprocess.Popen([sys.executable,__file__,'--worker',str(shard)],
                                                  cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT))
        write(RUN/'status.json',{'status':'running','workers':[p.pid for p in processes],'driver':os.getpid()})
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Trace worker failed')
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError('Trace worker failed')
        write(RUN/'status.json',{'status':'complete'})
    except BaseException as exc:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        write(RUN/'status.json',{'status':'failed','error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=int,choices=(0,1))
    args=p.parse_args()
    main() if args.worker is None else worker(args.worker)
