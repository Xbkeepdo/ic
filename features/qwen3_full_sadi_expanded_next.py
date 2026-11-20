"""Expand original adjacent feedback after complete SADI, preserving its math."""

from contextlib import contextmanager
from unittest.mock import patch
import torch

from features import qwen3_sar_ems_layer_feedback as original
from features.llava_sar_ems_attention import SarPatchTerms


class CompactSarPatchTerms(SarPatchTerms):
    """Keep selected original BF16 logits and the full-vocabulary normalizer."""
    @classmethod
    def from_full(cls,full,tokens):
        assert not isinstance(full,cls)
        obj=cls.__new__(cls);obj.tokens=tuple(sorted(tokens));obj.columns={t:i for i,t in enumerate(obj.tokens)}
        columns=torch.tensor(obj.tokens,device=full.visual_logits.device)
        obj.visual_logits=full.visual_logits[:,columns].contiguous()
        obj.gram=full.gram;obj.log_normalizer=full.log_normalizer;obj.log_vocab=full.log_vocab
        return obj

    def distribution(self,attention,token):
        return super().distribution(attention,self.columns[token])

    def log_semantics(self,tokens):
        columns=torch.tensor([self.columns[int(t)] for t in tokens.tolist()],device=tokens.device)
        return super().log_semantics(columns)


def identity(controller,source):
    weights=[controller.head.weight,controller.decoder.norm.weight,
        controller.decoder.layers[source].self_attn.o_proj.weight]
    return tuple((w.data_ptr(),w._version,tuple(w.shape),str(w.dtype)) for w in weights)


def metadata(controller,key):
    source,row=key
    return dict(identity=identity(controller,source),hidden=controller.visual_hpre[source][row].detach().clone(),
        values=controller.visual_values[source][row].detach().clone())


def seed_from_original(controller,shared):
    assert not shared.get('terms') and not shared.get('metadata')
    shared['terms']=dict(controller.patch_terms)
    shared['metadata']={key:metadata(controller,key) for key in controller.patch_terms}
    shared['seeded_terms']=len(controller.patch_terms)


def compact_shared(shared,tokens):
    before=after=0
    for key,terms in list(shared['terms'].items()):
        before+=terms.visual_logits.numel()*terms.visual_logits.element_size()
        if not isinstance(terms,CompactSarPatchTerms):shared['terms'][key]=CompactSarPatchTerms.from_full(terms,tokens)
        after+=shared['terms'][key].visual_logits.numel()*shared['terms'][key].visual_logits.element_size()
    return dict(before_logit_bytes=before,after_logit_bytes=after,terms=len(shared['terms']))


class Qwen3FullSadiExpandedNext(original.Qwen3BatchLayerFeedback):
    def __init__(self,model,span,nouns,*,targets=tuple(range(19,25)),strength=5.,shared=None,
                 compact=True,disable_guidance=False,force_guidance=False,zero_budget=False):
        targets=tuple(targets)
        assert targets and targets==tuple(range(targets[0],targets[-1]+1)) and targets[0]==19
        assert tuple(original.ATTENTION_LAYERS)==tuple(range(5,19))
        super().__init__(model,span,nouns,mode='next',strength=strength,disable_guidance=disable_guidance,
            force_guidance=force_guidance)
        self.targets=targets;self.sources=tuple(i-1 for i in targets)
        assert targets[-1]<len(self.decoder.layers)
        self.shared={} if shared is None else shared
        self.patch_terms=self.shared.setdefault('terms',{})
        self.shared.setdefault('metadata',{})
        self.compact=bool(compact);self.zero_budget=bool(zero_budget);self.checked=set()
        self.cache_hits=self.new_terms=0
        if force_guidance:
            assert not compact and not any(isinstance(t,CompactSarPatchTerms) for t in self.patch_terms.values())

    def _budget(self,source):
        existing={key for key in self.patch_terms if key[0]==source}
        for key in existing-self.checked:
            row=key[1];record=self.shared['metadata'][key]
            assert record['identity']==identity(self,source)
            assert torch.equal(record['hidden'],self.visual_hpre[source][row])
            assert torch.equal(record['values'],self.visual_values[source][row])
            terms=self.patch_terms[key]
            if isinstance(terms,CompactSarPatchTerms):assert terms.tokens==tuple(sorted(self.nouns))
            self.checked.add(key);self.cache_hits+=1
        rows,budget,details=super()._budget(source)
        for key in set(self.patch_terms)-set(self.shared['metadata']):
            self.shared['metadata'][key]=metadata(self,key);self.checked.add(key);self.new_terms+=1
        if self.zero_budget and budget is not None:
            budget=torch.zeros_like(budget);details=[dict(value,budget_max=0.) for value in details]
        return rows,budget,details

    def statistics(self,responses):
        result=super().statistics(responses)
        for row,(tokens,stats) in enumerate(zip(responses,result,strict=True)):
            events=[e for e in self.events.get(row,[]) if e['step']<len(tokens)]
            stats.update(full_events=events,source_layers=list(self.sources),target_layers=list(self.targets),
                unchanged_base_layers=list(range(5,19)),strength=self.strength,
                nominal_strength_sum=len(self.targets)*self.strength,
                diagnostic_zero_budget=self.zero_budget,disabled=not self.enabled,
                compact_semantic_cache=self.compact,shared_term_hits=self.cache_hits,new_term_builds=self.new_terms,
                executed_events=len(self.events.get(row,[])),unreported_terminal_events=len(self.events.get(row,[]))-len(events),
                accepted_guidance_calls=len(events),all_applied_during_original_single_forward=True)
        return result

    @contextmanager
    def capture(self):
        def factory(*args,**kwargs):
            full=SarPatchTerms(*args,**kwargs)
            return CompactSarPatchTerms.from_full(full,self.nouns) if self.compact else full
        with patch.object(original,'SarPatchTerms',factory):
            with super().capture():yield self

    def close(self):
        for name in ['scratch','pending','visual_hpre','visual_normalized','visual_values']:getattr(self,name).clear()
        self.patch_terms={};self.shared={}
