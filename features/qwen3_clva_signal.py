"""CLVA-inspired anchors after full SADI, with SAR/EMS-guided feedback.

Project adaptation to Qwen3: anchors0/16, actions19--35. The raw anchor
operator runs once per query; optional signal feedback replays only19--35
for actual final lexical object candidates. No trained detector is used.
"""

from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features.qwen3_sar_ems_final_target import LastPositionCache
from features.qwen3_online_risk_f import native_write_norms
from features.qwen3_bounded_visual_tilt import evidence_preference
from models.dgst_capture import apply_decoder_final_norm


def normalize_map(values):
    return values/values.sum(-1,keepdim=True).clamp_min(1e-30)


def anchor_map(attention,sensitive):
    intensity=attention.float().sum(-1)
    mean=intensity.mean(-1,keepdim=True);std=intensity.std(-1,correction=0,keepdim=True)
    selected=intensity>mean+std if sensitive else intensity<mean-std
    values=(attention.float()*selected[:,:,None]).sum(1)/selected.sum(-1,keepdim=True).clamp_min(1)
    return normalize_map(values),selected.sum(-1)


def salient(values):
    mean=values.mean(-1,keepdim=True);std=values.std(-1,correction=0,keepdim=True)
    return (values-mean)/(std+1e-8)>2.


def reanchor(probability,value,context,span,positive,negative,alpha,active):
    """Modify only last-query rows with nonempty masks, normalizing all keys."""
    start,end=span;rows=active.nonzero().flatten().tolist()
    if not rows:return context,probability,[]
    changed=context.clone();weights=probability.clone();mass=[]
    for row in rows:
        if not (positive[row].any() or negative[row].any()):continue
        p=probability[row:row+1,:,-1:].float();factor=torch.ones_like(p)
        factor[...,start:end]=1+float(alpha[row])*positive[row].float()-.9*negative[row].float()
        assert float(factor.min())>=.099
        proposed=p*factor;proposed/=proposed.sum(-1,keepdim=True).clamp_min(1e-30)
        mass.append((row,float(p[...,start:end].sum(-1).mean()),float(proposed[...,start:end].sum(-1).mean())))
        rounded=proposed.to(probability.dtype)
        changed[row:row+1,-1:]=(rounded@value[row:row+1]).transpose(1,2).contiguous()
        weights[row:row+1,:,-1:]=rounded
    return changed,weights,mass


class Qwen3CLVASignal:
    MODES=('clva','clva_fixed','clva_drift','clva_sar')
    def __init__(self,wrapper,span,nouns,*,mode='clva',shared_static=None,disabled=False,
                 zero_guidance=False,reject_all=False):
        assert mode in self.MODES
        self.wrapper,self.model=wrapper,wrapper.model;self.decoder=self.model.model.language_model
        self.head=self.model.get_output_embeddings();self.start,self.end=span
        self.nouns=frozenset(nouns);self.tokens=tuple(sorted(nouns));self.columns={t:i for i,t in enumerate(self.tokens)}
        self.mode=mode;self.disabled=bool(disabled);self.zero_guidance=bool(zero_guidance);self.reject_all=bool(reject_all)
        self.shared={} if shared_static is None else shared_static
        self.static={};self.step=-1;self.replaying=False;self.active=None
        self.visual=self.values=self.attention=self.update=None
        self.anchor=self.tail_kwargs=self.reference=None
        self.positive=self.negative=self.positive_mask=self.negative_mask=None
        self.proposal_alpha=self.proposal_mask=None;self.events={}
        self.head_calls=self.static_hits=self.feedback_calls=self.encoder_calls=self.vision_cache_hits=0
        self.raw_layer_steps={};self.mass_samples={};self.head_counts={}
        eos=self.model.generation_config.eos_token_id;self.eos=tuple(eos if isinstance(eos,list) else [eos])

    def _begin(self,_m,args,kwargs):
        assert not self.replaying
        self.step+=1;ids=kwargs.get('input_ids',args[0] if args else None);assert ids is not None
        if self.active is None:self.active=torch.ones(len(ids),dtype=torch.bool,device=ids.device)
        else:
            for row,t in enumerate(ids[:,-1].tolist()):
                if t in self.eos:self.active[row]=False
        self.anchor=self.tail_kwargs=self.reference=None

    def _input16(self,_m,args,kwargs):
        if self.replaying:return
        h=args[0] if args else kwargs['hidden_states']
        if self.step==0:self.visual=h[:,self.start:self.end].detach().clone()

    def _ffn16(self,_m,_args,output):
        if not self.replaying:self.update=output[:,-1].detach().clone()

    def _anchor18(self,_m,_args,output):
        if not self.replaying:self.anchor=output[:,-1:].clone()

    def _tail19(self,_m,_args,kwargs):
        if not self.replaying:self.tail_kwargs=dict(kwargs)

    def _decoder(self,_m,_args,output):
        if not self.replaying:self.reference=output.last_hidden_state[:,-1:].clone()

    def _static16(self,row):
        if row in self.static:return self.static[row]
        hidden,values=self.visual[row],self.values[row]
        identity=(self.tokens,self.start,self.end,self.head.weight.data_ptr(),self.head.weight._version)
        if row in self.shared:
            entry=self.shared[row];assert entry['identity']==identity
            assert torch.equal(entry['hidden'],hidden) and torch.equal(entry['values'],values)
            self.static_hits+=1
        else:
            normalized=apply_decoder_final_norm(self.decoder.norm,hidden)
            columns=torch.tensor(self.tokens,device=hidden.device);parts=[]
            for offset in range(0,len(hidden),64):
                logits=self.head(normalized[offset:offset+64]).float()
                selected=(logits[:,columns]-logits.logsumexp(-1,keepdim=True)).clamp_max(0).exp()
                parts.append(torch.log1p(logits.shape[-1]*selected));self.head_calls+=1
            entry=dict(identity=identity,hidden=hidden,values=values,log_s=torch.cat(parts))
            self.shared[row]=entry
        self.static[row]=entry;return entry

    def _finish(self,_module,_args,out):
        if self.disabled or self.mode=='clva' or self.step==0:return out
        logits=out.logits[:,-1].float();tokens=logits.argmax(-1)
        rows=[r for r,t in enumerate(tokens.tolist()) if self.active[r] and t in self.nouns]
        if not rows:return out
        assert self.anchor is not None and self.tail_kwargs is not None
        alpha=logits.new_full((len(tokens),),14.);masks=self.positive_mask.clone();records=[]
        for row in rows:
            token=int(tokens[row]);drift=.5;changed_mask=0
            if self.mode!='clva_fixed':
                values=qwen.repeat_kv(self.values[row:row+1],self.decoder.layers[16].self_attn.num_key_value_groups)[0]
                norms=native_write_norms(values,self.attention[row],self.decoder.layers[16].self_attn.o_proj.weight)
                log_s=self._static16(row)['log_s'][:,self.columns[token]]
                q,p=evidence_preference(norms,log_s,self.visual[row],self.update[row])
                drift=float((1-(q*p).sqrt().sum()).clamp(0,1))
            applied=0. if self.zero_guidance else drift
            alpha[row]=14*(1+applied)
            if self.mode=='clva_sar' and applied:
                masks[row]=salient((1-applied)*self.positive[row]+applied*q)
                changed_mask=int((masks[row]!=self.positive_mask[row]).sum())
            records.append(dict(step=self.step,token=token,D_native=drift,D_applied=applied,alpha=float(alpha[row]),
                positive_mask_changed_positions=changed_mask,before_logp=float(logits[row].log_softmax(0)[token]),
                mode=self.mode,committed=not self.reject_all))
        native=self.tail_kwargs['past_key_values'];layers=list(range(19,36))
        backup={i:(native.layers[i].keys[rows,:,-1:].clone(),native.layers[i].values[rows,:,-1:].clone()) for i in layers}
        cache=LastPositionCache(native,rows,check_identical=self.zero_guidance)
        kwargs=dict(self.tail_kwargs,past_key_values=cache);current=self.anchor.clone()
        old_alpha,old_mask=self.proposal_alpha,self.proposal_mask
        self.replaying=True;self.proposal_alpha,self.proposal_mask=alpha,masks
        try:
            for layer in layers:current=self.decoder.layers[layer](current,**kwargs)
            normalized=self.decoder.norm(current);assert cache.updates==layers
            others=[r for r in range(len(tokens)) if r not in rows]
            if others:assert torch.equal(normalized[others],self.reference[others])
            if self.zero_guidance:assert torch.equal(normalized,self.reference)
            final=self.head(normalized)
            if self.zero_guidance:assert torch.equal(final[:,0],out.logits[:,-1])
            for row,record in zip(rows,records,strict=True):
                new=int(final[row,0].argmax());record.update(proposed_token=new,emitted_token=int(tokens[row]) if self.reject_all else new,
                    token_changed=not self.reject_all and new!=int(tokens[row]),
                    after_logp=float(final[row,0].float().log_softmax(0)[int(tokens[row])]),
                    committed_layers=[] if self.reject_all else layers)
                self.events.setdefault(row,[]).append(record)
            if self.reject_all:
                for i,(keys,values) in backup.items():
                    native.layers[i].keys[rows,:,-1:]=keys;native.layers[i].values[rows,:,-1:]=values
            else:
                modified=out.logits.clone();modified[rows,-1]=final[rows,0];out.logits=modified
            self.feedback_calls+=1
        except BaseException:
            for i,(keys,values) in backup.items():native.layers[i].keys[rows,:,-1:]=keys;native.layers[i].values[rows,:,-1:]=values
            raise
        finally:
            self.proposal_alpha,self.proposal_mask=old_alpha,old_mask;self.replaying=False
        return out

    @contextmanager
    def capture(self):
        original_vision=self.model.model.visual.forward
        def vision(pixels,grid_thw,**kwargs):
            entry=self.shared.get('_vision')
            if entry is not None:
                assert torch.equal(entry['pixels'],pixels) and torch.equal(entry['grid'],grid_thw)
                self.vision_cache_hits+=1;return entry['output']
            result=original_vision(pixels,grid_thw,**kwargs)
            self.shared['_vision']=dict(pixels=pixels.detach().clone(),grid=grid_thw.detach().clone(),
                output=(result[0].detach().clone(),[x.detach().clone() for x in result[1]]))
            return result
        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(self.model,(self.start,self.end),frozenset(),'sadi').capture())
            stack.enter_context(patch.object(self.model.model.visual,'forward',vision))
            original=qwen.eager_attention_forward
            def attention(module,query,key,value,*args,**kwargs):
                context,probability=original(module,query,key,value,*args,**kwargs)
                if not isinstance(module,qwen.Qwen3VLTextAttention):return context,probability
                layer=module.layer_idx
                if not self.replaying and layer in [0,16]:
                    a=probability[:,:,-1,self.start:self.end].contiguous()
                    anchor,counts=anchor_map(a,sensitive=layer==16)
                    if layer==0:self.negative=anchor;self.negative_mask=salient(anchor)
                    else:
                        self.positive=anchor;self.positive_mask=salient(anchor);self.attention=a
                        if self.step==0:self.values=value[:,:,self.start:self.end].detach().clone()
                    self.head_counts[layer]=counts.tolist()
                if self.disabled or layer<19:return context,probability
                positive=self.proposal_mask if self.replaying else self.positive_mask
                alpha=self.proposal_alpha if self.replaying else query.new_full((len(query),),14.,dtype=torch.float32)
                expanded=qwen.repeat_kv(value,module.num_key_value_groups)
                changed,weights,mass=reanchor(probability,expanded,context,(self.start,self.end),positive,self.negative_mask,alpha,self.active)
                if not self.replaying:
                    for row,before,after in mass:
                        self.raw_layer_steps[row]=self.raw_layer_steps.get(row,0)+1
                        count,old,new=self.mass_samples.get(row,(0,0.,0.));self.mass_samples[row]=(count+1,old+before,new+after)
                return changed,weights
            stack.enter_context(patch.object(qwen,'eager_attention_forward',attention))
            stack.callback(self.model.register_forward_pre_hook(self._begin,with_kwargs=True).remove)
            stack.callback(self.decoder.layers[16].register_forward_pre_hook(self._input16,with_kwargs=True).remove)
            stack.callback(self.decoder.layers[16].mlp.register_forward_hook(self._ffn16).remove)
            stack.callback(self.decoder.layers[18].register_forward_hook(self._anchor18).remove)
            stack.callback(self.decoder.layers[19].register_forward_pre_hook(self._tail19,with_kwargs=True).remove)
            stack.callback(self.decoder.register_forward_hook(self._decoder).remove)
            stack.callback(self.model.register_forward_hook(self._finish).remove)
            stack.callback(self.model.model.visual.blocks[0].register_forward_hook(lambda *_:setattr(self,'encoder_calls',self.encoder_calls+1)).remove)
            yield self

    def statistics(self,responses):
        result=[]
        for row,tokens in enumerate(responses):
            events=self.events.get(row,[])
            assert all(e['step']<len(tokens) and tokens[e['step']]==e['emitted_token'] for e in events)
            count,before,after=self.mass_samples.get(row,(0,0.,0.))
            result.append(dict(mode=self.mode,full_events=events,accepted=sum(e['committed'] for e in events),
                guided_attention_calls=len(events),accepted_guidance_calls=sum(e['committed'] for e in events),
                token_changes=sum(e['token_changed'] for e in events),raw_clva_layer_steps=self.raw_layer_steps.get(row,0),
                mean_raw_visual_mass_before=before/max(1,count),mean_raw_visual_mass_after=after/max(1,count),
                source_positive_layer=16,source_negative_layer=0,action_layers=list(range(19,36)),unchanged_base_layers=list(range(5,19)),
                diagnostic_disabled=self.disabled,diagnostic_zero_guidance=self.zero_guidance,diagnostic_reject_all=self.reject_all,
                shared_static_hits=self.static_hits,static_head_calls=self.head_calls,feedback_calls_in_batch=self.feedback_calls,
                model_steps=self.step+1,encoder_calls_in_batch=self.encoder_calls,vision_cache_hits=self.vision_cache_hits,
                no_trained_detector=True,no_history_repeat_filter=True,
                feedback_acceptance='Commit all selected lexical-object feedback rows; no posterior probability-decrease gate.'))
        return result

    def close(self):
        self.static.clear();self.visual=self.values=self.attention=self.update=None
        self.anchor=self.tail_kwargs=self.reference=None
