"""Use class-normalized support in attention writes to choose a visual region."""

from unittest.mock import patch
import torch
from features.qwen3_region_guidance import Qwen3RegionGuidance,prefix_sha
from features.qwen3_region_memory import rectangle,LAYERS
from features import qwen3_region_memory as memory_module
from features import qwen3_region_consistency as consistency_module
from features.qwen3_bounded_visual_tilt import evidence_preference
from scripts.probe_qwen3_region_counterevidence import signed_attention


class Qwen3SupportedRivalGuidance(Qwen3RegionGuidance):
    def __init__(self,*args,policy='calibrated',**kwargs):
        assert policy in ['calibrated','calibrated_shifted']
        super().__init__(*args,policy='unverified',**kwargs)
        self.policy=policy
        self.class_maps=self.shared.setdefault('rival_class_maps',{})
        self.rival_choices={}

    def _class_map(self,row,canonical,cached):
        key=(row,canonical)
        if key not in self.class_maps:
            columns=[self.reader.columns[t] for t in self.aliases[canonical]]
            value=(cached[:,columns].expm1()/self.head.weight.shape[0]).sum(-1)
            self.class_maps[key]=(value,float(value.mean()))
        return self.class_maps[key]

    def _finish(self,module,args,output):
        if self.suspended or self.step==0 or self.disabled:return super()._finish(module,args,output)
        assert self.reader.step==self.step
        reference=output.last_hidden_state[:,-1:].clone()
        logits=self.head(reference).float()[:,0];tokens=logits.argmax(-1)
        eligible=self._eligible(tokens)
        if not eligible:return output
        features=self.reader.features(eligible,tokens[eligible].tolist());probability=self.probe(features)
        selected=[]
        for k,row in enumerate(eligible):
            p=float(probability[k]);positive=p<.418672114610672
            self.gates.setdefault(row,[]).append(dict(step=self.step,token=int(tokens[row]),p_real=p,
                threshold=.418672114610672,positive=positive,prefix_sha256=prefix_sha(self.history[row]),features=features[k].cpu().tolist()))
            if positive:selected.append(row)
        cache=self.tail_kwargs['past_key_values'];commits=[]
        for row in selected:
            token=int(tokens[row]);target=self.token_classes[token]
            prior=self.parser.classes(self.tokenizer.decode(self.history[row],skip_special_tokens=True,clean_up_tokenization_spaces=False))
            top=logits[row].topk(32).indices.tolist();names={target}
            for t in top:
                if t in self.nouns and self.token_classes[t] not in prior:names.add(self.token_classes[t])
            names=sorted(names);others=[c for c in names if c!=target]
            if not others:
                self.rival_choices.setdefault(row,[]).append(dict(step=self.step,target=target,rival=None,reason='No unseen alternative object class in native top32'))
                continue
            cached=self.reader._static(19,row)['log_s'];norms=self.reader.last_write_norms[row]
            weights=norms/norms.sum().clamp_min(1e-30)
            maps={c:self._class_map(row,c,cached) for c in names}
            lift={c:float((weights*value).sum())/max(mean,1e-30) for c,(value,mean) in maps.items()}
            logp=logits[row].log_softmax(0)
            native_mass={c:sum(float(logp[t].exp()) for t in top if self.token_classes.get(t)==c) for c in names}
            rival=max(others,key=lambda c:(native_mass[c]*lift[c],c))
            confidence=max(0.,lift[rival]-lift[target])/max(1e-30,lift[rival]+lift[target])
            choice=dict(step=self.step,target=target,rival=rival,lift=lift,native_class_mass=native_mass,
                confidence=confidence,prior_classes=sorted(prior))
            self.rival_choices.setdefault(row,[]).append(choice)
            q,p=evidence_preference(norms,cached[:,self.reader.columns[token]],self.reader.visual[19][row],self.reader.current[19]['update'][row])
            drift=float((1-(q*p).sqrt().sum()).clamp(0,1));strength=5*drift*confidence
            if self.zero_strength:strength=0.
            if not strength and not self.zero_strength:continue
            a,ma=maps[target];b,mb=maps[rival]
            heat=(b/max(mb,1e-30)-a/max(ma,1e-30)).clamp_min(0)*norms
            rect=rectangle(heat,*self.memory.grid(row),'shifted' if self.policy=='calibrated_shifted' else 'evidence')
            proposed_kv={};parent=memory_module.LastPositionCache
            class RecordingCache(parent):
                def update(cache_self,keys,values,layer_idx,cache_kwargs):
                    result=super().update(keys,values,layer_idx,cache_kwargs)
                    proposed_kv[layer_idx]=(keys[row:row+1].clone(),values[row:row+1].clone())
                    return result
            with patch.object(memory_module,'LastPositionCache',RecordingCache),patch.object(consistency_module,'mask_attention',signed_attention):
                hidden,mass_error=self.memory.attenuate(row,cache,self.reader,rect,-strength,reference,zero=self.zero_strength)
            assert set(proposed_kv)==set(LAYERS)
            final=self.head(hidden).float()[row,0]
            before=float(logp[token]);after=float(final.log_softmax(0)[token]);new=int(final.argmax())
            accepted=bool(new!=token and after<before and not self.reject_all)
            delta=max(float((proposed_kv[i][j]-getattr(cache.layers[i],name)[row:row+1,:,-1:]).abs().max())
                      for i in LAYERS for j,name in enumerate(['keys','values']))
            self.events.setdefault(row,[]).append(dict(step=self.step,token=token,proposed_token=new,accepted=accepted,
                before_logp=before,after_logp=after,actual_logp_change=after-before,drift=drift,
                support_confidence=confidence,strength=strength,rival_choice=choice,applied_rect=rect,
                visual_mass_relative_error_fp32=mass_error,proposed_current_kv_max_difference=delta,
                prefix_sha256=prefix_sha(self.history[row]),cache_position=int(self.tail_kwargs['cache_position'].item()),
                committed_layers=list(LAYERS) if accepted else [],policy=self.policy))
            if accepted:commits.append((row,hidden[row,0].clone(),proposed_kv))
        if commits:
            hidden=output.last_hidden_state.clone()
            for row,normalized,proposal in commits:
                hidden[row,-1]=normalized
                for i,(keys,values) in proposal.items():
                    cache.layers[i].keys[row:row+1,:,-1:]=keys
                    cache.layers[i].values[row:row+1,:,-1:]=values
            output.last_hidden_state=hidden
        return output

    def statistics(self,responses):
        stats=super().statistics(responses)
        for row,stat in enumerate(stats):
            stat.update(rival_choices=self.rival_choices.get(row,[]),class_support_normalized_by_image_mean=True,
                counterfactual_classifier_ranking=False,no_extra_crop_encoding=True)
        return stats

    def close(self):
        self.class_maps={}
        super().close()
