"""Isolate immediate-token acceptance in the frozen semantic-ROI operator.

The source, first-class risk gate, budget5D, native values, tail20--35 and
full SADI5--18 are unchanged. A commit is execution, never a quality label.
"""

import math
from unittest.mock import patch

from features.qwen3_region_guidance import Qwen3RegionGuidance,prefix_sha
from features import qwen3_region_memory as memory_module
from features import qwen3_region_consistency as consistency_module
from features.qwen3_region_memory import rectangle,LAYERS
from features.qwen3_region_consistency import category_probabilities
from features.qwen3_bounded_visual_tilt import evidence_preference
from scripts.probe_qwen3_region_counterevidence import signed_attention


def should_commit(mode,before,after,token,proposed,*,zero=False,reject=False):
    assert mode in ['wordchange','probability','direct','shifted']
    assert math.isfinite(before) and math.isfinite(after)
    if zero or reject:return False
    if mode=='wordchange':return proposed!=token and after<before
    if mode=='probability':return after<before
    return True


class Qwen3RegionCommit(Qwen3RegionGuidance):
    def __init__(self,*args,commit_mode='direct',**kwargs):
        assert commit_mode in ['wordchange','probability','direct','shifted']
        assert 'policy' not in kwargs
        super().__init__(*args,policy='unverified',**kwargs)
        self.commit_mode=commit_mode

    def _finish(self,_module,_args,output):
        if self.suspended or self.step==0 or self.disabled:
            return super()._finish(_module,_args,output)
        assert self.reader.step==self.step
        reference=output.last_hidden_state[:,-1:].clone()
        logits=self.head(reference).float()[:,0];tokens=logits.argmax(-1)
        eligible=self._eligible(tokens)
        if not eligible:return output
        features=self.reader.features(eligible,tokens[eligible].tolist())
        probability=self.probe(features);selected=[]
        for index,row in enumerate(eligible):
            value=float(probability[index]);positive=value<.418672114610672
            self.gates.setdefault(row,[]).append(dict(step=self.step,token=int(tokens[row]),p_real=value,
                threshold=.418672114610672,positive=positive,prefix_sha256=prefix_sha(self.history[row]),
                features=features[index].cpu().tolist()))
            if positive:selected.append(row)
        if not selected:return output
        native_cache=self.tail_kwargs['past_key_values'];commits=[]
        for row in selected:
            token=int(tokens[row]);aliases=self.aliases[self.token_classes[token]]
            cached=self.reader._static(19,row);norms=self.reader.last_write_norms[row]
            q,p=evidence_preference(norms,cached['log_s'][:,self.reader.columns[token]],
                self.reader.visual[19][row],self.reader.current[19]['update'][row])
            drift=float((1-(q*p).sqrt().sum()).clamp(0,1))
            semantic=category_probabilities(self.reader,row,aliases)
            rect=rectangle(semantic,*self.memory.grid(row),'evidence')
            applied=rectangle(semantic,*self.memory.grid(row),'shifted') if self.commit_mode=='shifted' else rect
            strength=0. if self.zero_strength else 5*drift
            if not strength and not self.zero_strength:continue
            proposed_kv={};parent=memory_module.LastPositionCache
            class RecordingCache(parent):
                def update(cache_self,keys,values,layer_idx,cache_kwargs):
                    result=super().update(keys,values,layer_idx,cache_kwargs)
                    proposed_kv[layer_idx]=(keys[row:row+1].clone(),values[row:row+1].clone())
                    return result
            with patch.object(memory_module,'LastPositionCache',RecordingCache),\
                 patch.object(consistency_module,'mask_attention',signed_attention):
                hidden,mass_error=self.memory.attenuate(row,native_cache,self.reader,
                    applied,-strength,reference,zero=self.zero_strength)
            assert set(proposed_kv)==set(LAYERS)
            current_logits=self.head(hidden).float()[:,0]
            before=float(logits[row].log_softmax(0)[token]);after=float(current_logits[row].log_softmax(0)[token])
            new=int(current_logits[row].argmax())
            accepted=should_commit(self.commit_mode,before,after,token,new,zero=self.zero_strength,reject=self.reject_all)
            delta=max(float((proposed_kv[index][j]-getattr(native_cache.layers[index],name)[row:row+1,:,-1:]).abs().max())
                for index in LAYERS for j,name in enumerate(['keys','values']))
            self.events.setdefault(row,[]).append(dict(step=self.step,token=token,proposed_token=new,accepted=accepted,
                before_logp=before,after_logp=after,actual_logp_change=after-before,drift=drift,
                reliability=dict(reliability_deficit=1.,skipped=True),strength=strength,rect=rect,applied_rect=applied,
                visual_mass_relative_error_fp32=mass_error,proposed_current_kv_max_difference=delta,
                prefix_sha256=prefix_sha(self.history[row]),cache_position=int(self.tail_kwargs['cache_position'].item()),
                committed_layers=list(LAYERS) if accepted else [],policy='unverified',commit_mode=self.commit_mode,
                same_token_commit=accepted and new==token))
            if accepted:commits.append((row,hidden[row,0].clone(),proposed_kv))
        # Every proposal above uses the same pristine native batch state.
        if commits:
            hidden=output.last_hidden_state.clone()
            for row,normalized,proposed in commits:
                hidden[row,-1]=normalized
                for index,(keys,values) in proposed.items():
                    native_cache.layers[index].keys[row:row+1,:,-1:]=keys
                    native_cache.layers[index].values[row:row+1,:,-1:]=values
            output.last_hidden_state=hidden
        return output

    def statistics(self,responses):
        result=super().statistics(responses)
        for stats in result:
            stats.update(commit_mode=self.commit_mode,
                same_token_commits=sum(e['same_token_commit'] for e in stats['full_events']),
                guided_attention_calls=len(stats['full_events']),accepted_guidance_calls=stats['accepted'],
                acceptance_is_execution_not_quality=True)
            assert stats['crop_encoder_calls_in_batch']==stats['fine_prefill_calls_in_batch']==0
        return result
