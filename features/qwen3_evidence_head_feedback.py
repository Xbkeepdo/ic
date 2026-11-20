"""Use source-layer SAR/FFN evidence to route next-layer visual head mass."""

from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_bounded_visual_tilt import original_sam_rows, evidence_preference
from features.qwen3_causal_visual_tilt import Qwen3BatchCausalVisualTilt, SOURCES, TARGETS
from features.qwen3_evidence_head_routing import head_routing_attention, evidence_head_scores
from features.qwen3_online_risk_f import native_write_norms
from features.qwen3_sar_ems_final_target import LastPositionCache


class Qwen3BatchEvidenceHeadFeedback(Qwen3BatchCausalVisualTilt):
    def __init__(self, *args, head_mode='evidence', diagnostic_flat_head_score=False, **kwargs):
        assert kwargs.pop('tilt_mode', 'evidence') == 'evidence'
        super().__init__(*args, tilt_mode='evidence', **kwargs)
        assert self.gamma in (0.,1.) and head_mode in ('uniform','evidence','roll')
        self.head_mode=head_mode
        self.flat_head_score=bool(diagnostic_flat_head_score)
        self.maximum_fp32_mean_mass_difference=0.
        self.maximum_bf16_mean_mass_difference=0.
        self.maximum_conditional_visual_difference=0.

    def _source_distributions(self, source, rows, attention=None, update=None):
        distributions=[]
        for row in rows:
            if attention is None:
                norms=self.reader.write_norms[source,row]
                actual_update=self.scratch[source]['update'][row]
            else:
                assert update is not None
                values=self.reader.values[source][row:row+1]
                groups=attention.shape[1]//values.shape[1]
                norms=native_write_norms(qwen.repeat_kv(values,groups)[0],attention[row],
                    self.decoder.layers[source].self_attn.o_proj.weight)
                actual_update=update[row];self.refreshed_norm_calls+=1
            cached=self.reader._static(source,row)
            log_s=cached['log_s'][:,self.reader.columns[int(self.final_tokens[row])]]
            distributions.append(evidence_preference(norms,log_s,self.reader.visual[source][row],actual_update))
        return tuple(torch.stack([pair[i] for pair in distributions]) for i in (0,1))

    def _budget(self,_source):
        rows,real=original_sam_rows(self)
        if not rows or not self.gamma:
            return [],None,[]
        native={target:self._source_distributions(target-1,rows) for target in TARGETS}
        details=[dict(token=int(self.final_tokens[row]),gain=0.,drift=0.,budget_max=0.,
            budget_units='constant_visual_logit_bias_per_head',probe_p_real=float(real[i]),
            probe_real_threshold=self.real_threshold,signal_guided=False,
            joint_layers=list(TARGETS),layer_details={},bounded_visual_tilt=True,
            causal_visual_tilt=True,evidence_head_routing=True,tilt_mode='evidence',
            head_mode=self.head_mode,refresh_source=self.refresh,
            same_final_target_throughout_replay=True) for i,row in enumerate(rows)]
        return rows,native,details

    def _replay(self,rows,native):
        current=self.source_outputs[19].clone();old_enabled=self.enabled;self.enabled=False
        self.replaying_tail=self._replaying=self._causal_replaying=True
        self._fresh_attention.clear();self._fresh_updates.clear()
        cache=LastPositionCache(self.tail_kwargs['past_key_values'],rows,check_identical=self.zero_tilt)
        kwargs=dict(self.tail_kwargs,past_key_values=cache)
        original=qwen.eager_attention_forward;active={};layer_records={}

        def inject(module,query,key,value,attention_mask,scaling,dropout=0.,**kw):
            index=getattr(module,'layer_idx',None)
            if not isinstance(module,qwen.Qwen3VLTextAttention) or index not in TARGETS:
                return original(module,query,key,value,attention_mask,scaling,dropout,**kw)
            assert index==active['target'] and not dropout and not module.training
            key=qwen.repeat_kv(key,module.num_key_value_groups)
            value=qwen.repeat_kv(value,module.num_key_value_groups)
            scores=(query@key.transpose(2,3))*scaling
            if attention_mask is not None:
                scores=scores+attention_mask[...,:key.shape[-2]]
            output,probability,diagnostic=head_routing_attention(scores,value,(self.start,self.end),rows,
                active['q'],active['p'],mode=self.head_mode,zero_tilt=self.zero_tilt,
                flat_score=self.flat_head_score)
            visual=scores[rows,:,0,self.start:self.end].float()
            nonvisual=torch.cat((scores[rows,:,0,:self.start],scores[rows,:,0,self.end:]),-1).float()
            diagnostic['visual_log_odds']=visual.logsumexp(-1)-nonvisual.logsumexp(-1)
            native_score,_=evidence_head_scores(visual,*active['native'])
            diagnostic['native_source_head_score_max_difference']=(diagnostic['raw_head_score']-native_score).abs().amax(-1)
            self.maximum_fp32_mean_mass_difference=max(self.maximum_fp32_mean_mass_difference,
                float(diagnostic['fp32_mean_visual_mass_difference'].max()))
            self.maximum_bf16_mean_mass_difference=max(self.maximum_bf16_mean_mass_difference,
                float(diagnostic['bf16_mean_visual_mass_difference'].max()))
            self.maximum_conditional_visual_difference=max(self.maximum_conditional_visual_difference,
                float(diagnostic['conditional_visual_difference'].max()))
            if index!=TARGETS[-1]:
                self._fresh_attention[index]=probability[:,:,0,self.start:self.end].detach()
            active['diagnostic']=diagnostic;self.causal_layer_calls+=1
            return output,probability

        try:
            with patch.object(qwen,'eager_attention_forward',inject):
                for target in TARGETS:
                    source=target-1;q0,p0=native[target]
                    attention_difference=q0.new_zeros(len(rows));update_difference=attention_difference.clone()
                    if source==SOURCES[0]:
                        q,p=q0,p0
                    else:
                        attention=self._fresh_attention.pop(source);update=self._fresh_updates.pop(source)
                        attention_difference=(attention[rows].float()-self.reader.current[source]['attention'][rows].float()).abs().amax((1,2))
                        update_difference=(update[rows].float()-self.scratch[source]['update'][rows].float()).abs().amax(-1)
                        q,p=self._source_distributions(source,rows,attention,update) if self.refresh else (q0,p0)
                    active=dict(target=target,q=q,p=p,native=(q0,p0))
                    current=self.decoder.layers[target](current,**kwargs)
                    d=active['diagnostic'];bias=d['visual_bias']
                    records=[]
                    for i in range(len(rows)):
                        masses=d['reference_visual_mass'][i]
                        records.append(dict(source_layer=source,target_layer=target,
                            source_refreshed=self.refresh and source!=SOURCES[0],
                            drift=float(d['drift'][i]),evidence_head_routing=True,head_mode=self.head_mode,
                            raw_head_score=d['raw_head_score'][i].tolist(),
                            applied_head_score=d['applied_head_score'][i].tolist(),
                            visual_bias=bias[i].tolist(),reference_bias=d['reference_bias'][i].tolist(),
                            visual_log_odds=d['visual_log_odds'][i].tolist(),matching_offset=float(d['offset'][i]),
                            flat_or_saturated=bool(d['applied_head_score'][i].max()==d['applied_head_score'][i].min()
                                or (masses*(1-masses)).sum()==0),
                            visual_logit_shift_min=float(bias[i].min()),visual_logit_shift_max=float(bias[i].max()),
                            budget_max=float(bias[i].abs().max()),
                            source_q_max_difference=float((q[i]-q0[i]).abs().max()),
                            source_p_max_difference=float((p[i]-p0[i]).abs().max()),
                            source_attention_max_difference=float(attention_difference[i]),
                            source_ffn_max_difference=float(update_difference[i]),
                            native_source_head_score_max_difference=float(d['native_source_head_score_max_difference'][i]),
                            fp32_mean_visual_mass_difference=float(d['fp32_mean_visual_mass_difference'][i]),
                            bf16_mean_visual_mass_difference=float(d['bf16_mean_visual_mass_difference'][i]),
                            conditional_visual_difference=float(d['conditional_visual_difference'][i]),
                            maximum_individual_head_mass_change=float(d['maximum_individual_head_mass_change'][i])))
                    layer_records[target]=records
            assert cache.updates==list(TARGETS) and not self._fresh_attention and not self._fresh_updates
            normalized=self.decoder.norm(current)
            unselected=[row for row in range(len(current)) if row not in rows]
            if unselected:
                assert torch.equal(normalized[unselected],self.reference_hidden[unselected])
            if self.zero_tilt:
                difference=float((normalized-self.reference_hidden).abs().max())
                self.maximum_zero_replay_hidden_difference=max(self.maximum_zero_replay_hidden_difference,difference)
                assert difference==0;self.zero_replay_checks+=1
            self._causal_layer_records=layer_records
            return normalized
        finally:
            self.enabled=old_enabled
            self.replaying_tail=self._replaying=self._causal_replaying=False
            self._fresh_attention.clear();self._fresh_updates.clear()

    def statistics(self,responses):
        result=super().statistics(responses)
        for stat in result:
            stat.update(evidence_head_routing=True,head_mode=self.head_mode,
                diagnostic_flat_head_score=self.flat_head_score,
                maximum_fp32_mean_mass_difference=self.maximum_fp32_mean_mass_difference,
                maximum_bf16_mean_mass_difference=self.maximum_bf16_mean_mass_difference,
                maximum_conditional_visual_difference=self.maximum_conditional_visual_difference,
                matching_scope='Per-image mean visual mass across heads at the same actual query.',
                routing_preserves_conditional_visual_pattern=True,
                transport_mode='adjacent_evidence_head_visual_mass_routing',
                borrowed_operator='Head-selection idea informed by ACT/AHEAD; this is a project adaptation, not their full methods.',
                acg_scope='Original108 gate and refreshed source Q/P; head-wise constant visual biases, no gradients.')
        return result
