"""Keep complete SADI; gate actual text routes in native-conflict late heads.

This is a local CRG Conflict-B adaptation, not a reproduction of full CRG.
Unlike the old layer31 history-route experiment, this covers19--35, keeps
the same gate multiset across VRI/signal/roll, and does not use the probe.
"""

import torch
from unittest.mock import patch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_block_acg_signal import POST
from features.qwen3_block_acg_headroll import Qwen3BatchBlockACGLogged
from features.qwen3_bounded_visual_tilt import evidence_preference
from features.qwen3_conflict_route_gates import (
    split_head_routes,preference_over_evidence,conflict_text_gates,apply_text_gates)
from features.qwen3_online_risk_f import native_write_norms
from features.qwen3_sar_ems_final_target import LastPositionCache


class ForbiddenRouteProbe(torch.nn.Module):
    def forward(self,*_args,**_kwargs):
        raise AssertionError('This route adaptation must not call a learned probe')


class Qwen3BatchConflictRoute(Qwen3BatchBlockACGLogged):
    def __init__(self,model,span,nouns,*,tokenizer,history_parser,shared_static=None,
                 route_mode='signal',k=8,gamma=1.,diagnostic_zero_route=False,
                 diagnostic_flat_signal=False,**kwargs):
        assert route_mode in ('vri','signal','roll') and gamma in (0.,1.)
        assert not kwargs.get('diagnostic_zero_replay',False)
        super().__init__(model,span,nouns,tokenizer=tokenizer,history_parser=history_parser,
            probe=ForbiddenRouteProbe(),semantic_tokens=sorted(nouns),real_threshold=.5,
            shared_static=shared_static,signal_guided=True,gamma=gamma,**kwargs)
        self.route_mode=route_mode;self.route_k=int(k)
        self.zero_route=bool(diagnostic_zero_route);self.flat_signal=bool(diagnostic_flat_signal)
        self.route_signal_calls=0;self.route_layer_calls=0
        self.maximum_split_roundoff=0.;self.maximum_replayed_text_route_difference=0.
        self._native_text_routes={}

    def _eligible_rows(self):
        assert not self.force_probe_gate
        rows=[]
        for row,(token,active) in enumerate(zip(self.final_tokens.tolist(),self.active.tolist(),strict=True)):
            if not active or token not in self.nouns or token in self.seen[row]:continue
            assert len(self.output_history[row])==self.step
            prefix=self.tokenizer.decode(self.output_history[row],skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            prior=self.history_parser.classes(prefix);canonical=self.token_classes[token]
            blocked=canonical in prior
            self.class_decisions.setdefault(row,[]).append(dict(step=self.step,token=token,
                canonical=canonical,prior_classes=sorted(prior),blocked=blocked))
            if blocked:continue
            rows.append(row)
            self.gate_events.setdefault(row,[]).append(dict(step=self.step,token=token,
                p_real=None,threshold=None,positive=True,gate_source='lexical_first_class_only'))
        return rows

    def _risk_for_heads(self,index,rows,attention):
        risks=[]
        for row in rows:
            cached=self.reader._static(index,row)
            values=self.reader.values[index][row:row+1]
            groups=attention.shape[1]//values.shape[1]
            norms=native_write_norms(qwen.repeat_kv(values,groups)[0],
                attention[row,:,0,self.start:self.end],self.decoder.layers[index].self_attn.o_proj.weight)
            log_s=cached['log_s'][:,self.reader.columns[int(self.final_tokens[row])]]
            q,p=evidence_preference(norms,log_s,self.reader.visual[index][row],self.scratch[index]['update'][row])
            visual=attention[row,:,0,self.start:self.end].float()
            conditional=visual/visual.sum(-1,keepdim=True).clamp_min(torch.finfo(visual.dtype).tiny)
            risks.append(preference_over_evidence(conditional[None],q[None],p[None])[0])
            self.route_signal_calls+=1
        return torch.stack(risks)

    def _budget(self,_source):
        rows=self._eligible_rows()
        if not rows or not self.gamma:return [],None,[]
        self._block_gradient(rows)
        allgates={};records={};native_text={}
        self.first_context=None
        for index in POST:
            state=self.scratch[index]
            context,visual,textual,attention=split_head_routes(state['scores'],state['values'],(self.start,self.end))
            if index==19:self.first_context=context
            self.maximum_split_roundoff=max(self.maximum_split_roundoff,float((context.float()-visual.float()-textual.float()).abs().max()))
            projected=(self.block_gradients[index].float()@self.decoder.layers[index].self_attn.o_proj.weight.float()).reshape(len(rows),context.shape[1],-1)
            vis=(visual[rows].float()*projected).sum(-1)
            txt=(textual[rows].float()*projected).sum(-1)
            risk=torch.full_like(vis,.5)
            conflict_rows=((vis<0)&(txt>0)).any(-1).nonzero().flatten().tolist()
            if self.route_mode!='vri' and not self.flat_signal and conflict_rows:
                actual_rows=[rows[i] for i in conflict_rows]
                risk[conflict_rows]=self._risk_for_heads(index,actual_rows,attention)
            gates,metrics=conflict_text_gates(vis,txt,risk,mode=self.route_mode,k=self.route_k)
            if self.zero_route:gates=torch.ones_like(gates)
            allgates[index]=gates;native_text[index]=textual[rows].clone()
            predicted=((gates-1)*txt).sum(-1)
            records[index]=[dict(source_layer=index,target_layer=index,native_point=True,
                route_mode=self.route_mode,visual_effect=vis[i].tolist(),textual_effect=txt[i].tolist(),
                native_conflict_mask=metrics['conflict'][i].tolist(),vri=metrics['vri'][i].tolist(),
                raw_head_risk=risk[i].tolist(),applied_head_risk=metrics['applied_head_risk'][i].tolist(),
                text_gates=gates[i].tolist(),selected_heads=(gates[i]<1).nonzero().flatten().tolist(),
                predicted_log_probability_change=float(predicted[i]),
                budget_max=float(((gates[i]-1)[:,None]*textual[rows[i]].float()).abs().max()),
                native_text_route_norm=float(textual[rows[i]].float().norm())) for i in range(len(rows))]
        maximum=torch.stack([torch.tensor([r['budget_max'] for r in records[index]],device=self.final_tokens.device) for index in POST]).amax(0)
        effective=list(range(len(rows))) if self.zero_route else (maximum>0).nonzero().flatten().tolist()
        if not effective:return [],None,[]
        self._native_text_routes={index:value[effective] for index,value in native_text.items()}
        details=[dict(token=int(self.final_tokens[rows[i]]),gain=0.,drift=0.,
            budget_max=float(maximum[i]),probe_p_real=None,probe_real_threshold=None,
            signal_guided=True,no_learned_gate=True,route_mode=self.route_mode,
            joint_layers=list(POST),layer_details={str(index):records[index][i] for index in POST},
            predicted_log_probability_change=sum(records[index][i]['predicted_log_probability_change'] for index in POST)) for i in effective]
        return [rows[i] for i in effective],{index:gates[effective] for index,gates in allgates.items()},details

    def _replay(self,rows,gates):
        layer=self.decoder.layers[19]
        changed=apply_text_gates(self.first_context[rows],self._native_text_routes[19],gates[19])
        context=self.first_context.clone();context[rows]=changed
        attended=self.scratch[19]['residual']+layer.self_attn.o_proj(context[:, :, None].transpose(1,2).contiguous().flatten(2))
        old_enabled=self.enabled;self.enabled=False
        self.replaying_tail=self._replaying=True
        try:
            corrected=attended+layer.mlp(layer.post_attention_layernorm(attended))
            current=self.source_outputs[19].clone();current[rows]=corrected[rows]
            cache=LastPositionCache(self.tail_kwargs['past_key_values'],rows,check_identical=self.zero_route)
            kwargs=dict(self.tail_kwargs,past_key_values=cache)
            original=qwen.eager_attention_forward
            seen=[]
            def inject(module,query,key,value,attention_mask,scaling,dropout=0.,**kw):
                output,attention=original(module,query,key,value,attention_mask,scaling,dropout,**kw)
                index=getattr(module,'layer_idx',None)
                if isinstance(module,qwen.Qwen3VLTextAttention) and index in POST[1:]:
                    assert output.shape[1]==1 and not dropout
                    values=qwen.repeat_kv(value,module.num_key_value_groups)
                    weights=torch.cat((attention[:,:,:,:self.start],attention[:,:,:,self.end:]),-1)
                    nonvisual=torch.cat((values[:,:,:self.start],values[:,:,self.end:]),-2)
                    textual=(weights@nonvisual)[:,:,0]
                    delta=float((textual[rows].float()-self._native_text_routes[index].float()).abs().max())
                    self.maximum_replayed_text_route_difference=max(self.maximum_replayed_text_route_difference,delta)
                    output=output.clone();output[rows,0]=apply_text_gates(output[rows,0],textual[rows],gates[index])
                    seen.append(index)
                return output,attention
            with patch.object(qwen,'eager_attention_forward',inject):
                for tail in self.decoder.layers[20:]:current=tail(current,**kwargs)
            assert seen==cache.updates==list(range(20,36))
            self.route_layer_calls+=len(POST)
            result=self.decoder.norm(current)
            if self.zero_route:assert torch.equal(result,self.reference_hidden)
            return result
        finally:
            self.enabled=old_enabled;self.replaying_tail=self._replaying=False
            self._native_text_routes.clear()

    def statistics(self,responses):
        result=super().statistics(responses)
        assert self.reader.feature_calls==self.reader.risk_solver_calls==0
        for stat in result:
            stat.update(probe=None,probe_real_threshold=None,probe_gated_recycle=False,
                no_learned_gate=True,route_mode=self.route_mode,route_k=self.route_k,
                diagnostic_zero_route=self.zero_route,diagnostic_flat_signal=self.flat_signal,
                original108_features_computed=False,source_only_threshold_calibration=False,
                route_signal_calls=self.route_signal_calls,route_layer_calls=self.route_layer_calls,
                maximum_route_split_roundoff=self.maximum_split_roundoff,
                maximum_replayed_text_route_difference=self.maximum_replayed_text_route_difference,
                transport_mode='native_conflict_actual_text_route_gating',
                acg_scope='CRG Conflict-B adaptation; all native gates frozen, applied to actual current-layer text routes.',
                visual_attention_weights_preserved_at_each_modified_attention=True,
                head_ranking_feature_guided=self.route_mode!='vri',
                borrowed_operator='CRG signed visual/text route effects and VRI; SAR/FFN ranking is project adaptation.')
        return result
