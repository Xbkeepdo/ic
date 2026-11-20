"""Teacher-force cached captions with SADI on every prediction query.

Parallel mode edits rows from the last prompt query onward. Last mode
matches the released project controller's one-current-query convention.
Saved image-prefix probabilities supply semantics; no vocabulary readout
is performed by this collector.
"""

from contextlib import ExitStack,contextmanager
from unittest.mock import patch

import numpy as np
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.llava_decoding_evidence import ATTENTION_LAYERS
from features.llava_sadi import sadi_visual_logits
from features.qwen3_online_risk_f import exact_risk,native_write_norms


def sadi_query_slice(scores,start,end,query_start):
    """Apply the unchanged scalar-query SADI rule independently to many rows."""
    assert scores.ndim==4 and 0<=query_start<scores.shape[2]
    visual=scores[:,:,query_start:,start:end]
    b,h,q,m=visual.shape
    corrected=sadi_visual_logits(visual.permute(0,2,1,3).reshape(b*q,h,m))
    result=scores.clone()
    result[:,:,query_start:,start:end]=corrected.reshape(b,q,h,m).permute(0,2,1,3)
    return result


class TeacherSadiCapture:
    def __init__(self,model,span,prompt_length,positions,*,mode='parallel',sadi=True,capture=True):
        assert mode in ('parallel','last') and model.training is False
        self.model=model;self.core=model.model;self.decoder=model.model.language_model
        self.start,self.end=span;self.prompt_length=int(prompt_length)
        assert 0<=self.start<self.end<self.prompt_length
        self.positions=tuple(map(int,positions));assert self.positions or mode=='last'
        self.mode=mode;self.sadi=bool(sadi);self.capture_states=bool(capture)
        self.visual={};self.values={};self.current={};self.index=None
        self.calls=self.risk_solver_calls=self.feature_targets=0

    def _begin(self,_module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        assert ids is not None and ids.shape[0]==1,'Source trace intentionally uses one image per forward.'
        length=ids.shape[1]
        positions=(length-1,) if self.mode=='last' else self.positions
        assert all(0<=position<length for position in positions)
        self.index=torch.tensor(positions,device=ids.device,dtype=torch.long)
        self.current={};self.calls+=1

    def _input(self,index):
        def remember(_module,args,kwargs):
            if not self.capture_states:return
            hidden=args[0] if args else kwargs['hidden_states']
            self.current[index]=dict(prediction_hpre=hidden[0].index_select(0,self.index).contiguous())
            if hidden.shape[1]>=self.end:self.visual[index]=hidden[0,self.start:self.end].contiguous()
        return remember

    def _ffn(self,index):
        def remember(_module,_args,output):
            if self.capture_states:self.current[index]['update']=output[0].index_select(0,self.index).contiguous()
        return remember

    @torch.inference_mode()
    def features(self,log_s,semantics):
        assert self.capture_states and set(self.current)==set(range(36))
        logs=torch.as_tensor(np.array(log_s,copy=True),device=self.index.device,dtype=torch.float32)
        semantic=np.asarray(semantics,dtype=np.float32)
        count=len(self.index);assert logs.shape==(count,36,self.end-self.start) and semantic.shape==(count,36)
        risk=np.empty((count,36),np.float32);intensity=np.empty_like(risk)
        for index in range(36):
            state=self.current[index]
            values=self.values[index][None]
            groups=state['attention'].shape[1]//values.shape[1]
            values=qwen.repeat_kv(values,groups)[0]
            for target in range(count):
                value,*_=exact_risk(self.visual[index],state['prediction_hpre'][target],
                    state['update'][target],state['attention'][target],logs[target,index])
                norms=native_write_norms(values,state['attention'][target],
                    self.decoder.layers[index].self_attn.o_proj.weight)
                risk[target,index]=value;intensity[target,index]=float(norms.sum())
                self.risk_solver_calls+=1
        self.feature_targets+=count
        result=np.concatenate([risk,intensity,semantic],axis=1)
        assert result.shape==(count,108) and np.isfinite(result).all()
        return result

    @contextmanager
    def context(self):
        original=qwen.eager_attention_forward
        def forward(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
            if not isinstance(module,qwen.Qwen3VLTextAttention):
                return original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
            assert not dropout and not module.training
            index=module.layer_idx
            if self.sadi and index in ATTENTION_LAYERS:
                keys=qwen.repeat_kv(key,module.num_key_value_groups)
                values=qwen.repeat_kv(value,module.num_key_value_groups)
                scores=(query@keys.transpose(2,3))*scaling
                if attention_mask is not None:scores=scores+attention_mask[...,:keys.shape[-2]]
                query_start=query.shape[2]-1 if self.mode=='last' else self.prompt_length-1
                scores=sadi_query_slice(scores,self.start,self.end,query_start)
                attention=scores.softmax(-1,dtype=torch.float32).to(query.dtype)
                output=(attention@values).transpose(1,2).contiguous()
            else:
                output,attention=original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
            if self.capture_states:
                self.current[index]['attention']=attention[0].index_select(1,self.index)[:,:,self.start:self.end].permute(1,0,2).contiguous()
                if query.shape[2]>=self.end:self.values[index]=value[0,:,self.start:self.end].contiguous()
            return output,attention
        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen,'eager_attention_forward',forward))
            stack.callback(self.core.register_forward_pre_hook(self._begin,with_kwargs=True).remove)
            if self.capture_states:
                for index,layer in enumerate(self.decoder.layers):
                    stack.callback(layer.register_forward_pre_hook(self._input(index),with_kwargs=True).remove)
                    stack.callback(layer.mlp.register_forward_hook(self._ffn(index)).remove)
            yield self
