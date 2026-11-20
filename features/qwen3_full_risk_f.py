"""Complete original risk/F features, with paired native/local-pre-SADI reads.

Reuses the frozen K32/tau.03/w1 math. Keeps compact native GQA values;
head expansion is temporary, instead of retaining36 expanded visual caches.
"""

from contextlib import ExitStack,contextmanager
import time
from unittest.mock import patch

import numpy as np
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.llava_decoding_evidence import ATTENTION_LAYERS
from features.qwen3_online_risk_f import OnlineRiskF,exact_risk,native_write_norms

LAYERS=tuple(range(36))
EARLY=tuple(ATTENTION_LAYERS)


class FullOnlineRiskF(OnlineRiskF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.risk_solver_calls=0

    def features(self,*_args,**_kwargs):
        raise RuntimeError('Use full_features: the4-column API uses a different value-cache layout')

    def _prepare_pre(self):
        for index in EARLY:
            state=self.current[index]
            if 'pre_attention' in state:continue
            query,key,mask,scaling,groups=state['pre_inputs']
            keys=qwen.repeat_kv(key,groups)
            scores=(query@keys.transpose(2,3))*scaling
            if mask is not None:scores=scores+mask
            state['pre_attention']=scores.softmax(-1,dtype=torch.float32).to(query.dtype)[:,:,0,self.start:self.end].contiguous()

    def full_features(self,rows,tokens):
        started=time.perf_counter()
        self._prepare_pre()
        native_r,native_f,pre_r,pre_f,native_i,pre_i,semantic=[],[],[],[],[],[],[]
        for row,token in zip(rows,tokens,strict=True):
            nr,nf,pr,pf,ni,pi,el=[],[],[],[],[],[],[]
            for index in LAYERS:
                state=self.current[index]
                log_s=self._static(index,row)['log_s'][:,self.columns[token]]
                hpre=self.visual[index][row]
                # Native values are causal image-prefix values, shared by both reads.
                values=self.values[index][row:row+1]
                groups=state['attention'].shape[1]//values.shape[1]
                values=qwen.repeat_kv(values,groups)[0]
                def measure(attention):
                    risk,*_=exact_risk(hpre,state['hpre'][row],state['update'][row],attention,log_s)
                    self.risk_solver_calls+=1
                    norms=native_write_norms(values,attention,self.decoder.layers[index].self_attn.o_proj.weight)
                    return risk,float((norms*(1+log_s)).sum()),float(norms.sum())
                r,f,i=measure(state['attention'][row])
                nr.append(r);nf.append(f);ni.append(i)
                if index in EARLY:
                    r,f,i=measure(state['pre_attention'][row])
                pr.append(r);pf.append(f);pi.append(i)
                el.append(float((log_s.expm1()/self.head.weight.shape[0]).sum()))
            native_r.append(nr);native_f.append(nf);pre_r.append(pr);pre_f.append(pf)
            native_i.append(ni);pre_i.append(pi);semantic.append(el)
        result={name:np.asarray(value,dtype=np.float32).reshape(-1,36) for name,value in
            [('native_risk',native_r),('native_F_log',native_f),('pre_risk',pre_r),('pre_F_log',pre_f),
             ('native_I',native_i),('pre_I',pre_i),('E_l_from_log_s',semantic)]}
        result['native']=np.concatenate([result['native_risk'],result['native_F_log']],axis=1)
        result['pre']=np.concatenate([result['pre_risk'],result['pre_F_log']],axis=1)
        assert all(np.isfinite(v).all() for v in result.values())
        self.feature_calls+=len(rows)
        self.feature_seconds+=time.perf_counter()-started
        return result

    @contextmanager
    def capture(self):
        original=qwen.eager_attention_forward
        def read(module,query,key,value,*args,**kwargs):
            output=original(module,query,key,value,*args,**kwargs)
            if isinstance(module,qwen.Qwen3VLTextAttention) and not self._ignore():
                index=module.layer_idx
                self.current[index]['attention']=output[1][:,:,-1,self.start:self.end].contiguous()
                if self.step==0:
                    self.values[index]=value[:,:,self.start:self.end].contiguous()
                if index in EARLY:
                    scaling=kwargs.get('scaling',args[1] if len(args)>1 else module.scaling)
                    mask=kwargs.get('attention_mask',args[0] if args else None)
                    if mask is not None:mask=mask[...,-1:,:key.shape[-2]].contiguous()
                    self.current[index]['pre_inputs']=(query[:,:,-1:].contiguous(),key,mask,scaling,module.num_key_value_groups)
            return output
        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen,'eager_attention_forward',read))
            stack.callback(self.model.register_forward_pre_hook(self._begin).remove)
            for index in LAYERS:
                layer=self.decoder.layers[index]
                stack.callback(layer.register_forward_pre_hook(self._input(index),with_kwargs=True).remove)
                stack.callback(layer.mlp.register_forward_hook(self._ffn(index)).remove)
            if self.callback:
                stack.callback(self.model.register_forward_hook(lambda _m,_a,out:self.callback(self,out)).remove)
            yield self
