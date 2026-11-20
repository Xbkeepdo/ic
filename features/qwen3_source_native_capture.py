"""Read original108 features from a forced native cached source trajectory.

The existing numerical feature implementation is unchanged. The collector
hooks the multimodal core so that its unused final vocabulary projection
does not run at every forced text step.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_native_risk_i_el import NativeRiskIEl


class SourceNativeCapture(NativeRiskIEl):
    @contextmanager
    def capture(self):
        original=qwen.eager_attention_forward

        def read(module,query,key,value,*args,**kwargs):
            result=original(module,query,key,value,*args,**kwargs)
            if isinstance(module,qwen.Qwen3VLTextAttention):
                index=module.layer_idx
                self.current[index]['attention']=result[1][:,:,-1,self.start:self.end].contiguous()
                if self.step==0:self.values[index]=value[:,:,self.start:self.end].contiguous()
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen,'eager_attention_forward',read))
            stack.callback(self.model.model.register_forward_pre_hook(self._begin).remove)
            for index,layer in enumerate(self.decoder.layers):
                stack.callback(layer.register_forward_pre_hook(self._input(index),with_kwargs=True).remove)
                stack.callback(layer.mlp.register_forward_hook(self._ffn(index)).remove)
            yield self
