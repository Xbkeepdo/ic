"""Native 108-feature reader with reusable, fully normalized visual semantics."""

from contextlib import ExitStack, contextmanager
import time
from unittest.mock import patch

import numpy as np
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.qwen3_online_risk_f import OnlineRiskF, exact_risk, native_write_norms
from models.dgst_capture import apply_decoder_final_norm


class NativeRiskIEl(OnlineRiskF):
    def __init__(self, *args, shared_static=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_static = {} if shared_static is None else shared_static
        self.cache_hits = self.risk_solver_calls = 0
        self.last_write_norms = {}

    def _static(self, layer, row):
        key = (layer, row)
        if key in self.static:
            return self.static[key]
        hidden, values = self.visual[layer][row], self.values[layer][row]
        identity = (self.tokens, self.start, self.end, self.head.weight.data_ptr(), self.head.weight._version)
        if key in self.shared_static:
            entry = self.shared_static[key]
            assert entry['identity'] == identity
            assert torch.equal(entry['hidden'], hidden) and torch.equal(entry['values'], values)
            self.cache_hits += 1
        else:
            started = time.perf_counter()
            normalized = apply_decoder_final_norm(self.decoder.norm, hidden)
            columns = torch.tensor(self.tokens, device=hidden.device)
            probabilities = []
            for offset in range(0, len(hidden), 64):
                logits = self.head(normalized[offset:offset+64]).float()
                probabilities.append((logits[:, columns]-logits.logsumexp(-1, keepdim=True)).clamp_max(0).exp())
                self.head_calls += 1
            probabilities = torch.cat(probabilities)
            entry = dict(identity=identity, hidden=hidden, values=values,
                         probabilities=probabilities, E_l=probabilities.sum(0),
                         log_s=torch.log1p(self.head.weight.shape[0]*probabilities))
            self.shared_static[key] = entry
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            self.static_seconds += time.perf_counter()-started
        self.static[key] = entry
        return entry

    def features(self, rows, tokens):
        started = time.perf_counter()
        vectors = []
        self.last_write_norms = {}
        for row, token in zip(rows, tokens, strict=True):
            risks, writes, semantics = [], [], []
            for index in range(36):
                cached, state = self._static(index, row), self.current[index]
                column = self.columns[token]
                risk, *_ = exact_risk(self.visual[index][row], state['hpre'][row],
                    state['update'][row], state['attention'][row], cached['log_s'][:, column])
                self.risk_solver_calls += 1
                groups = state['attention'].shape[1]//self.values[index].shape[1]
                values = qwen.repeat_kv(self.values[index][row:row+1], groups)[0]
                norms = native_write_norms(values, state['attention'][row],
                                          self.decoder.layers[index].self_attn.o_proj.weight)
                if index == 19:
                    self.last_write_norms[row] = norms
                risks.append(risk)
                writes.append(float(norms.sum()))
                semantics.append(float(cached['E_l'][column]))
            vectors.append(risks+writes+semantics)
        result = np.asarray(vectors, dtype=np.float32).reshape(-1, 108)
        assert np.isfinite(result).all()
        self.feature_calls += len(rows)
        self.feature_seconds += time.perf_counter()-started
        return torch.from_numpy(result).to(self.head.weight.device)

    @contextmanager
    def capture(self):
        original = qwen.eager_attention_forward
        def read(module, query, key, value, *args, **kwargs):
            output = original(module, query, key, value, *args, **kwargs)
            if isinstance(module, qwen.Qwen3VLTextAttention) and not self._ignore():
                index = module.layer_idx
                self.current[index]['attention'] = output[1][:, :, -1, self.start:self.end].contiguous()
                if self.step == 0:
                    self.values[index] = value[:, :, self.start:self.end].contiguous()
            return output
        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, 'eager_attention_forward', read))
            stack.callback(self.model.register_forward_pre_hook(self._begin).remove)
            for index, layer in enumerate(self.decoder.layers):
                stack.callback(layer.register_forward_pre_hook(self._input(index), with_kwargs=True).remove)
                stack.callback(layer.mlp.register_forward_hook(self._ffn(index)).remove)
            yield self
