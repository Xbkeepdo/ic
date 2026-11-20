"""Read the original K32/tau.03/w1 risk and F_log at two frozen layers.

Read-only capture. Native BF16 attention/value multiplication and output
projection follow the original re-extractor, rather than a FP32 Gram shortcut.
"""

from contextlib import ExitStack, contextmanager
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.dgst_t import (_cosine_distance_matrix, _renormalize,
    _solve_transport_problem, _topk_union_indices)
from models.dgst_capture import apply_decoder_final_norm

LAYERS = (27, 28)


class CompactRiskFGate:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.parameters = {k: data[k].copy() for k in data.files}
        assert self.parameters['feature_names'].tolist() == ['risk', 'F_log', 'risk', 'F_log']
        assert self.parameters['layers'].tolist() == [27, 27, 28, 28]
        self.threshold = float(self.parameters['threshold'])

    def predict(self, features):
        # StandardScaler preserves the input float32 array after EACH operation.
        x = np.array(features, dtype=np.float32, copy=True).reshape(-1, 4)
        p = self.parameters
        x -= p['mean']
        x /= p['scale']
        hidden = np.maximum(0, x @ p['weight0'] + p['bias0'])
        from scipy.special import expit
        return expit(hidden @ p['weight1'] + p['bias1'])[:, 0]


def exact_risk(visual_hpre, prediction_hpre, update, attention, log_s, visual_cost=None):
    source = F.cosine_similarity(update.float()[None], visual_hpre.float(), dim=-1).div(.03).softmax(0)
    mean = attention.float().mean(0)
    mean = mean / mean.sum().clamp_min(1e-30)
    target = mean * log_s
    target = target / target.sum().clamp_min(1e-30)
    support = _topk_union_indices(source, target, 32)
    a = _renormalize(source[support]).cpu().numpy()
    b = _renormalize(target[support]).cpu().numpy()
    local_cost = (_cosine_distance_matrix(visual_hpre[support]) if visual_cost is None
                  else visual_cost[support][:, support])
    cost = local_cost.float().cpu().numpy()
    penalties = (1-F.cosine_similarity(prediction_hpre.float()[None],
                visual_hpre.float(), dim=-1)[support].clamp(-1, 1)).float().cpu().numpy()
    problem = (a, b, cost)
    transport = _solve_transport_problem(problem, 'emd')
    penalty = float(np.dot(a, penalties)+np.dot(b, penalties))
    return transport+penalty, dict(transport=transport, penalty=penalty, support=len(support)), problem, penalties


def native_write_norms(values, attention, weight):
    weighted = (attention.T[..., None] * values.permute(1, 0, 2)).flatten(1)
    return F.linear(weighted, weight, bias=None).float().norm(dim=-1)


class OnlineRiskF:
    def __init__(self, model, span, tokens, *, callback=None, owner=None):
        self.model, self.decoder = model, model.model.language_model
        self.head = model.get_output_embeddings()
        self.start, self.end = span
        self.tokens = tuple(sorted(tokens))
        self.columns = {token: i for i, token in enumerate(self.tokens)}
        self.callback, self.owner = callback, owner
        self.step = -1
        self.visual, self.values, self.current = {}, {}, {}
        self.static = {}
        self.feature_seconds = self.static_seconds = 0.
        self.feature_calls = self.head_calls = 0

    def _ignore(self):
        return self.owner is not None and (self.owner._replaying or self.owner.replaying_tail)

    def _begin(self, *_):
        if not self._ignore():
            self.step += 1
            self.current.clear()

    def _input(self, index):
        def read(_module, args, kwargs):
            if self._ignore():
                return
            hidden = args[0] if args else kwargs['hidden_states']
            self.current[index] = dict(hpre=hidden[:, -1].contiguous())
            if self.step == 0:
                self.visual[index] = hidden[:, self.start:self.end].contiguous()
        return read

    def _ffn(self, index):
        def read(_module, _args, value):
            if not self._ignore():
                self.current[index]['update'] = value[:, -1].contiguous()
        return read

    def _static(self, layer, row):
        key = (layer, row)
        if key not in self.static:
            started = time.perf_counter()
            hidden = self.visual[layer][row]
            normalized = apply_decoder_final_norm(self.decoder.norm, hidden)
            columns = torch.tensor(self.tokens, device=hidden.device)
            semantic = []
            for offset in range(0, len(hidden), 64):
                logits = self.head(normalized[offset:offset+64]).float()
                selected = logits[:, columns]-logits.logsumexp(-1, keepdim=True)
                semantic.append(torch.log1p(logits.shape[-1]*selected.clamp_max(0).exp()))
                self.head_calls += 1
            self.static[key] = dict(log_s=torch.cat(semantic))
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            self.static_seconds += time.perf_counter()-started
        return self.static[key]

    def features(self, rows, tokens, *, reference_details=False):
        started = time.perf_counter()
        result, details = [], []
        for row, token in zip(rows, tokens, strict=True):
            vector, record = [], {}
            for index in LAYERS:
                cached = self._static(index, row)
                hpre, state = self.visual[index][row], self.current[index]
                log_s = cached['log_s'][:, self.columns[token]]
                attention = state['attention'][row]
                risk, terms, problem, penalties = exact_risk(hpre, state['hpre'][row],
                    state['update'][row], attention, log_s)
                norms = native_write_norms(self.values[index][row], attention,
                                          self.decoder.layers[index].self_attn.o_proj.weight)
                f_log = float((norms*(1+log_s)).sum())
                vector.extend([risk, f_log])
                record[str(index)] = dict(terms, F_log=f_log)
                if reference_details:
                    record[str(index)].update(problem=problem, penalties=penalties,
                        log_s=log_s, norms=norms, hpre=hpre,
                        prediction_hpre=state['hpre'][row], update=state['update'][row],
                        attention=attention, values=self.values[index][row])
            result.append(vector)
            details.append(record)
        values = np.asarray(result, dtype=np.float32).reshape(-1, 4)
        assert np.isfinite(values).all()
        self.feature_calls += len(rows)
        self.feature_seconds += time.perf_counter()-started
        return values, details

    @contextmanager
    def capture(self):
        original = qwen.eager_attention_forward
        def attention(module, query, key, value, *args, **kwargs):
            output = original(module, query, key, value, *args, **kwargs)
            if (isinstance(module, qwen.Qwen3VLTextAttention)
                    and module.layer_idx in LAYERS and not self._ignore()):
                index = module.layer_idx
                self.current[index]['attention'] = output[1][:, :, -1, self.start:self.end].contiguous()
                if self.step == 0:
                    self.values[index] = qwen.repeat_kv(value, module.num_key_value_groups)[:, :, self.start:self.end].contiguous()
            return output
        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, 'eager_attention_forward', attention))
            stack.callback(self.model.register_forward_pre_hook(self._begin).remove)
            for index in LAYERS:
                layer = self.decoder.layers[index]
                stack.callback(layer.register_forward_pre_hook(self._input(index), with_kwargs=True).remove)
                stack.callback(layer.mlp.register_forward_hook(self._ffn(index)).remove)
            if self.callback:
                stack.callback(self.model.register_forward_hook(lambda _m, _a, out: self.callback(self, out)).remove)
            yield self
