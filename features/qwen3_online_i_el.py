"""Read the original I/E_l features without changing Qwen3 generation.

E_l is cached per image/layer/target during prefill. I reads the actual
attention row and native KV values, and is evaluated only on requested rows.
"""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from models.dgst_capture import apply_decoder_final_norm, target_probabilities_multi


class FullIElProbe(torch.nn.Module):
    """Frozen original probe, including its original input standardization."""

    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        state = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        with np.load(directory / 'input_scaler.npz', allow_pickle=False) as scaler:
            for key in ('mean', 'scale'):
                self.register_buffer(key, torch.from_numpy(scaler[key].copy()))
        for name, key in (('weight0', 'net.0.weight'), ('bias0', 'net.0.bias'),
                          ('weight1', 'net.3.weight'), ('bias1', 'net.3.bias')):
            self.register_buffer(name, state[key])
        assert self.weight0.shape == (128, 72) and self.weight1.shape == (1, 128)

    @torch.inference_mode()
    def forward(self, features):
        standardized = ((features.to(self.mean.dtype) - self.mean) / self.scale).float()
        hidden = F.linear(standardized, self.weight0, self.bias0).relu()
        return F.linear(hidden, self.weight1, self.bias1).sigmoid()[:, 0]


class OnlineIElCapture:
    def __init__(self, model, span, token_ids, *, callback=None, validate=False):
        self.model = model
        self.decoder = model.model.language_model
        self.head = model.get_output_embeddings()
        self.start, self.end = span
        self.tokens = tuple(sorted(set(map(int, token_ids))))
        self.columns = {token: i for i, token in enumerate(self.tokens)}
        self.callback = callback
        self.validate = validate
        self.step = -1
        self.attention = {}
        self.values = {}
        self.semantics = {}
        self.validation_states = {}
        self.mid31 = self.ffn31 = None
        self.semantic_seconds = 0.
        self.feature_seconds = 0.

    @torch.inference_mode()
    def _semantic_sum(self, visual):
        normalized = apply_decoder_final_norm(self.decoder.norm, visual)
        ids = torch.tensor(self.tokens, device=self.head.weight.device)
        results = []
        for image in normalized:
            probabilities = []
            # Keep each image's original 64-position GEMM shapes. Vectorize
            # target gathering instead of assigning one column per kernel.
            for offset in range(0, image.shape[0], 64):
                logits = F.linear(image[offset:offset + 64].to(self.head.weight.dtype),
                                  self.head.weight, getattr(self.head, 'bias', None)).float()
                denominator = logits.logsumexp(-1, keepdim=True)
                probabilities.append((logits.index_select(1, ids) - denominator).clamp_max(0).exp())
            results.append(torch.cat(probabilities).sum(0))
        return torch.stack(results)

    def _input(self, index):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs['hidden_states']
            if self.step == 0:
                assert hidden.shape[1] >= self.end
                started = time.perf_counter()
                self.semantics[index] = self._semantic_sum(hidden[:, self.start:self.end])
                self.semantic_seconds += time.perf_counter() - started
                if self.validate:
                    self.validation_states[index] = hidden.detach().clone()
        return hook

    def _begin(self, _module, _args, _kwargs):
        self.step += 1
        self.attention.clear()
        self.values.clear()
        self.mid31 = self.ffn31 = None

    def _mid(self, _module, args):
        self.mid31 = args[0][:, -1]

    def _ffn(self, _module, _args, output):
        self.ffn31 = output[:, -1]

    @torch.inference_mode()
    def features(self, rows, tokens):
        assert len(rows) == len(tokens) and len(self.attention) == len(self.decoder.layers) == 36
        started = time.perf_counter()
        intensity, semantics = [], []
        for index, layer in enumerate(self.decoder.layers):
            # The tensors refer to native KV, not a duplicate projected cache.
            raw = self.values[index][rows, :, self.start:self.end]
            weights = self.attention[index][rows]
            groups = weights.shape[1] // raw.shape[1]
            visual = raw.repeat_interleave(groups, dim=1)
            writes = (weights.transpose(1, 2).unsqueeze(-1)
                      * visual.permute(0, 2, 1, 3)).reshape(len(rows), self.end - self.start, -1)
            writes = F.linear(writes, layer.self_attn.o_proj.weight, bias=None).float()
            intensity.append(writes.norm(dim=-1).sum(-1))
            cols = [self.columns[int(token)] for token in tokens]
            semantics.append(self.semantics[index][rows, cols])
        result = torch.cat((torch.stack(intensity, 1), torch.stack(semantics, 1)), 1)
        self.feature_seconds += time.perf_counter() - started
        return result

    @torch.inference_mode()
    def visual_support31(self, rows, tokens):
        weights = self.attention[31][rows].float()
        raw = self.values[31][rows, :, self.start:self.end].float()
        visual = raw.repeat_interleave(weights.shape[1] // raw.shape[1], dim=1)
        context = (weights.unsqueeze(-1) * visual).sum(2).flatten(1)
        update = F.linear(context, self.decoder.layers[31].self_attn.o_proj.weight.float())
        residual = self.mid31[rows].float()
        norm = self.decoder.norm

        def lens(x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.float()

        direction = self.head.weight[list(map(int, tokens))].float()
        return ((lens(residual) - lens(residual - update)) * direction).sum(-1)

    @torch.inference_mode()
    def reference_check(self, rows, tokens):
        from features.attention_writes import visual_attention_writes
        assert self.validate and self.step == 0
        actual = self.features(rows, tokens)
        reference = torch.empty_like(actual)
        for index, layer in enumerate(self.decoder.layers):
            for slot, (row, token) in enumerate(zip(rows, tokens, strict=True)):
                hidden = self.validation_states[index][row:row + 1]
                length = hidden.shape[1]
                capture = dict(h_prev=hidden,
                               attn_weights=self.attention[index][row:row + 1, :, None],
                               attention_query_positions=(length - 1,))
                # visual_attention_writes only consumes the visual slice of
                # this row; compact attention is padded to absolute keys.
                padded = torch.zeros((1, capture['attn_weights'].shape[1], 1, length),
                                     device=hidden.device, dtype=capture['attn_weights'].dtype)
                padded[:, :, :, self.start:self.end] = capture['attn_weights']
                capture['attn_weights'] = padded
                write = visual_attention_writes(layer, capture, [length - 1], self.start, self.end)
                reference[slot, index] = write.norm(dim=-1).sum()
                visual = hidden[0, self.start:self.end].float()
                probability = target_probabilities_multi(output_layer=self.head,
                    states=apply_decoder_final_norm(self.decoder.norm, visual),
                    target_token_ids=[token], chunk_size=64)
                reference[slot, index + 36] = probability.sum()
        difference = (actual - reference).abs()
        relative = difference / reference.abs().clamp_min(1e-8)
        return dict(I_max_abs=float(difference[:, :36].max()),
                    I_max_relative=float(relative[:, :36].max()),
                    E_l_max_abs=float(difference[:, 36:].max()),
                    E_l_max_relative=float(relative[:, 36:].max()),
                    values=actual.cpu().tolist(), reference=reference.cpu().tolist())

    def _finish(self, _module, _args, output):
        try:
            if self.callback is not None:
                self.callback(self, output)
        finally:
            self.attention.clear()
            self.values.clear()

    @contextmanager
    def capture(self):
        original = qwen.eager_attention_forward

        def read(module, query, key, value, *args, **kwargs):
            result = original(module, query, key, value, *args, **kwargs)
            if isinstance(module, qwen.Qwen3VLTextAttention):
                self.attention[module.layer_idx] = result[1][:, :, -1, self.start:self.end].contiguous()
                self.values[module.layer_idx] = value
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(qwen, 'eager_attention_forward', read))
            for module, hook, options in (
                (self.model, self._begin, dict(pre=True, with_kwargs=True)),
                (self.model, self._finish, dict(pre=False)),
                (self.decoder.layers[31].post_attention_layernorm, self._mid, dict(pre=True)),
                (self.decoder.layers[31].mlp, self._ffn, dict(pre=False)),
            ):
                pre = options.pop('pre')
                register = module.register_forward_pre_hook if pre else module.register_forward_hook
                handle = register(hook, **options)
                stack.callback(handle.remove)
            for index, layer in enumerate(self.decoder.layers):
                handle = layer.register_forward_pre_hook(self._input(index), with_kwargs=True)
                stack.callback(handle.remove)
            yield self
