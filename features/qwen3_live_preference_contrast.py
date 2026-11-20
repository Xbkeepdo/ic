"""Native full SADI with a read-only FFN-preference contrastive branch."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from features.qwen3_region_memory import RegionEvidenceReader, LAYERS
from features.qwen3_sar_ems_final_target import LastPositionCache
from features.qwen3_sar_ems_attention_batch import Qwen3BatchLateAttention
from features.qwen3_signal_preference_contrast import mix_visual
from features.qwen3_signal_erasure_contrast import contrast_logits


@contextmanager
def preference_rows(span, items):
    original = qwen.eager_attention_forward; calls = []
    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        index = getattr(module, 'layer_idx', None)
        if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in LAYERS or not any(a for _, _, a in items):
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        assert not dropout and not module.training and query.shape[-2] == 1
        key = qwen.repeat_kv(key, module.num_key_value_groups)
        value = qwen.repeat_kv(value, module.num_key_value_groups)
        scores = (query @ key.transpose(2, 3))*scaling
        if attention_mask is not None: scores = scores+attention_mask[..., :key.shape[-2]]
        probability = scores.softmax(-1, dtype=torch.float32)
        for row, prior, amount in items: probability = mix_visual(probability, span, row, prior, amount)
        calls.append(index); probability = probability.to(query.dtype)
        return (probability @ value).transpose(1, 2).contiguous(), probability
    with patch.object(qwen, 'eager_attention_forward', attention): yield calls
    assert calls == (list(LAYERS) if any(a for _, _, a in items) else [])


class Qwen3LivePreferenceContrast:
    def __init__(self, wrapper, span, nouns, *, mode='p_drift', shared=None,
                 disabled=False, zero_alpha=False, zero_amount=False):
        assert mode in ['p_drift', 'p_fixed', 'uniform_drift', 'rolled_drift']
        self.wrapper = wrapper; self.model = wrapper.model; self.span = span; self.nouns = frozenset(nouns)
        self.decoder = self.model.model.language_model; self.head = self.model.get_output_embeddings()
        self.reader = RegionEvidenceReader(self.model, span, self.nouns)
        self.shared = {} if shared is None else shared
        self.reader.static = self.shared.setdefault('static', {})
        self.shared.setdefault('metadata', {})
        self.mode = mode; self.disabled = disabled; self.zero_alpha = zero_alpha; self.zero_amount = zero_amount
        self.checked = set(); self.step = -1; self.reference = None; self.events = {}; self.replay_steps = 0
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.active = None

    def _begin(self, _module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None); assert ids is not None
        self.step += 1; self.reference = None
        if self.active is None: self.active = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
        else:
            for token in self.eos: self.active &= ids[:, -1] != token

    def _remember(self, _module, _args, output):
        self.reference = output.last_hidden_state[:, -1:].clone()

    def _static_guard(self, row):
        key = (19, row)
        if key in self.checked: return
        weights = [self.head.weight, self.decoder.norm.weight, self.decoder.layers[19].self_attn.o_proj.weight]
        identity = tuple((w.data_ptr(), w._version, tuple(w.shape), str(w.dtype)) for w in weights)
        hidden = self.reader.visual[19][row]
        if key in self.shared['metadata']:
            old = self.shared['metadata'][key]
            assert old['identity'] == identity and torch.equal(old['hidden'], hidden)
            assert old['tokens'] == self.reader.tokens
        else:
            self.shared['metadata'][key] = dict(identity=identity, hidden=hidden.clone(), tokens=self.reader.tokens)
        self.checked.add(key)

    def _negative(self, cache, rows, items):
        reader = self.reader
        backups = {i:(cache.layers[i].keys[rows, :, -1:].clone(), cache.layers[i].values[rows, :, -1:].clone()) for i in LAYERS}
        proxy = LastPositionCache(cache, rows, check_identical=self.zero_amount)
        current = reader.anchor.clone(); kwargs = dict(reader.tail_kwargs, past_key_values=proxy)
        try:
            with reader.paused(), preference_rows(self.span, items):
                for index in LAYERS: current = self.decoder.layers[index](current, **kwargs)
            result = self.decoder.norm(current)
            assert proxy.updates == list(LAYERS)
            other = [i for i in range(len(current)) if i not in rows]
            if other: assert torch.equal(result[other], self.reference[other])
            if self.zero_amount: assert torch.equal(result, self.reference)
            return self.head(result[:, -1:]).float()[:, 0]
        finally:
            for index, (keys, values) in backups.items():
                cache.layers[index].keys[rows, :, -1:] = keys
                cache.layers[index].values[rows, :, -1:] = values
                assert torch.equal(cache.layers[index].keys[rows, :, -1:], keys)
                assert torch.equal(cache.layers[index].values[rows, :, -1:], values)

    def _finish(self, _module, _args, output):
        assert self.reader.step == self.step and self.reference is not None
        if self.step == 0 or self.disabled: return output
        native = output.logits[:, -1].float(); tokens = native.argmax(-1)
        rows = [i for i,t in enumerate(tokens.tolist()) if self.active[i] and t in self.nouns]
        if not rows: return output
        items = []; details = []
        for row in rows:
            self._static_guard(row); token = int(tokens[row]); q,p,drift = self.reader.distributions(row, token)
            prior = p if self.mode in ['p_drift', 'p_fixed'] else (torch.ones_like(p)/len(p) if self.mode == 'uniform_drift' else p.roll(len(p)//2))
            amount = .5 if self.mode == 'p_fixed' else .5*drift
            if self.zero_amount: amount = 0.
            items.append((row, prior, amount)); details.append(dict(step=self.step, token=token, drift=drift, amount=amount))
        negative = self._negative(output.past_key_values, rows, items); self.replay_steps += 1
        changed_logits = None
        for row, detail in zip(rows, details, strict=True):
            scores = contrast_logits(native[row], negative[row], alpha=0. if self.zero_alpha else 1., beta=.1)
            apply = not self.zero_alpha and detail['amount'] > 0
            new = int(scores.argmax()) if apply else int(tokens[row])
            detail.update(new_token=new, new_token_text=self.wrapper.tokenizer.decode([new]),
                native_logp=float(native[row].log_softmax(0)[detail['token']]),
                negative_logp=float(negative[row].log_softmax(0)[detail['token']]),
                applied=apply, immediate_change=new != detail['token'], counterfactual_kv_committed=False)
            self.events.setdefault(row, []).append(detail)
            if apply:
                if changed_logits is None: changed_logits = output.logits.float().clone()
                changed_logits[row, -1] = scores
        if changed_logits is not None: output.logits = changed_logits
        return output

    def statistics(self, responses):
        result = []
        for row, tokens in enumerate(responses):
            events = [e for e in self.events.get(row, []) if e['step'] < len(tokens)]
            result.append(dict(guided_steps=sum(e['applied'] for e in events),
                target_matches_generated=sum(e['applied'] and tokens[e['step']] == e['token'] for e in events),
                full_events=events, mode=self.mode, base_layers=list(range(5,19)), source_layer=19,
                action_layers=list(LAYERS), read_only_counterfactual=True, model_steps=self.step+1,
                replay_steps_in_batch=self.replay_steps, semantic_head_calls_in_controller=self.reader.head_calls,
                disabled=self.disabled, zero_alpha=self.zero_alpha, zero_amount=self.zero_amount,
                no_mlp_or_history_filter=True, eligibility_tokens=len(self.nouns), repeated_targets_allowed=True))
        return result

    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            stack.enter_context(Qwen3BatchLateAttention(self.model, self.span, frozenset(), 'sadi').capture())
            stack.callback(self.model.register_forward_pre_hook(self._begin, with_kwargs=True).remove)
            stack.enter_context(self.reader.capture())
            stack.callback(self.model.model.register_forward_hook(self._remember).remove)
            stack.callback(self.model.register_forward_hook(self._finish).remove)
            yield self

    def close(self):
        self.reference = None; self.reader.anchor = self.reader.tail_kwargs = None
        self.reader.current.clear(); self.reader.visual.clear(); self.reader.values.clear(); self.reader.static = {}
