"""Final-target semantic SAR/EMS attention after complete SADI, without a probe.

The static cache retains selected semantic columns, never full vocabulary
logits. Layer31 reads the actual model argmax, replays only its current query
and layers32--35, and uses no backward pass or detector trained on captions.
"""

from contextlib import ExitStack, contextmanager
import math
from unittest.mock import patch

import torch
from torch.nn import functional as F

from features import qwen3_sar_ems_layer_feedback as feedback
from features.llava_sar_ems_attention import rebalance_visual_logits, visual_write_norms
from features.qwen3_sar_ems_final_target import LastPositionCache
from features.qwen3_sar_ems_removal_gate import Qwen3BatchWriteRemovalGate
from models.dgst_capture import apply_decoder_final_norm


def blocked_write_gram(values, weight, chunk=64):
    heads, count, dim = values.shape
    projection = weight.float().reshape(weight.shape[0], heads, dim).permute(1, 2, 0)
    blocks = []
    for begin in range(0, count, chunk):
        writes = torch.bmm(values[:, begin:begin+chunk].float(), projection).permute(1, 0, 2)
        blocks.append(torch.bmm(writes, writes.transpose(1, 2)))
    return torch.cat(blocks)


class StaticSemanticWrites:
    def __init__(self, hpre, values, weight, norm, head, tokens, chunk=64):
        self.hpre, self.values = hpre.clone(), values.clone()
        self.tokens = tuple(tokens)
        self.columns = {token: index for index, token in enumerate(tokens)}
        self.weight = (weight.data_ptr(), weight._version)
        self.head_weight = (head.weight.data_ptr(), head.weight._version)
        self.norm_weight = (norm.weight.data_ptr(), norm.weight._version)
        self.gram = blocked_write_gram(values, weight, chunk)
        normalized = apply_decoder_final_norm(norm, hpre)
        selected = torch.tensor(tokens, device=hpre.device)
        blocks = []
        self.projection_calls = 0
        for begin in range(0, len(hpre), chunk):
            logits = head(normalized[begin:begin+chunk]).float()
            log_z = logits.logsumexp(-1, keepdim=True)
            blocks.append(F.softplus(logits[:, selected] - log_z + math.log(logits.shape[-1])))
            self.projection_calls += 1
        self.log_s = torch.cat(blocks)

    def check(self, hpre, values, weight, norm, head, tokens):
        assert self.tokens == tuple(tokens)
        assert self.weight == (weight.data_ptr(), weight._version)
        assert self.head_weight == (head.weight.data_ptr(), head.weight._version)
        assert self.norm_weight == (norm.weight.data_ptr(), norm.weight._version)
        assert torch.equal(hpre, self.hpre), 'Static cache has a different image prefix'
        assert torch.equal(values, self.values), 'Static cache has different visual values'

    def distribution(self, attention, token, mode):
        semantic = self.log_s[:, self.columns[token]]
        weights = visual_write_norms(self.gram, attention)
        weights = weights * (1 + semantic if mode == 'full_sar' else semantic)
        return weights / weights.sum().clamp_min(1e-12)


class Qwen3BatchSemanticFinal(Qwen3BatchWriteRemovalGate):
    def __init__(self, model, span, nouns, *, tokenizer, history_parser,
                 semantic_mode='full_sar', static_cache=None, strength=5.,
                 disable_guidance=False, diagnostic_zero_replay=False,
                 diagnostic_reject_all=False, diagnostic_force_guidance=False, permute=False):
        super().__init__(model, span, nouns)
        if semantic_mode not in ('full_sar', 'semantic_increment'):
            raise ValueError(semantic_mode)
        self.sources = self.targets = (31,)
        self.mode = 'same'
        self.strength = float(strength)
        self.enabled = not disable_guidance
        self.semantic_mode = semantic_mode
        self.static_cache = {} if static_cache is None else static_cache
        self.sorted_tokens = tuple(sorted(nouns))
        self.tokenizer, self.history_parser = tokenizer, history_parser
        self.token_classes = history_parser.token_classes(tokenizer, nouns)
        self.diagnostic_zero_replay = diagnostic_zero_replay
        self.diagnostic_reject_all = diagnostic_reject_all
        self.force_guidance = diagnostic_force_guidance
        self.permute = permute
        self.output_history, self.class_decisions = [], {}
        self.source_output = self.tail_kwargs = self.native_hidden = None
        self.replaying_tail = False
        self.replay_steps = self.cache_builds = self.cache_reuses = self.semantic_head_calls = 0
        self.checked_cache_rows = set()

    def _begin_step(self, module, args, kwargs):
        if kwargs.get('output_hidden_states'):
            raise RuntimeError('Semantic feedback supports generation without hidden-state output')
        continuation = bool(self.steps)
        super()._begin_step(module, args, kwargs)
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not continuation:
            self.output_history = [[] for _ in range(len(ids))]
        else:
            for history, token in zip(self.output_history, ids[:, -1].tolist(), strict=True):
                history.append(token)
        self.source_output = self.tail_kwargs = self.native_hidden = None

    def _after_layer(self, index):
        def retain(_module, _args, output):
            self.source_output = output[:, -1:].clone()
            return output
        return retain

    def _remember_tail(self, _module, _args, kwargs):
        if not self.replaying_tail:
            self.tail_kwargs = dict(kwargs)

    def _remember_decoder(self, _module, _args, output):
        if not self.replaying_tail:
            self.native_hidden = output.last_hidden_state[:, -1:].clone()

    def _terms(self, row):
        layer = self.decoder.layers[31]
        args = (self.visual_hpre[31][row], self.visual_values[31][row],
                layer.self_attn.o_proj.weight, self.decoder.norm, self.head, self.sorted_tokens)
        if row not in self.static_cache:
            self.static_cache[row] = StaticSemanticWrites(*args)
            self.cache_builds += 1
            self.semantic_head_calls += self.static_cache[row].projection_calls
        elif row not in self.checked_cache_rows:
            self.static_cache[row].check(*args)
            self.cache_reuses += 1
        self.checked_cache_rows.add(row)
        return self.static_cache[row]

    def _budget(self, source, final_tokens):
        state = self.scratch[source]
        residual, update = state['ffn_residual'], state['update']
        rows, budgets, details = [], [], []
        for row, (token, active) in enumerate(zip(final_tokens.tolist(), self.active.tolist(), strict=True)):
            if not active or token not in self.nouns or token in self.seen[row]:
                continue
            assert len(self.output_history[row]) == self.step
            text = self.tokenizer.decode(self.output_history[row], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            prior = self.history_parser.classes(text)
            canonical = self.token_classes[token]
            blocked = canonical in prior
            self.class_decisions.setdefault(row, []).append(dict(step=self.step, token=token,
                canonical=canonical, prior_classes=sorted(prior), blocked=blocked))
            if blocked:
                continue
            before = apply_decoder_final_norm(self.decoder.norm, residual[row]).float()
            after = apply_decoder_final_norm(self.decoder.norm, residual[row]+update[row]).float()
            gain = self.head.weight[token].float() @ (after-before)
            if self.force_guidance:
                gain = gain.clamp_min(1.)
            elif gain.item() <= 0:
                continue
            target = self._terms(row).distribution(state['attention'][row], token, self.semantic_mode)
            preference = (self.visual_normalized[source][row]
                @ F.normalize(update[row].float(), dim=-1) / .03).softmax(-1)
            drift = (1-(target*preference).sqrt().sum()).clamp(0, 1)
            deficit = (target-preference).clamp_min(0)
            budget = self.strength * drift * (gain/2).tanh() * deficit / deficit.amax().clamp_min(1e-6)
            assert torch.isfinite(budget).all(), 'Nonfinite semantic attention budget'
            if budget.max().item() <= 0:
                continue
            rows.append(row)
            budgets.append(budget)
            details.append(dict(token=token, gain=float(gain), drift=float(drift),
                budget_max=float(budget.max()), target_sum=float(target.sum()),
                preference_sum=float(preference.sum()), canonical=canonical))
        return rows, torch.stack(budgets) if budgets else None, details

    def _recompute(self, index, original_output, rows, budget):
        def reweight(visual, current):
            return rebalance_visual_logits(visual, current, permute=self.permute)
        with patch.object(feedback, 'rebalance_visual_logits', reweight):
            return super()._recompute(index, original_output, rows, budget)

    def _finish_model(self, _module, _args, output):
        if self.step == 0:
            self.scratch.clear()
            return output
        assert self.native_hidden is not None and self.source_output is not None and self.tail_kwargs is not None
        native_logits = output.logits[:, -1].float().clone()
        final_tokens = native_logits.argmax(-1)
        if self.diagnostic_zero_replay:
            rows = self.active.nonzero().flatten().tolist()
            budget = self.source_output.new_zeros((len(rows), self.end-self.start))
            details = [dict(token=int(final_tokens[row]), gain=0., drift=0., budget_max=0.) for row in rows]
        else:
            rows, budget, details = self._budget(31, final_tokens)
        if not rows:
            self.scratch.clear()
            return output
        native_p = native_logits.log_softmax(-1)
        kv = self.tail_kwargs['past_key_values']
        backup = {i: (kv.layers[i].keys[rows, :, -1:].clone(), kv.layers[i].values[rows, :, -1:].clone())
                  for i in range(32, len(self.decoder.layers))} if self.diagnostic_reject_all else {}
        cache = LastPositionCache(kv, rows, check_identical=self.diagnostic_zero_replay)
        kwargs = dict(self.tail_kwargs, past_key_values=cache)
        current = self._recompute(31, self.source_output, rows, budget)
        self.replaying_tail = True
        try:
            for layer in self.decoder.layers[32:]:
                current = layer(current, **kwargs)
        finally:
            self.replaying_tail = False
        assert cache.updates == list(range(32, len(self.decoder.layers)))
        corrected = self.decoder.norm(current)
        edited = self.head(corrected)[:, 0]
        edited_p = edited.float().log_softmax(-1)
        if self.diagnostic_zero_replay:
            assert torch.equal(corrected[rows], self.native_hidden[rows])
            assert torch.equal(edited[rows], output.logits[rows, -1])
        for row, detail in zip(rows, details, strict=True):
            detail.update(actual_log_probability_change=float(edited_p[row, detail['token']]-native_p[row, detail['token']]),
                          accepted=not self.diagnostic_reject_all)
        if self.diagnostic_reject_all:
            for i, (keys, values) in backup.items():
                kv.layers[i].keys[rows, :, -1:] = keys
                kv.layers[i].values[rows, :, -1:] = values
        else:
            output.logits[rows, -1] = edited[rows].to(output.logits.dtype)
        self._record(31, 31, rows, details)
        self.replay_steps += 1
        self.scratch.clear()
        return output

    @contextmanager
    def capture(self):
        with ExitStack() as stack:
            stack.enter_context(super().capture())
            if self.enabled:
                stack.callback(self.decoder.layers[32].register_forward_pre_hook(self._remember_tail, with_kwargs=True).remove)
                stack.callback(self.decoder.register_forward_hook(self._remember_decoder).remove)
                stack.callback(self.model.register_forward_hook(self._finish_model).remove)
            yield self

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stats) in enumerate(zip(responses, result, strict=True)):
            events = [dict(e) for e in self.events.get(row, ()) if e['step'] < len(tokens)]
            decisions = [dict(e) for e in self.class_decisions.get(row, ()) if e['step'] < len(tokens)]
            stats.update(semantic_mode=self.semantic_mode, semantic_events=events,
                first_object_class_gate=True, history_uses_generated_prefix_only=True,
                class_decisions=decisions, blocked_by_class_history=sum(d['blocked'] for d in decisions),
                gradient_scope='none', gradient_probe_calls=0,
                accepted_guidance_calls=sum(e['accepted'] for e in events),
                static_cache_builds=self.cache_builds, static_cache_reuses=self.cache_reuses,
                semantic_projection_calls=self.semantic_head_calls, replay_steps=self.replay_steps,
                diagnostic_zero_replay=self.diagnostic_zero_replay,
                diagnostic_reject_all=self.diagnostic_reject_all, permuted_spatial_control=self.permute)
        return result
