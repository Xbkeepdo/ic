"""Joint native-state ACG proposals on all layers after complete SADI.

The 17 directions are computed at one native point, then jointly replayed.
They are not assumed to remain exact after an earlier layer changes.
"""

from contextlib import ExitStack
import time
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features import qwen3_native_risk_i_el as native
from features.qwen3_acg_signal_feedback import Qwen3BatchACGSignal, acg_contexts, sar_ffn_drift, head_scales
from features.qwen3_sar_ems_final_target import LastPositionCache
from features.qwen3_sar_ems_head_gradient import ReadOnlyTailCache

POST = tuple(range(19, 36))


class BlockRiskReader(native.NativeRiskIEl):
    """Reuse the frozen 108-feature calculation and its already-computed norms."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_norms = {}
        self.signature_seconds = 0.

    def _static(self, layer, row):
        key = (layer, row)
        if key in self.static:
            return self.static[key]
        if key in self.shared_static:
            started = time.perf_counter()
            entry = self.shared_static[key]
            identity = (self.tokens, self.start, self.end, self.head.weight.data_ptr(), self.head.weight._version)
            assert entry['identity'] == identity
            assert torch.equal(entry['hidden_cpu'], self.visual[layer][row].cpu())
            assert torch.equal(entry['values_cpu'], self.values[layer][row].cpu())
            self.static[key] = entry
            self.cache_hits += 1
        else:
            entry = super()._static(layer, row)
            started = time.perf_counter()
            # Only the shared validation copies move to host RAM. The reader
            # retains native GPU inputs; no hidden state is saved to disk.
            entry['hidden_cpu'] = entry.pop('hidden').cpu()
            entry['values_cpu'] = entry.pop('values').cpu()
        self.signature_seconds += time.perf_counter()-started
        return entry

    def features(self, rows, tokens):
        original, calls = native.native_write_norms, []
        self.write_norms = {}
        def remember(values, attention, weight):
            slot, layer = divmod(len(calls), 36)
            assert slot < len(rows)
            assert weight.data_ptr() == self.decoder.layers[layer].self_attn.o_proj.weight.data_ptr()
            result = original(values, attention, weight)
            if layer in POST:
                self.write_norms[layer, rows[slot]] = result
            calls.append(layer)
            return result
        with patch.object(native, 'native_write_norms', remember):
            result = super().features(rows, tokens)
        assert len(calls) == 36*len(rows)
        assert len(self.write_norms) == len(POST)*len(rows)
        return result


class Qwen3BatchBlockACG(Qwen3BatchACGSignal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shared = self.reader.shared_static
        self.reader = BlockRiskReader(self.model, (self.start, self.end), self.nouns,
                                      owner=self, shared_static=shared)
        self.sources = self.targets = POST
        self.source_outputs, self.block_gradients = {}, {}
        # The parent capture otherwise retains 17 expanded visual caches.
        # Norms and semantics use the reader's compact GQA caches instead.
        self.visual_values = dict.fromkeys(POST)

    def _begin_step(self, module, args, kwargs):
        super()._begin_step(module, args, kwargs)
        self.source_outputs.clear()

    def _remember_input(self, index):
        def remember(_module, args, kwargs):
            if not self.replaying_tail:
                hidden = args[0] if args else kwargs['hidden_states']
                self.scratch[index] = dict(residual=hidden[:, -1:].contiguous())
        return remember

    def _after_layer(self, index):
        def remember(_module, _args, output):
            if not self.replaying_tail:
                self.source_outputs[index] = output[:, -1:].clone()
            return output
        return remember

    def _block_gradient(self, rows):
        old_enabled = self.enabled
        self.enabled = False
        self.replaying_tail = self._replaying = True
        try:
            with torch.inference_mode(False), torch.enable_grad(), ExitStack() as stack:
                residual = self.scratch[19]['ffn_residual'].detach().clone().requires_grad_(True)
                layer = self.decoder.layers[19]
                update = layer.mlp(layer.post_attention_layernorm(residual[:, None]))
                residuals = {19: residual}
                for index in POST[1:]:
                    def hook(_module, args, index=index):
                        residuals[index] = args[0]
                    stack.callback(self.decoder.layers[index].post_attention_layernorm.register_forward_pre_hook(hook).remove)
                current = residual[:, None]+update
                kwargs = dict(self.tail_kwargs, past_key_values=ReadOnlyTailCache(self.tail_kwargs['past_key_values']))
                kwargs['position_embeddings'] = tuple(x.detach().clone() for x in kwargs['position_embeddings'])
                kwargs['cache_position'] = kwargs['cache_position'].detach().clone()
                for tail in self.decoder.layers[20:]:
                    current = tail(current, **kwargs)
                assert set(residuals) == set(POST)
                for index in POST:
                    node = residuals[index] if index == 19 else residuals[index][:, 0]
                    assert torch.equal(node.detach(), self.scratch[index]['ffn_residual'])
                normalized = self.decoder.norm(current)
                difference = float((normalized.detach()-self.reference_hidden).abs().max())
                self.maximum_probe_hidden_difference = max(self.maximum_probe_hidden_difference, difference)
                assert difference == 0
                logits = self.head(normalized).float()[:, 0]
                objective = logits.log_softmax(-1)[rows, self.final_tokens[rows].detach().clone()].sum()
                gradients = torch.autograd.grad(objective, tuple(residuals[i] for i in POST))
                self.block_gradients = {index: (gradients[j][rows] if index == 19 else gradients[j][rows, 0]).detach()
                                        for j, index in enumerate(POST)}
                assert all(torch.isfinite(g).all() for g in self.block_gradients.values())
                self.gradient_probes += 1
        finally:
            self.enabled = old_enabled
            self.replaying_tail = self._replaying = False

    def _budget(self, _source):
        rows = []
        for row, (token, active) in enumerate(zip(self.final_tokens.tolist(), self.active.tolist(), strict=True)):
            if not active or token not in self.nouns or token in self.seen[row]:
                continue
            prefix = self.tokenizer.decode(self.output_history[row], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            assert len(self.output_history[row]) == self.step
            prior, canonical = self.history_parser.classes(prefix), self.token_classes[token]
            blocked = canonical in prior
            self.class_decisions.setdefault(row, []).append(dict(step=self.step, token=token,
                canonical=canonical, prior_classes=sorted(prior), blocked=blocked))
            if not blocked or self.force_probe_gate:
                rows.append(row)
        if not rows:
            return [], None, []
        real = self.probe(self.reader.features(rows, self.final_tokens[rows].tolist()))
        for row, probability in zip(rows, real.tolist(), strict=True):
            self.probe_scores.setdefault(row, []).append(probability)
            self.gate_events.setdefault(row, []).append(dict(step=self.step, token=int(self.final_tokens[row]),
                p_real=probability, threshold=self.real_threshold, positive=probability < self.real_threshold))
        chosen = list(range(len(rows))) if self.force_probe_gate else (real < self.real_threshold).nonzero().flatten().tolist()
        rows, real = [rows[i] for i in chosen], real[chosen]
        if not rows or not self.gamma:
            return [], None, []
        if self.signal_guided:
            self._block_gradient(rows)
        budgets, layers = {}, {}
        self.first_context = None
        for index in POST:
            state = self.scratch[index]
            conditional, direction, _ = acg_contexts(state['scores'], state['values'], self.start, self.end)
            if index == 19:
                self.first_context = conditional
            direction = direction[rows]
            if self.signal_guided:
                projected = (self.block_gradients[index].float()@self.decoder.layers[index].self_attn.o_proj.weight.float()).reshape(len(rows), direction.shape[1], -1)
                drift = torch.stack([sar_ffn_drift(self.reader.write_norms[index, row],
                    self.reader._static(index, row)['log_s'][:, self.reader.columns[int(self.final_tokens[row])]],
                    self.reader.visual[index][row], state['update'][row]) for row in rows])
                scales, predicted, effects = head_scales(direction, projected, drift, self.gamma)
            else:
                drift = direction.new_zeros(len(rows))
                scales = direction.new_full(direction.shape[:2], self.gamma)
                predicted = effects = None
            budgets[index] = scales[:, :, None]*direction
            layers[index] = [dict(drift=float(drift[i]), head_scales=scales[i].tolist(),
                nonzero_heads=int((scales[i]>0).sum()), maximum_head_scale=float(scales[i].max()),
                budget_max=float(budgets[index][i].abs().max()),
                **(dict(predicted_log_probability_change=float(predicted[i]), head_effects=effects[i].tolist())
                   if predicted is not None else {})) for i in range(len(rows))]
        maximum = torch.stack([b.abs().amax((1, 2)) for b in budgets.values()]).amax(0)
        effective = (maximum > 0).nonzero().flatten().tolist()
        details = [dict(token=int(self.final_tokens[rows[i]]), gain=0., drift=sum(layers[k][i]['drift'] for k in POST)/len(POST),
            budget_max=float(maximum[i]), probe_p_real=float(real[i]), probe_real_threshold=self.real_threshold,
            signal_guided=self.signal_guided, joint_layers=list(POST),
            layer_details={str(k): layers[k][i] for k in POST},
            **(dict(predicted_log_probability_change=sum(layers[k][i]['predicted_log_probability_change'] for k in POST))
               if self.signal_guided else {})) for i in effective]
        return [rows[i] for i in effective], {k: b[effective] for k, b in budgets.items()}, details

    def _replay(self, rows, budgets):
        layer = self.decoder.layers[19]
        context = self.first_context.clone()
        context[rows] = (context[rows].float()+budgets[19]).to(context.dtype)
        context = context[:, :, None].transpose(1, 2).contiguous().flatten(2)
        attended = self.scratch[19]['residual']+layer.self_attn.o_proj(context)
        old_enabled = self.enabled
        self.enabled = False
        self.replaying_tail = self._replaying = True
        try:
            corrected = attended+layer.mlp(layer.post_attention_layernorm(attended))
            current = self.source_outputs[19].clone()
            current[rows] = corrected[rows]
            cache = LastPositionCache(self.tail_kwargs['past_key_values'], rows, check_identical=self.diagnostic_zero_replay)
            kwargs = dict(self.tail_kwargs, past_key_values=cache)
            original = qwen.eager_attention_forward
            def inject(module, *args, **kwargs):
                output, attention = original(module, *args, **kwargs)
                index = getattr(module, 'layer_idx', None)
                if isinstance(module, qwen.Qwen3VLTextAttention) and index in POST[1:]:
                    assert output.shape[1] == 1
                    output = output.clone()
                    output[rows, 0] = (output[rows, 0].float()+budgets[index]).to(output.dtype)
                return output, attention
            with patch.object(qwen, 'eager_attention_forward', inject):
                for tail in self.decoder.layers[20:]:
                    current = tail(current, **kwargs)
            assert cache.updates == list(range(20, 36))
            return self.decoder.norm(current)
        finally:
            self.enabled = old_enabled
            self.replaying_tail = self._replaying = False

    def _finish_decoder(self, _module, _args, output):
        if self.step == 0:
            self.scratch.clear()
            return output
        self.reference_hidden = output.last_hidden_state[:, -1:]
        native_logits = self.head(self.reference_hidden).float()[:, 0]
        self.final_tokens = native_logits.argmax(-1)
        if self.diagnostic_zero_replay:
            rows = self.active.nonzero().flatten().tolist()
            conditional, _, _ = acg_contexts(self.scratch[19]['scores'], self.scratch[19]['values'], self.start, self.end)
            self.first_context = conditional
            budgets = {k: torch.zeros_like(conditional[rows], dtype=torch.float32) for k in POST}
            details = [dict(token=int(self.final_tokens[row]), gain=0., drift=0., budget_max=0., joint_layers=list(POST)) for row in rows]
        else:
            rows, budgets, details = self._budget(19)
        if not rows:
            self.scratch.clear()
            return output
        cache = self.tail_kwargs['past_key_values']
        backup = {k: (cache.layers[k].keys[rows, :, -1:].clone(), cache.layers[k].values[rows, :, -1:].clone()) for k in range(20, 36)}
        corrected = self._replay(rows, budgets)
        hidden = output.last_hidden_state.clone()
        after = self.head(corrected).float()[:, 0]
        before_logp, after_logp = native_logits.log_softmax(-1), after.log_softmax(-1)
        if self.diagnostic_zero_replay:
            assert torch.equal(corrected[rows, 0], self.reference_hidden[rows, 0])
        rejected = []
        for i, (row, detail) in enumerate(zip(rows, details, strict=True)):
            target = detail['token']
            change = float(after_logp[row, target]-before_logp[row, target])
            keep = change < 0 and not self.diagnostic_reject_all and not self.diagnostic_zero_replay
            top2, top_ids = native_logits[row].topk(2)
            proposed = int(after[row].argmax())
            detail.update(actual_log_probability_change=change, accepted=keep,
                native_runner_up=int(top_ids[1]), native_top2_gap=float(top2[0]-top2[1]),
                original_target_log_probability=float(before_logp[row, target]), proposed_token=proposed,
                immediate_token_changed=proposed != target)
            if keep:
                hidden[row, -1] = corrected[row, 0]
            else:
                rejected.append(i)
        if rejected:
            indices = [rows[i] for i in rejected]
            for k, (keys, values) in backup.items():
                cache.layers[k].keys[indices, :, -1:] = keys[rejected]
                cache.layers[k].values[indices, :, -1:] = values[rejected]
        output.last_hidden_state = hidden
        self._record(19, 19, rows, details)
        self.replay_steps += 1
        self.scratch.clear()
        return output

    def statistics(self, responses):
        result = super().statistics(responses)
        for stat in result:
            events = stat['recycle_events']
            stat.update(intervention_layer='joint19--35', intervention_layers=list(POST),
                replayed_tail_layers=list(range(20, 36)), acg_scope='SAM-gated joint post-SADI block adaptation.',
                gradient_sources=list(POST), one_backward_for_all_layers=True,
                common_native_point=True, shared_validation_in_host_RAM=True,
                semantic_signature_seconds=self.reader.signature_seconds)
            stat['by_layer'] = {str(k): dict(source_layer=k, target_layer=k,
                joint_proposals=len(events),
                nonzero_budget_events=sum(e.get('layer_details', {}).get(str(k), {}).get('budget_max', 0.) > 0 for e in events),
                accepted_nonzero_events=sum(e.get('accepted', False) and e.get('layer_details', {}).get(str(k), {}).get('budget_max', 0.) > 0 for e in events)) for k in POST}
        return result
