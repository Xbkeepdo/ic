"""Select one post-SADI attention layer from target-conditioned readouts."""

from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from features.llava_decoding_evidence import TAU
from features.qwen3_probe_attention_recycle import normalize_nonnegative
from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
from features.qwen3_sar_ems_final_target import LastPositionCache
from features.qwen3_sar_ems_head_gradient import ReadOnlyTailCache
from features.qwen3_visual_write_contrast import readout_contrast


LAYERS = (19, 23, 27, 31)


def signed_candidate(scores, values, gradient, projection, visual_normalized, update,
                     update_gradient, start, end, prompt_length):
    """The original signed recycle action plus a comparable layer score."""
    projected = (gradient.float() @ projection).reshape(len(gradient), values.shape[1], -1)
    effect = torch.einsum('bhkd,bhd->bhk', values.float(), projected)
    actual = scores.softmax(-1, dtype=torch.float32).to(values.dtype).float()
    visual = actual[:, :, start:end]
    writes = visual * effect[:, :, start:end]
    negative = (-writes).clamp_min(0)
    valid_head = negative.amax(-1, keepdim=True) > torch.finfo(negative.dtype).tiny
    evidence = normalize_nonnegative(negative.sum(1))
    similarity = (visual_normalized @ F.normalize(update.float(), dim=-1).unsqueeze(-1)).squeeze(-1)
    preference = (similarity / TAU).softmax(-1)
    drift = (1 - (evidence * preference).sqrt().sum(-1)).clamp(0, 1)
    removed = .5 * drift[:, None, None] * actual[:, :, prompt_length:] * (
        effect[:, :, prompt_length:] > 0) * valid_head
    mass = removed.sum(-1, keepdim=True)
    deficit = (evidence - preference).clamp_min(0)
    spatial = .25 + .75 * deficit / deficit.amax(-1, keepdim=True).clamp_min(1e-12)
    destination = normalize_nonnegative(normalize_nonnegative(negative) * spatial[:, None])
    delta = torch.zeros_like(actual)
    delta[:, :, start:end] = mass * destination
    delta[:, :, prompt_length:] -= removed
    predicted = (delta * effect).sum((1, 2))
    negative_fraction = (negative.sum((1, 2)) / writes.abs().sum((1, 2)).clamp_min(
        torch.finfo(writes.dtype).tiny)).clamp(0, 1)
    ffn_push = (update.float() * update_gradient.float()).sum(-1)
    valid = delta.abs().amax((1, 2)) > 0
    score = ffn_push.clamp_min(0) * negative_fraction * drift * valid
    assert torch.isfinite(delta).all() and torch.isfinite(score).all()
    assert delta.sum(-1).abs().max() < 1e-6 and (actual + delta).min() >= -1e-7
    assert (predicted <= 1e-6).all()
    return delta, dict(score=score, ffn_target_push=ffn_push,
        negative_visual_fraction=negative_fraction, drift=drift, valid_recycle=valid,
        predicted_log_probability_change=predicted,
        moved_attention_mass=mass.sum((1, 2)), budget_max=delta.abs().amax((1, 2)))


class Qwen3BatchFFNLayerSelection(Qwen3BatchProbeFirstClass):
    def __init__(self, *args, selection_mode='adaptive', diagnostic_force_layer=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.guided and selection_mode in ('adaptive', 'fixed19', 'fixed31')
        assert diagnostic_force_layer is None or diagnostic_force_layer in LAYERS
        self.sources = self.targets = LAYERS
        self.selection_mode, self.diagnostic_force_layer = selection_mode, diagnostic_force_layer
        self.source_outputs, self.gradient_rows, self.layer_gradients = {}, (), {}
        self.layer_decisions = {}
        self.proposal_layers = []
        self.native_selected_attention_checks = 0

    def _begin_step(self, module, args, kwargs):
        super()._begin_step(module, args, kwargs)
        self.source_outputs.clear()

    def _remember_input(self, index):
        original = super()._remember_input(index)
        def remember(module, args, kwargs):
            if not self.replaying_tail:
                return original(module, args, kwargs)
        return remember

    def _after_layer(self, index):
        def remember(_module, _args, output):
            if not self.replaying_tail:
                self.source_outputs[index] = output[:, -1:].clone()
            return output
        return remember

    def _gradient(self, source, rows):
        assert source == 19
        old_enabled = self.enabled
        self.replaying_tail = self._replaying = True
        self.enabled = False
        try:
            with torch.inference_mode(False), torch.enable_grad(), ExitStack() as stack:
                residual = self.scratch[19]['ffn_residual'].detach().clone().requires_grad_(True)
                layer = self.decoder.layers[19]
                update = layer.mlp(layer.post_attention_layernorm(residual[:, None]))
                residuals, updates = {19: residual}, {19: update}
                def remember_input(index):
                    def hook(_module, args):
                        residuals[index] = args[0]
                    return hook
                def remember_output(index):
                    def hook(_module, _args, output):
                        updates[index] = output
                    return hook
                for index in LAYERS[1:]:
                    handle = self.decoder.layers[index].post_attention_layernorm.register_forward_pre_hook(remember_input(index))
                    stack.callback(handle.remove)
                    handle = self.decoder.layers[index].mlp.register_forward_hook(remember_output(index))
                    stack.callback(handle.remove)
                current = residual[:, None] + update
                kwargs = dict(self.tail_kwargs)
                kwargs['past_key_values'] = ReadOnlyTailCache(kwargs['past_key_values'])
                kwargs['position_embeddings'] = tuple(x.detach().clone() for x in kwargs['position_embeddings'])
                kwargs['cache_position'] = kwargs['cache_position'].detach().clone()
                for tail in self.decoder.layers[20:]:
                    current = tail(current, **kwargs)
                assert set(residuals) == set(updates) == set(LAYERS)
                for index in LAYERS:
                    actual_residual = residuals[index] if index == 19 else residuals[index][:, 0]
                    assert torch.equal(actual_residual.detach(), self.scratch[index]['ffn_residual'])
                    assert torch.equal(updates[index][:, 0].detach(), self.scratch[index]['update'])
                normalized = self.decoder.norm(current)
                difference = float((normalized.detach() - self.reference_hidden).abs().max())
                self.maximum_probe_hidden_difference = max(self.maximum_probe_hidden_difference, difference)
                assert difference == 0
                logits = self.head(normalized).float()[:, 0]
                objective = logits.log_softmax(-1)[rows, self.final_tokens[rows].detach().clone()].sum()
                nodes = tuple(residuals[i] for i in LAYERS) + tuple(updates[i] for i in LAYERS)
                gradients = torch.autograd.grad(objective, nodes)
                self.layer_gradients = {index: ((gradients[j][rows] if index == 19 else gradients[j][rows, 0]).detach(), gradients[j+4][rows, 0].detach())
                    for j, index in enumerate(LAYERS)}
                assert all(torch.isfinite(g).all() for pair in self.layer_gradients.values() for g in pair)
                self.gradient_rows = tuple(rows)
                self.gradient_probes += 1
        finally:
            self.enabled = old_enabled
            self.replaying_tail = self._replaying = False
        return self.layer_gradients[19][0]

    def _budget(self, source):
        self.gradient_rows, self.layer_gradients = (), {}
        parent_rows, parent_budget, _parent_details = super()._budget(19)
        if not self.gradient_rows:
            return [], None, []
        rows = list(self.gradient_rows)
        proposals, diagnostics = {}, {}
        for index in LAYERS:
            state = self.scratch[index]
            if index not in self.projections:
                self.projections[index] = self.decoder.layers[index].self_attn.o_proj.weight.float()
            proposals[index], diagnostics[index] = signed_candidate(state['scores'][rows, :, 0],
                state['values'][rows], self.layer_gradients[index][0], self.projections[index],
                self.visual_normalized[index][rows], state['update'][rows], self.layer_gradients[index][1],
                self.start, self.end, self.prompt_length)
        # This exact action is the frozen parent formula, not a retuned amount.
        expected_rows = [rows[i] for i in diagnostics[19]['valid_recycle'].nonzero().flatten().tolist()]
        assert expected_rows == parent_rows
        if parent_rows:
            positions = [rows.index(row) for row in parent_rows]
            assert torch.equal(proposals[19][positions], parent_budget)
        selected, chosen_budgets, details = [], [], []
        self.proposal_layers = []
        for position, row in enumerate(rows):
            options = {index: {key: (bool(value[position]) if key == 'valid_recycle' else float(value[position]))
                for key, value in diagnostics[index].items()} for index in LAYERS}
            if self.diagnostic_force_layer is not None:
                chosen = self.diagnostic_force_layer if options[self.diagnostic_force_layer]['valid_recycle'] else None
            elif self.selection_mode != 'adaptive':
                layer = 19 if self.selection_mode == 'fixed19' else 31
                chosen = layer if options[layer]['score'] > 0 else None
            else:
                chosen = max(LAYERS, key=lambda layer: options[layer]['score'])
                if options[chosen]['score'] <= 0:
                    chosen = None
            self.layer_decisions.setdefault(row, []).append(dict(step=self.step,
                token=int(self.final_tokens[row]), selected_layer=chosen, candidates=options))
            if chosen is None:
                continue
            visual, gain, _, _ = readout_contrast(self, chosen, [row], self.final_tokens[row:row+1, None])
            selected.append(row)
            self.proposal_layers.append(chosen)
            chosen_budgets.append(proposals[chosen][position])
            item = options[chosen]
            details.append(dict(token=int(self.final_tokens[row]), gain=float(gain[0, 0]),
                visual_support=float(visual[0, 0]), probe_p_real=self.probe_scores[row][-1],
                selected_layer=chosen, selection_score=item['score'], ffn_target_push=item['ffn_target_push'],
                negative_visual_fraction=item['negative_visual_fraction'], drift=item['drift'],
                budget_max=item['budget_max'], moved_attention_mass=item['moved_attention_mass'],
                predicted_log_probability_change=item['predicted_log_probability_change']))
        return selected, torch.stack(chosen_budgets) if selected else None, details

    def _replay(self, rows, budget):
        by_layer = {index: [j for j, layer in enumerate(self.proposal_layers) if layer == index] for index in LAYERS}
        current = self.source_outputs[19]
        if by_layer[19]:
            positions = by_layer[19]
            current = super()._recompute(19, current, [rows[j] for j in positions], budget[positions])
        original = qwen.eager_attention_forward
        old_enabled = self.enabled

        def attend(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            index = getattr(module, 'layer_idx', None)
            if not isinstance(module, qwen.Qwen3VLTextAttention) or index not in by_layer or index == 19 or not by_layer[index]:
                return original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
            assert not dropout and not module.training and query.shape[2] == 1
            keys = qwen.repeat_kv(key, module.num_key_value_groups)
            values = qwen.repeat_kv(value, module.num_key_value_groups)
            scores = (query @ keys.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., :keys.shape[-2]]
            positions = by_layer[index]
            selected_rows = [rows[j] for j in positions]
            assert torch.equal(scores[selected_rows], self.scratch[index]['scores'][selected_rows])
            assert torch.equal(values[selected_rows], self.scratch[index]['values'][selected_rows])
            self.native_selected_attention_checks += len(selected_rows)
            attention = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
            updated = attention[selected_rows, :, 0].float() + budget[positions]
            assert updated.min() >= -1e-7
            attention[selected_rows, :, 0] = updated.clamp_min(0).to(attention.dtype)
            return (attention @ values).transpose(1, 2).contiguous(), attention

        cache = LastPositionCache(self.tail_kwargs['past_key_values'], rows,
            check_identical=self.diagnostic_zero_replay)
        kwargs = dict(self.tail_kwargs, past_key_values=cache)
        self.enabled = False
        self.replaying_tail = self._replaying = True
        try:
            with patch.object(qwen, 'eager_attention_forward', attend):
                for tail in self.decoder.layers[20:]:
                    current = tail(current, **kwargs)
        finally:
            self.enabled = old_enabled
            self.replaying_tail = self._replaying = False
        assert cache.updates == list(range(20, 36))
        return self.decoder.norm(current)

    def _finish_decoder(self, _module, _args, output):
        if self.step == 0:
            self.scratch.clear()
            return output
        self.reference_hidden = output.last_hidden_state[:, -1:]
        native = self.head(self.reference_hidden).float()[:, 0]
        self.final_tokens = native.argmax(-1)
        if self.diagnostic_zero_replay:
            rows = self.active.nonzero().flatten().tolist()
            self.proposal_layers = [LAYERS[(row+self.step) % len(LAYERS)] for row in rows]
            budget = torch.zeros_like(self.scratch[19]['scores'][rows, :, 0], dtype=torch.float32)
            details = [dict(token=int(self.final_tokens[row]), gain=0., drift=0., budget_max=0.,
                selected_layer=layer) for row, layer in zip(rows, self.proposal_layers, strict=True)]
        else:
            rows, budget, details = self._budget(19)
        if not rows:
            self.scratch.clear()
            return output
        cache = self.tail_kwargs['past_key_values']
        backup = {index: (cache.layers[index].keys[rows, :, -1:].clone(),
            cache.layers[index].values[rows, :, -1:].clone()) for index in range(20, 36)}
        corrected = self._replay(rows, budget)
        hidden = output.last_hidden_state.clone()
        if self.diagnostic_zero_replay:
            assert torch.equal(corrected[rows, 0], self.reference_hidden[rows, 0])
            changes = native.new_zeros((len(native),))
        else:
            after, before = self.head(corrected).float()[:, 0].log_softmax(-1), native.log_softmax(-1)
            changes = after[range(len(native)), self.final_tokens] - before[range(len(native)), self.final_tokens]
        rejected = []
        for position, (row, detail) in enumerate(zip(rows, details, strict=True)):
            change = float(changes[row])
            accepted = change < 0 and not self.diagnostic_reject_all and not self.diagnostic_zero_replay
            detail.update(actual_log_probability_change=change, accepted=accepted)
            hidden[row, -1] = corrected[row, 0] if accepted else self.reference_hidden[row, 0]
            if not accepted:
                rejected.append(position)
            self._record(detail['selected_layer'], detail['selected_layer'], [row], [detail])
        if rejected:
            selected_rows = [rows[j] for j in rejected]
            for index, (keys, values) in backup.items():
                cache.layers[index].keys[selected_rows, :, -1:] = keys[rejected]
                cache.layers[index].values[selected_rows, :, -1:] = values[rejected]
        output.last_hidden_state = hidden
        self.replay_steps += 1
        self.scratch.clear()
        return output

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, (tokens, stat) in enumerate(zip(responses, result, strict=True)):
            stat.update(base_variant='sadi', local_I_before_base_edit='sadi',
                intervention_layer='selected_per_event', eligible_layers=list(LAYERS),
                selection_mode=self.selection_mode, diagnostic_force_layer=self.diagnostic_force_layer,
                layer_decisions=[d for d in self.layer_decisions.get(row, ()) if d['step'] < len(tokens)],
                native_selected_attention_checks=self.native_selected_attention_checks,
                gradient_sources=list(LAYERS), one_backward_for_all_layers=True,
                extra_model_forward_calls=0, extra_backward_calls=0,
                ffn_update_edited=False)
        return result
