"""Original SAM-V2 gate and semantic donor attention after complete SADI."""

import torch
from torch.nn import functional as F

from features.qwen3_native_risk_i_el import NativeRiskIEl
from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
from features.qwen3_semantic_donor_transport import relative_class_support, semantic_donor_transport


class Qwen3BatchSemanticDonor(Qwen3BatchProbeFirstClass):
    def __init__(self, *args, transport_mode='semantic', real_threshold, shared_static=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transport_mode = transport_mode
        self.real_threshold = float(real_threshold)
        assert 0 < self.real_threshold < 1
        self.reader = NativeRiskIEl(self.model, (self.start, self.end), self.nouns,
                                   owner=self, shared_static=shared_static)
        self.class_columns = {}
        for token, canonical in self.token_classes.items():
            self.class_columns.setdefault(canonical, []).append(self.reader.columns[token])
        self.gate_events = {}

    def _budget(self, source):
        # Class-history filtering is copied explicitly so the direct original
        # probability threshold never passes through a surrogate score adapter.
        rows = []
        for row, (token, active) in enumerate(zip(self.final_tokens.tolist(), self.active.tolist(), strict=True)):
            if not active or token not in self.nouns or token in self.seen[row]:
                continue
            assert len(self.output_history[row]) == self.step
            prefix = self.tokenizer.decode(self.output_history[row], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            prior, canonical = self.history_parser.classes(prefix), self.token_classes[token]
            blocked = canonical in prior
            self.class_decisions.setdefault(row, []).append(dict(step=self.step, token=token,
                canonical=canonical, prior_classes=sorted(prior), blocked=blocked))
            if not blocked or self.force_probe_gate:
                rows.append(row)
        if not rows:
            return [], None, []
        targets = self.final_tokens[rows].tolist()
        real = self.probe(self.reader.features(rows, targets))
        for row, probability in zip(rows, real.tolist(), strict=True):
            self.probe_scores.setdefault(row, []).append(probability)
            self.gate_events.setdefault(row, []).append(dict(step=self.step,
                token=int(self.final_tokens[row]), p_real=probability, threshold=self.real_threshold,
                positive=probability < self.real_threshold))
        selected = list(range(len(rows))) if self.force_probe_gate else (real < self.real_threshold).nonzero().flatten().tolist()
        if not selected:
            return [], None, []
        rows, real = [rows[i] for i in selected], real[selected]
        state = self.scratch[source]
        gradient = self._gradient(source, rows).float()
        values = state['values'][rows].float()
        if source not in self.projections:
            self.projections[source] = self.decoder.layers[source].self_attn.o_proj.weight.float()
        projected = (gradient @ self.projections[source]).reshape(len(rows), values.shape[1], -1)
        effect = torch.einsum('bhkd,bhd->bhk', values, projected)
        actual = state['scores'][rows, :, 0].softmax(-1, dtype=torch.float32).to(state['values'].dtype).float()
        similarity = (self.visual_normalized[source][rows]
                      @ F.normalize(state['update'][rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity/.03).softmax(-1)
        supports = [relative_class_support(self.reader._static(source, row)['probabilities'],
                    self.class_columns[self.token_classes[int(self.final_tokens[row])]], self.head.weight.shape[0])
                    for row in rows]
        support = torch.stack(supports)
        writes = torch.stack([self.reader.last_write_norms[row] for row in rows])
        delta, measures = semantic_donor_transport(actual, effect, preference, writes, support,
            start=self.start, end=self.end, prompt_length=self.prompt_length, mode=self.transport_mode)
        effective = (delta.abs().amax((1, 2)) > 0).nonzero().flatten().tolist()
        if not effective:
            return [], None, []
        details = [dict(token=int(self.final_tokens[rows[i]]), gain=0., gain_used=False,
            drift=float(measures['drift'][i]), budget_max=float(delta[i].abs().max()),
            probe_p_real=float(real[i]), probe_real_threshold=self.real_threshold,
            unsupported_fraction=float(measures['unsupported_fraction'][i]),
            transfer_fraction=float(measures['amount'][i]), transport_mode=self.transport_mode,
            moved_history_mass=float(measures['history_mass'][i]),
            moved_visual_mass=float(measures['visual_mass'][i]),
            predicted_log_probability_change=float(measures['predicted_change'][i])) for i in effective]
        final_rows = [rows[i] for i in effective]
        cache = self.tail_kwargs['past_key_values']
        self.cache_backup = {layer: (cache.layers[layer].keys[final_rows, :, -1:].clone(),
                                    cache.layers[layer].values[final_rows, :, -1:].clone())
                             for layer in range(source+1, len(self.decoder.layers))}
        return final_rows, delta[effective], details

    def statistics(self, responses):
        result = super().statistics(responses)
        for row, stats in enumerate(result):
            probabilities = self.probe_scores.get(row, ())
            stats.update(probe='original_108_SAM_V2_native', probe_real_threshold=self.real_threshold,
                probe_positive_calls=sum(p < self.real_threshold for p in probabilities),
                transport_mode=self.transport_mode, source_only_threshold_calibration=True,
                gate_events=[e for e in self.gate_events.get(row, ()) if e['step'] < len(responses[row])],
                native_risk_solver_calls=self.reader.risk_solver_calls,
                native_feature_seconds=self.reader.feature_seconds, static_seconds=self.reader.static_seconds,
                static_head_calls=self.reader.head_calls, shared_static_hits=self.reader.cache_hits)
        return result
