"""Keep an attention proposal only if FFN preference approaches its fixed evidence."""

from unittest.mock import patch

import torch
from torch.nn import functional as F

from features import qwen3_probe_attention_recycle as recycle
from features.llava_decoding_evidence import TAU
from features.qwen3_probe_svar_first_class import Qwen3BatchSvarFirstClass


class Qwen3BatchSvarFFNAlignment(Qwen3BatchSvarFirstClass):
    def __init__(self, *args, diagnostic_reject_ffn_alignment=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_reject_ffn_alignment = diagnostic_reject_ffn_alignment
        self.feedback_rows = ()
        self.feedback_evidence = self.feedback_update = None
        self.fixed_evidence_reads = self.alignment_readouts = 0

    def _budget(self, source):
        gradient, normalize = self._gradient, recycle.normalize_nonnegative
        gradient_rows, evidence = [], []

        def remember_gradient(index, rows):
            assert index == source
            gradient_rows[:] = rows
            return gradient(index, rows)

        def remember_evidence(values):
            result = normalize(values)
            if values.ndim == 2:
                evidence.append(result.detach())
            return result

        self._gradient = remember_gradient
        try:
            with patch.object(recycle, 'normalize_nonnegative', remember_evidence):
                rows, budget, details = super()._budget(source)
        finally:
            self._gradient = gradient
        self.fixed_evidence_reads += len(evidence)
        self.feedback_rows = tuple(rows)
        if rows:
            assert len(evidence) == 1 and len(gradient_rows) == evidence[0].shape[0]
            positions = [gradient_rows.index(row) for row in rows]
            self.feedback_evidence = evidence[0][positions]
            assert torch.isfinite(self.feedback_evidence).all()
            assert (self.feedback_evidence.sum(-1) - 1).abs().max() < 1e-6
        return rows, budget, details

    def _recompute(self, index, original_output, rows, budget):
        updates = []

        def remember_update(_module, _args, output):
            updates.append(output[:, -1].detach())

        # Read the source FFN call already made by the unchanged attention action.
        handle = self.decoder.layers[index].mlp.register_forward_hook(remember_update)
        try:
            result = super()._recompute(index, original_output, rows, budget)
        finally:
            handle.remove()
        assert len(updates) == 1
        self.feedback_update = updates[0]
        return result

    def _finish_decoder(self, module, args, output):
        self.feedback_rows = ()
        self.feedback_evidence = self.feedback_update = None
        result = super()._finish_decoder(module, args, output)
        if self.diagnostic_zero_replay or not self.feedback_rows:
            return result
        rows = list(self.feedback_rows)
        source, = self.sources
        assert source == 19 and self.feedback_update is not None
        similarity = (self.visual_normalized[source][rows]
            @ F.normalize(self.feedback_update[rows].float(), dim=-1).unsqueeze(-1)).squeeze(-1)
        preference = (similarity / TAU).softmax(-1)
        after = (1 - (self.feedback_evidence * preference).sqrt().sum(-1)).clamp(0, 1)
        assert torch.isfinite(after).all()
        self.alignment_readouts += 1
        evidence_saved = self.feedback_evidence.cpu().tolist()
        preference_saved = preference.cpu().tolist()
        rejected_rows, rejected_positions = [], []
        for position, row in enumerate(rows):
            event = self.events[row][-1]
            assert event['step'] == self.step
            improved = float(after[position]) < event['drift']
            parent_accepted = event['accepted']
            rejected = parent_accepted and (not improved or self.diagnostic_reject_ffn_alignment)
            event.update(probability_decrease_accepted=parent_accepted,
                ffn_drift_after=float(after[position]), fixed_negative_evidence=True,
                ffn_negative_evidence_before=evidence_saved[position],
                ffn_preference_after=preference_saved[position],
                ffn_drift_improved=improved, rejected_ffn_alignment=rejected,
                diagnostic_reject_ffn_alignment=self.diagnostic_reject_ffn_alignment)
            if rejected:
                event['accepted'] = False
                result.last_hidden_state[row, -1] = self.reference_hidden[row, 0]
                rejected_rows.append(row)
                rejected_positions.append(position)
        if rejected_rows:
            cache = self.tail_kwargs['past_key_values']
            for layer, (keys, values) in self.cache_backup.items():
                cache.layers[layer].keys[rejected_rows, :, -1:] = keys[rejected_positions]
                cache.layers[layer].values[rejected_rows, :, -1:] = values[rejected_positions]
        return result

    def statistics(self, responses):
        result = super().statistics(responses)
        for tokens, stats, row in zip(responses, result, range(len(responses)), strict=True):
            events = [e for e in self.events.get(row, ()) if e['step'] < len(tokens) and 'accepted' in e]
            stats.update(acceptance_requires_ffn_alignment=True,
                fixed_negative_evidence_before_proposal=True, ffn_alignment_layer=19,
                diagnostic_reject_ffn_alignment=self.diagnostic_reject_ffn_alignment,
                fixed_evidence_reads=self.fixed_evidence_reads,
                ffn_alignment_readouts=self.alignment_readouts,
                rejected_ffn_alignment_calls=sum(e.get('rejected_ffn_alignment', False) for e in events),
                ffn_drift_improves_in_proposals=sum(e.get('ffn_drift_improved', False) for e in events),
                probability_decrease_proposals=sum(e['actual_log_probability_change'] < 0 for e in events),
                additional_model_forward_calls=0, additional_mlp_forward_calls=0,
                additional_lm_head_calls=0)
        return result
