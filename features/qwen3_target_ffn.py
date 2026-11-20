"""Condition the FFN spatial preference on its contribution to the final target.

The complete SADI base and the existing layer19 attention action are retained.
The FFN itself is never edited. Its output gradient is obtained in the same
backward pass already used for the attention-residual gradient.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from features import qwen3_probe_attention_recycle as recycle
from features.llava_decoding_evidence import TAU
from features.qwen3_probe_first_class import Qwen3BatchProbeFirstClass
from features.qwen3_sar_ems_head_gradient import ReadOnlyTailCache


def target_promoting_component(update, output_gradient):
    """Keep coordinates with positive local contribution to target log p.

    An empty component gives the zero direction, hence a uniform preference
    after the existing cosine/softmax rule. No new threshold is introduced.
    """
    assert update.shape == output_gradient.shape and update.ndim == 2
    assert torch.isfinite(update).all() and torch.isfinite(output_gradient).all()
    attribution = update * output_gradient
    return update.masked_fill(attribution <= 0, 0), attribution


class Qwen3BatchTargetFFN(Qwen3BatchProbeFirstClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.guided
        self.ffn_output_gradient = None
        self.ffn_gradient_rows = ()
        self.ffn_output_gradient_reads = self.target_ffn_preference_reads = 0

    def _gradient(self, source, rows):
        self.replaying_tail = self._replaying = True
        try:
            with torch.inference_mode(False), torch.enable_grad():
                residual = self.scratch[source]['ffn_residual'].detach().clone().requires_grad_(True)
                layer = self.decoder.layers[source]
                update = layer.mlp(layer.post_attention_layernorm(residual[:, None]))
                assert torch.equal(update[:, 0].detach(), self.scratch[source]['update'])
                current = residual[:, None] + update
                kwargs = dict(self.tail_kwargs)
                kwargs['past_key_values'] = ReadOnlyTailCache(kwargs['past_key_values'])
                kwargs['position_embeddings'] = tuple(x.detach().clone() for x in kwargs['position_embeddings'])
                kwargs['cache_position'] = kwargs['cache_position'].detach().clone()
                for tail in self.decoder.layers[source + 1:]:
                    current = tail(current, **kwargs)
                normalized = self.decoder.norm(current)
                logits = self.head(normalized).float()[:, 0]
                targets = self.final_tokens[rows].detach().clone()
                objective = logits.log_softmax(-1)[rows, targets]
                gradient, update_gradient = torch.autograd.grad(objective.sum(), (residual, update))
                gradient = gradient[rows].detach()
                self.ffn_output_gradient = update_gradient[rows, 0].detach().float()
                self.ffn_gradient_rows = tuple(rows)
                self.ffn_output_gradient_reads += 1
                difference = (normalized.detach() - self.reference_hidden).abs().max().item()
                self.maximum_probe_hidden_difference = max(self.maximum_probe_hidden_difference, difference)
                self.gradient_probes += 1
        finally:
            self.replaying_tail = self._replaying = False
        assert torch.isfinite(gradient).all() and torch.isfinite(self.ffn_output_gradient).all()
        return gradient

    def _budget(self, source):
        normalize, evidence = recycle.normalize_nonnegative, []
        captures = []

        def remember_evidence(values):
            result = normalize(values)
            if values.ndim == 2:
                evidence.append(result.detach())
            return result

        def normalize_update(update, *args, **kwargs):
            rows = list(self.ffn_gradient_rows)
            assert torch.equal(update, self.scratch[source]['update'][rows].float())
            assert len(evidence) == 1 and not captures
            component, attribution = target_promoting_component(update, self.ffn_output_gradient)
            conditioned = F.normalize(component, *args, **kwargs)
            patches = self.visual_normalized[source][rows]
            old_p = ((patches @ F.normalize(update, dim=-1).unsqueeze(-1)).squeeze(-1) / TAU).softmax(-1)
            new_p = ((patches @ conditioned.unsqueeze(-1)).squeeze(-1) / TAU).softmax(-1)
            old_d = (1 - (evidence[0] * old_p).sqrt().sum(-1)).clamp(0, 1)
            captures.extend(dict(
                target_conditioned_ffn_preference=True,
                ffn_gradient_scope='source_FFN_output_to_final_log_probability',
                ffn_positive_coordinates=int((attribution[i] > 0).sum()),
                ffn_positive_attribution=float(attribution[i].clamp_min(0).sum()),
                ffn_negative_attribution=float((-attribution[i]).clamp_min(0).sum()),
                ffn_preference_l1_change=float((new_p[i] - old_p[i]).abs().sum()),
                reference_ffn_drift=float(old_d[i])) for i in range(len(rows)))
            return conditioned

        # Change only this module's normalization reference. Do not patch the
        # shared torch.nn.functional object used by the rest of the model.
        with patch.object(recycle, 'normalize_nonnegative', remember_evidence), \
                patch.object(recycle, 'F', SimpleNamespace(normalize=normalize_update)):
            rows, budget, details = super()._budget(source)
        self.target_ffn_preference_reads += bool(captures)
        for row, detail in zip(rows, details, strict=True):
            detail.update(captures[self.ffn_gradient_rows.index(row)])
        return rows, budget, details

    def statistics(self, responses):
        result = super().statistics(responses)
        for stat in result:
            stat.update(base_variant='sadi', local_I_before_base_edit='sadi',
                target_conditioned_ffn_preference=True,
                ffn_output_gradient_reads=self.ffn_output_gradient_reads,
                target_ffn_preference_reads=self.target_ffn_preference_reads,
                ffn_update_edited=False, additional_gradient_backward_calls=0,
                additional_model_forward_calls=0, additional_mlp_forward_calls=0,
                additional_lm_head_calls=0)
        return result
