"""Verify uncertain SADI entity predictions with a target-conditioned FFN replay."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama import modeling_llama as llama

from features.dgst_t import _cosine_distance_matrix
from features.llava_decoding_evidence import (
    CANDIDATE_COUNT,
    EARLY_WRITE_LAYER,
    EMS_RISK_DROP,
    FFN_EDIT_LAYER,
    REPEAT_PHRASE_LENGTH,
    SAR_EMS_ALTERNATIVES,
    SAR_EMS_LOGIT_WINDOW,
    SAR_SUPPORT_GAIN,
    TAU,
    WriteEvidence,
)
from features.llava_sar_ems_attention import visual_write_norms
from features.llava_sar_ems_selection import candidate_ems_problems, layer_candidate_risks


FFN_ROTATION_STRENGTH = 0.55
MIN_LOCAL_CONTRAST = 0.05
DIRECT_LOGIT_WINDOW = 4.0
DIRECT_GRAMMATICAL_WINDOW = 2.75
DIRECT_ALTERNATIVES = 6
DIRECT_CONTRAST_STRENGTH = 1.35
DIRECT_MIN_FFN_GAIN = 0.60
DIRECT_MIN_DRIFT = 0.20
DIRECT_MAX_LOGIT_SHIFT = 3.5
GRAMMATICAL_WORDS = frozenset(
    "a an the this that these those another some several many more most in on at from to into by "
    "with without above behind under near among across along around beside between through against "
    "over and or but as while of for is are was were be has have had there it they".split()
)


@dataclass
class ReplayProposal:
    ffn: torch.Tensor
    approved_tokens: frozenset[int]
    support: dict[int, float]
    drift_before: float
    drift_after: float
    risks: dict[int, float] | None = None
    ffn_gain: float | None = None


class SadiSarEmsReplay:
    """Use one causal FFN counterfactual only for an ambiguous final entity token.

    Capture inside the ordinary SADI decoding context. Its shared prefill must
    request SAR evidence so the visual keys, norms and semantics can be reused.
    The replay is used for scoring. Subsequent tokens receive the original KV.
    """

    def __init__(self, evidence: WriteEvidence, *, mode: str = "strict", tokenizer=None):
        if mode not in {"strict", "direct"} or (mode == "direct" and tokenizer is None):
            raise ValueError("Expected strict replay, or direct replay with a tokenizer")
        self.evidence = evidence
        self.mode = mode
        self.tokenizer = tokenizer
        self.grammar_cache: dict[int, bool] = {}
        self.attention: torch.Tensor | None = None
        self.hpre: torch.Tensor | None = None
        self.write_mid: torch.Tensor | None = None
        self.mid: torch.Tensor | None = None
        self.ffn: torch.Tensor | None = None
        self.replay_ffn: torch.Tensor | None = None
        self.candidate_steps = 0
        self.ems_comparison_steps = 0
        self.replay_steps = 0
        self.accepted_steps = 0

    @contextmanager
    def capture(self):
        original = llama.eager_attention_forward
        layers = self.evidence.decoder.layers
        write_layer = layers[EARLY_WRITE_LAYER]
        ffn_layer = write_layer if self.mode == "strict" else layers[FFN_EDIT_LAYER]

        def observe_attention(module, *args, **kwargs):
            output, weights = original(module, *args, **kwargs)
            if module.layer_idx == EARLY_WRITE_LAYER and self.replay_ffn is None:
                self.attention = weights[0, :, -1, self.evidence.start:self.evidence.end]
            return output, weights

        def observe_hpre(_layer, args, kwargs):
            if self.replay_ffn is None:
                states = args[0] if args else kwargs["hidden_states"]
                self.hpre = states[0, -1]

        def observe_mid(_module, args):
            if self.replay_ffn is None:
                self.mid = args[0][0, -1]

        def observe_write_mid(_module, args):
            if self.replay_ffn is None:
                self.write_mid = args[0][0, -1]

        def observe_ffn(_module, _args, output):
            if self.replay_ffn is None:
                self.ffn = output[0, -1]
                return None
            edited = output.clone()
            edited[0, -1] = self.replay_ffn
            return edited

        with ExitStack() as stack:
            stack.enter_context(patch.object(llama, "eager_attention_forward", observe_attention))
            if self.mode == "strict":
                hook = write_layer.register_forward_pre_hook(observe_hpre, with_kwargs=True)
            else:
                hook = write_layer.post_attention_layernorm.register_forward_pre_hook(observe_write_mid)
            stack.callback(hook.remove)
            hook = ffn_layer.post_attention_layernorm.register_forward_pre_hook(observe_mid)
            stack.callback(hook.remove)
            hook = ffn_layer.mlp.register_forward_hook(observe_ffn)
            stack.callback(hook.remove)
            yield

    def step(self, model, token, cache: DynamicCache, response: list[int]):
        prefix_length = cache.get_seq_length()
        base = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        scores = base.logits[0, -1].float()
        winner = int(scores.argmax().item())
        candidates = self._eligible_tokens(scores, winner, response)
        if len(candidates) < 2:
            return base, winner, None

        self.candidate_steps += 1
        proposal = (self._strict_proposal(candidates) if self.mode == "strict"
                    else self._direct_proposal(candidates, scores))
        if proposal is None:
            return base, winner, None

        # DynamicCache appends with concatenation. Prefix views can be shared
        # with the base branch without copying image/prompt KV or editing it.
        branch = copy(base.past_key_values)
        branch.layers = [copy(layer) for layer in branch.layers]
        branch.crop(prefix_length)
        self.replay_ffn = proposal.ffn
        self.replay_steps += 1
        try:
            replay = model(input_ids=token, past_key_values=branch, use_cache=True, logits_to_keep=1)
        finally:
            self.replay_ffn = None
        replay_scores = replay.logits[0, -1].float()
        selection = (replay_scores if self.mode == "strict"
                     else scores + DIRECT_CONTRAST_STRENGTH * (replay_scores - scores))
        alternative = int(selection.argmax().item())
        if alternative not in proposal.approved_tokens:
            return base, winner, None

        self.accepted_steps += 1
        change = {
            "step": len(response),
            "old_token": winner,
            "new_token": alternative,
            "original_gap": float(scores[winner] - scores[alternative]),
            "replay_gap": float(replay_scores[alternative] - replay_scores[winner]),
            "ffn_preference_drift_before": proposal.drift_before,
            "ffn_preference_drift_after": proposal.drift_after,
        }
        if proposal.risks is not None:
            change.update({
                "old_support": proposal.support[winner],
                "new_support": proposal.support[alternative],
                "old_ems_risk": proposal.risks[winner],
                "new_ems_risk": proposal.risks[alternative],
            })
        else:
            change.update({
                "old_signed_visual_logit": proposal.support[winner],
                "new_signed_visual_logit": proposal.support[alternative],
                "winner_ffn_logit_gain": proposal.ffn_gain,
                "contrast_gap": float(selection[alternative] - selection[winner]),
                "new_token_is_noun": alternative in self.evidence.nouns,
            })
        return base, alternative, change

    def _eligible_tokens(self, scores, winner: int, response: list[int]) -> list[int]:
        if winner not in self.evidence.nouns:
            return [winner]
        direct = self.mode == "direct"
        window = DIRECT_LOGIT_WINDOW if direct else SAR_EMS_LOGIT_WINDOW
        scores_top, ids_top = scores.topk(min(CANDIDATE_COUNT, scores.numel()))
        ids_top = ids_top.masked_fill(scores_top < scores[winner] - window, -1)
        phrase_length = REPEAT_PHRASE_LENGTH - 1
        suffix = response[-phrase_length:]
        repeated_endings = {
            response[i] for i in range(phrase_length, len(response))
            if response[i - phrase_length:i] == suffix
        }
        repeated_endings.update(response[-2:])
        alternatives = [
            token for token in ids_top.tolist()
            if token >= 0 and token != winner and token not in repeated_endings
            and (token in self.evidence.nouns or (direct and response and self._is_grammatical(token)))
        ]
        limit = DIRECT_ALTERNATIVES if direct else SAR_EMS_ALTERNATIVES
        return [winner, *alternatives[:limit]]

    def _is_grammatical(self, token: int) -> bool:
        from nltk.corpus import wordnet

        if token not in self.grammar_cache:
            piece = self.tokenizer.convert_ids_to_tokens(token)
            word = piece.removeprefix("▁").lower() if isinstance(piece, str) else ""
            self.grammar_cache[token] = (
                word.isalpha() and word.isascii() and piece.startswith("▁")
                and token not in self.tokenizer.all_special_ids
                and (word in GRAMMATICAL_WORDS or any(synset.pos() != "n" for synset in wordnet.synsets(word)))
            )
        return self.grammar_cache[token]

    def _strict_proposal(self, candidates: list[int]) -> ReplayProposal | None:
        evidence = self.evidence
        layer = EARLY_WRITE_LAYER
        if self.ffn is None or self.mid is None or self.hpre is None or self.attention is None:
            raise RuntimeError("SAR/EMS replay requires the layer attention and FFN capture")

        terms = evidence._get_sar_patch_terms(layer)
        tokens = torch.tensor(candidates, device=self.ffn.device)
        semantics = terms.log_semantics(tokens)
        writes = visual_write_norms(terms.gram, self.attention)
        write_distribution = writes / writes.sum().clamp_min(1e-12)
        support = write_distribution @ semantics
        support_advantage = support[1:] > support[0] + SAR_SUPPORT_GAIN
        if not bool(support_advantage.any()):
            return None

        rows = evidence.head.weight[tokens].float()
        post = evidence.norm(self.mid + self.ffn).float()
        winner_gain = rows[0] @ (post - evidence.norm(self.mid).float())
        states = evidence.visual_states[layer]
        unit_ffn = F.normalize(self.ffn.float(), dim=0)
        preference = (states @ unit_ffn / TAU).softmax(dim=0)
        target = writes * (1 + semantics[:, 0])
        target = target / target.sum().clamp_min(1e-12)
        drift = (1 - (preference * target).sqrt().sum()).clamp(0, 1)

        # Move along the tangent to normalized FFN write directions. This is
        # the negative gradient of KL(target || preference), up to scale.
        direction = (target - preference) @ states
        tangent = direction - unit_ffn * (direction @ unit_ffn)
        strength = FFN_ROTATION_STRENGTH * drift.sqrt() * (winner_gain.clamp_min(0) / 2).tanh()
        edited_unit = F.normalize(unit_ffn + strength * F.normalize(tangent, dim=0), dim=0)
        edited_ffn = (edited_unit * self.ffn.float().norm()).to(self.ffn.dtype)
        new_preference = (states @ F.normalize(edited_ffn.float(), dim=0) / TAU).softmax(dim=0)
        new_drift = (1 - (new_preference * target).sqrt().sum()).clamp(0, 1)
        logit_shift = rows @ (evidence.norm(self.mid + edited_ffn).float() - post)
        promising = (support_advantage & (logit_shift[1:] - logit_shift[0] > MIN_LOCAL_CONTRAST)
                     & (winner_gain > 0) & (new_drift < drift))
        columns = [0, *(index + 1 for index in promising.nonzero().flatten().tolist())]
        if len(columns) < 2:
            return None

        self.ems_comparison_steps += 1
        visual_hpre = evidence.sar_visual_hpre[layer]
        visual_cost = evidence.sar_ems_visual_costs.get(layer)
        if visual_cost is None:
            visual_cost = _cosine_distance_matrix(visual_hpre)
            evidence.sar_ems_visual_costs[layer] = visual_cost
        problems = candidate_ems_problems(
            visual_hpre=visual_hpre, prediction_hpre=self.hpre, ffn=self.ffn,
            attention=self.attention, log_semantics=semantics[:, columns], visual_cost=visual_cost,
        )
        risks = [value[0] for value in layer_candidate_risks([problems], len(columns))]
        approved = {candidates[column]: risk for column, risk in zip(columns[1:], risks[1:], strict=True)
                    if risk < risks[0] - EMS_RISK_DROP}
        if not approved:
            return None
        approved_tokens = frozenset(approved)
        approved[candidates[0]] = risks[0]
        return ReplayProposal(edited_ffn, approved_tokens, dict(zip(candidates, support.tolist(), strict=True)),
                              float(drift), float(new_drift), risks=approved)

    def _direct_proposal(self, candidates: list[int], scores) -> ReplayProposal | None:
        evidence = self.evidence
        if self.ffn is None or self.mid is None or self.write_mid is None or self.attention is None:
            raise RuntimeError("Direct replay requires the two layers' attention and FFN capture")

        tokens = torch.tensor(candidates, device=self.ffn.device)
        rows = evidence.head.weight[tokens].float()
        residual = self.mid + self.ffn
        post = evidence.norm(residual).float()
        winner_gain = rows[0] @ (post - evidence.norm(self.mid).float())
        if not bool(winner_gain > DIRECT_MIN_FFN_GAIN):
            return None

        # Sum the actual visual values written by attention. The counterfactual
        # drops their additive residual update while holding attention fixed.
        values = evidence.visual_values[EARLY_WRITE_LAYER]
        projection = evidence.decoder.layers[EARLY_WRITE_LAYER].self_attn.o_proj.weight
        image_write = F.linear((self.attention[:, None, :] @ values).flatten(), projection)
        signed_image = rows @ (evidence.norm(self.write_mid).float()
                               - evidence.norm(self.write_mid - image_write).float())
        if not bool(signed_image[0] < 0.15 + 0.25 * winner_gain):
            return None

        patch_readout = evidence._project_per_patch(EARLY_WRITE_LAYER, [candidates[0]])[..., 0]
        positive = (self.attention.float() * patch_readout.float()).sum(dim=0).clamp_min(0)
        target = positive / positive.sum().clamp_min(1e-12)
        states = evidence.visual_states[FFN_EDIT_LAYER]
        ffn = self.ffn.float()
        preference = (states @ F.normalize(ffn, dim=0) / TAU).softmax(dim=0)
        drift = (1 - (preference * target).sqrt().sum()).clamp(0, 1)
        if not bool(drift >= DIRECT_MIN_DRIFT):
            return None

        # The local gradient of the target's RMS-normalized logit gives a
        # direction to attenuate without shrinking the entire FFN output.
        residual = residual.float()
        rms_squared = residual.square().mean() + evidence.norm.variance_epsilon
        readout = rows[0] * evidence.norm.weight.float()
        gradient = (readout - residual * (readout @ residual) / (residual.numel() * rms_squared))
        gradient = gradient / rms_squared.sqrt()
        direction = F.normalize(gradient, dim=0)
        component = (ffn @ direction).clamp_min(0)
        reduction = torch.minimum(
            0.8 * (0.5 + 0.5 * drift) * component,
            DIRECT_MAX_LOGIT_SHIFT / gradient.norm().clamp_min(1e-12),
        )
        edited_ffn = (ffn - reduction * direction).to(self.ffn.dtype)
        logit_shift = rows @ (evidence.norm(self.mid + edited_ffn).float() - post)

        noun = torch.tensor([token in evidence.nouns for token in candidates[1:]], device=tokens.device)
        noun_supported = (signed_image[1:] > (signed_image[0] + 0.15).clamp_min(0.10)) & noun
        grammatical = ((~noun) & (signed_image[0] <= 0.15) & (winner_gain >= 0.9) & (drift >= 0.28)
                       & (scores[candidates[0]] - scores[tokens[1:]] <= DIRECT_GRAMMATICAL_WINDOW))
        eligible = (noun_supported | grammatical) & (logit_shift[1:] - logit_shift[0] > MIN_LOCAL_CONTRAST)
        approved = frozenset(candidates[index + 1] for index in eligible.nonzero().flatten().tolist())
        if not approved:
            return None

        new_preference = (states @ F.normalize(edited_ffn.float(), dim=0) / TAU).softmax(dim=0)
        new_drift = (1 - (new_preference * target).sqrt().sum()).clamp(0, 1)
        return ReplayProposal(edited_ffn, approved, dict(zip(candidates, signed_image.tolist(), strict=True)),
                              float(drift), float(new_drift), ffn_gain=float(winner_gain))

    def statistics(self) -> dict[str, int]:
        return {
            "candidate_steps": self.candidate_steps,
            "ems_comparison_steps": self.ems_comparison_steps,
            "replay_steps": self.replay_steps,
            "accepted_steps": self.accepted_steps,
        }
