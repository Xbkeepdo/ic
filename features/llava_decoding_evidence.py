"""Visual attention write evidence and FFN disagreement for cached LLaVA decoding."""
from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import re
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.models.llama import modeling_llama as llama

from features.dgst_t import _cosine_distance_matrix
from features.llava_air import (
    AIR_LAYERS, AIR_LENS_LAYERS, AIR_TEXT_LAYERS, air_decode_attention, air_projection_gram,
)
from features.llava_attention_transfer import (
    projected_visual_head_scores, transfer_repeat_attention, transfer_template_attention,
    write_evidence_distribution,
)
from features.llava_sadi import sadi_visual_logits
from features.llava_sar_ems_attention import (
    SarPatchTerms, guide_visual_logits_with_writes, mix_supported_visual_attention, mix_visual_write_prior,
    rebalance_visual_logits, visual_write_gram, visual_write_norms,
)
from features.llava_sar_ems_selection import candidate_ems_problems, layer_candidate_risks
from features.llava_sparc import sparc_attention
from features.llava_spin import SpinAttention


@dataclass(frozen=True)
class Intervention:
    attention: str = "native"
    attention_strength: float = 1.0
    support_strength: float = 0.0
    drift_strength: float = 0.0
    ffn_mode: str | None = None
    ffn_strength: float = 0.0
    sar_ems_attention_strength: float = 0.0
    sar_ems_patch_strength: float = 0.0
    write_prior_strength: float = 0.0
    write_prior_logit_scale: bool = False
    write_prior_logit_cap: float | None = None
    sar_ems_selection: bool = False
    calibrated_penalty: float = 0.0
    calibrated_candidates: int = 2
    replay: str | None = None
    repeat_mode: str | None = None
    template_transfer: str | None = None
    visual_head_gain: str | None = None
    repeat_attention: str | None = None
    sar_ems_allocation: float = 0.0
    sadi_head_rank: str | None = None
    permute_attention_guidance: bool = False

    @property
    def transfer_uses_writes(self) -> bool:
        return self.template_transfer in {"write", "write_ffn"} or self.repeat_attention == "early_write"

    @property
    def compares_sar_ems(self) -> bool:
        return self.sar_ems_selection or self.calibrated_penalty > 0.0

    @property
    def rerank(self) -> bool:
        return (self.support_strength > 0.0 or self.drift_strength > 0.0
                or self.compares_sar_ems or self.repeat_mode is not None)

    @property
    def needs_visual_cache(self) -> bool:
        return self.rerank or self.ffn_mode == "rotate" or self.uses_sar_ems or self.replay is not None

    @property
    def uses_sar_ems(self) -> bool:
        return bool(self.sar_ems_attention_strength or self.sar_ems_patch_strength
                    or self.sar_ems_allocation)


VARIANTS = {
    "greedy": Intervention(),
    "svar": Intervention(attention="svar"),
    "sadi": Intervention(attention="sadi"),
    "sadi_head_vhd": Intervention(attention="sadi", sadi_head_rank="vhd"),
    "sadi_head_projected": Intervention(attention="sadi", sadi_head_rank="projected"),
    "write_support": Intervention(support_strength=2.0),
    "write_ffn": Intervention(support_strength=2.0, drift_strength=3.0),
    "svar_write_ffn": Intervention(attention="svar", support_strength=2.0, drift_strength=3.0),
    "sadi_ffn_rotate": Intervention(attention="sadi", ffn_mode="rotate", ffn_strength=0.6),
    "sadi_ffn_attenuate": Intervention(attention="sadi", ffn_mode="attenuate", ffn_strength=0.5),
    "sadi_soft_rotate": Intervention(attention="sadi", attention_strength=0.7,
                                     ffn_mode="rotate", ffn_strength=0.6),
    "sadi_sar_ems_attention": Intervention(attention="sadi", sar_ems_attention_strength=5.0),
    "svar_sar_ems_allocation": Intervention(attention="svar", sar_ems_allocation=0.6),
    "sadi_sar_ems_allocation": Intervention(attention="sadi", sar_ems_allocation=0.6),
    "pai_attention": Intervention(attention="pai"),
    "pai_sar_ems_allocation": Intervention(attention="pai", sar_ems_allocation=0.6),
    "pade_attention": Intervention(attention="pade"),
    "pade_sar_ems_allocation": Intervention(attention="pade", sar_ems_allocation=0.6),
    "air_decode": Intervention(attention="air"),
    "air_decode_projected": Intervention(attention="air_projected"),
    "air_decode_write_agreement": Intervention(attention="air_agreement"),
    "spin": Intervention(attention="spin"),
    "spin_projected": Intervention(attention="spin_projected"),
    "svar_spin": Intervention(attention="svar_spin"),
    "svar_spin_projected": Intervention(attention="svar_spin_projected"),
    "sparc": Intervention(attention="sparc"),
    "sparc_write": Intervention(attention="sparc_write"),
    "vga": Intervention(),
    "vga_write": Intervention(),
    "vhr_local": Intervention(visual_head_gain="vhr"),
    "vhr_local_sar": Intervention(visual_head_gain="vhr_write"),
    "svar_sar_ems_attention": Intervention(attention="svar", sar_ems_attention_strength=5.0),
    "sar_ems_patch_mixture": Intervention(sar_ems_patch_strength=4.0),
    "sar_ems_write_prior": Intervention(sar_ems_patch_strength=4.0, write_prior_strength=6.0),
    "sar_ems_write_logits": Intervention(sar_ems_patch_strength=4.0, write_prior_strength=0.9,
                                         write_prior_logit_scale=True),
    "sar_ems_write_logits_bounded": Intervention(sar_ems_patch_strength=4.0, write_prior_strength=0.9,
                                                 write_prior_logit_scale=True, write_prior_logit_cap=2.0),
    "sar_ems_rerank": Intervention(sar_ems_selection=True),
    "sadi_sar_ems_rerank": Intervention(attention="sadi", sar_ems_selection=True),
    "sar_ems_calibrated": Intervention(calibrated_penalty=2.0),
    "sadi_sar_ems_calibrated": Intervention(attention="sadi", sar_ems_attention_strength=5.0,
                                             calibrated_penalty=2.0),
    "sadi_sar_ems_pool6": Intervention(attention="sadi", calibrated_penalty=6.0,
                                      calibrated_candidates=8),
    "sadi_sar_ems_pool10": Intervention(attention="sadi", calibrated_penalty=10.0,
                                       calibrated_candidates=8),
    "sadi_sar_ems_replay": Intervention(attention="sadi", replay="strict"),
    "sadi_write_ffn_replay": Intervention(attention="sadi", replay="direct"),
    "sadi_sar_ems_text_repeat": Intervention(attention="sadi", sar_ems_attention_strength=5.0,
                                              repeat_mode="text"),
    "sadi_sar_ems_visual_repeat": Intervention(attention="sadi", sar_ems_attention_strength=5.0,
                                                repeat_mode="visual"),
    "template_attention_transfer": Intervention(template_transfer="plain"),
    "template_sar_attention": Intervention(template_transfer="write"),
    "template_sar_ems_attention": Intervention(template_transfer="write_ffn"),
    "visual_head_mass_gain": Intervention(visual_head_gain="mass"),
    "visual_head_write_gain": Intervention(visual_head_gain="write"),
    "sadi_repeat_attention": Intervention(attention="sadi", sar_ems_attention_strength=5.0,
                                          repeat_attention="plain"),
    "sadi_sar_repeat_attention": Intervention(attention="sadi", sar_ems_attention_strength=5.0,
                                              repeat_attention="write"),
    "sadi_sar_sustained_attention": Intervention(attention="sadi", sar_ems_attention_strength=5.0,
                                                 repeat_attention="early_write"),
}
for guided_variant in (
    "sadi_sar_ems_attention", "svar_sar_ems_allocation", "sadi_sar_ems_allocation",
    "pai_sar_ems_allocation", "pade_sar_ems_allocation", "vhr_local_sar",
):
    VARIANTS[f"{guided_variant}_permuted"] = replace(
        VARIANTS[guided_variant], permute_attention_guidance=True
    )
EVIDENCE_LAYERS = (12, 18, 22)
EARLY_WRITE_LAYER = 18
FFN_EDIT_LAYER = 20
CAPTURE_LAYERS = tuple(sorted(set(EVIDENCE_LAYERS) | {FFN_EDIT_LAYER}))
ATTENTION_LAYERS = tuple(range(5, 19))
VHR_LAYERS = (1, *range(18, 32))
SADI_HEAD_TILT = 0.5
ALLOCATION_ANCHOR = 12
SAR_EMS_ATTENTION_LAYERS = (21, 22)
PATCH_MIXTURE_ANCHORS = (12, 18)
PATCH_MIXTURE_LAYERS = range(13, 23)
PATCH_MIXTURE_CANDIDATES = 8
WRITE_PRIOR_ANCHORS = (5, 10, 15)
VISUAL_HEAD_GAIN = 2.5
CANDIDATE_COUNT = 12
LOGIT_WINDOW = 3.0
TAU = 0.03
SAR_EMS_ALTERNATIVES = 3
SAR_EMS_LOGIT_WINDOW = 2.0
SAR_SUPPORT_GAIN = 0.10
EMS_RISK_DROP = 0.02
REPEAT_PHRASE_LENGTH = 4
REPEAT_LOGIT_PENALTY = 4.0
REPEAT_VISUAL_OVERLAP = 0.90
REPEAT_ATTENTION_SPAN = 8
REPEAT_ATTENTION_LAYERS = range(19, 25)

_PHYSICAL_NOUNS = {
    "noun.animal", "noun.artifact", "noun.body", "noun.food", "noun.location",
    "noun.object", "noun.person", "noun.plant", "noun.substance",
}


def _matched_sadi_head_logits(native, guided, selected_heads):
    """Vary each head's SADI correction while preserving its visual mass."""
    native_float = native.float()
    guided_float = guided.float()
    selected = selected_heads.float()
    gain = 1.0 + SADI_HEAD_TILT * (selected - selected.mean())
    shifted = native_float + (guided_float - native_float) * gain[None, :, None]
    offset = (torch.logsumexp(guided_float, dim=-1, keepdim=True)
              - torch.logsumexp(shifted, dim=-1, keepdim=True))
    return (shifted + offset).to(guided.dtype)


@lru_cache(maxsize=1)
def lexical_nouns(tokenizer, *, word_start: str = "▁") -> frozenset[int]:
    """Use general WordNet noun senses to screen word-initial candidates."""
    from nltk.corpus import wordnet

    pattern = re.compile(re.escape(word_start) + r"([a-zA-Z][a-zA-Z-]{2,})$")
    candidates = []
    for piece, token_id in tokenizer.get_vocab().items():
        match = pattern.fullmatch(piece)
        if match is None:
            continue
        senses = wordnet.synsets(match.group(1).lower(), pos=wordnet.NOUN)
        if any(sense.lexname() in _PHYSICAL_NOUNS for sense in senses):
            candidates.append(int(token_id))
    return frozenset(candidates)


def has_repeated_noun_phrase(tokens: list[int], nouns: frozenset[int]) -> bool:
    phrases = set()
    for end in range(REPEAT_PHRASE_LENGTH, len(tokens) + 1):
        if tokens[end - 1] not in nouns:
            continue
        phrase = tuple(tokens[end - REPEAT_PHRASE_LENGTH:end])
        if phrase in phrases:
            return True
        phrases.add(phrase)
    return False


def _repeat_attention_starts(response: list[int], nouns: frozenset[int], sustained: bool) -> list[int]:
    span = REPEAT_ATTENTION_SPAN
    end = len(response)
    boundary = end - span
    if boundary <= 0:
        return []
    fragment = response[boundary:]
    if not nouns.intersection(fragment):
        return []
    starts = [boundary]
    earliest = max(0, end - 64) if sustained else 0
    for index in range(boundary - 1, earliest - 1, -1):
        if sustained and index + span > starts[-1]:
            continue
        if response[index:index + span] == fragment:
            starts.append(index)
    return starts if len(starts) >= (3 if sustained else 2) else []


def has_repeat_attention_span(tokens: list[int], nouns: frozenset[int], *, sustained: bool = False) -> bool:
    if sustained:
        return any(_repeat_attention_starts(tokens[:end], nouns, True)
                   for end in range(3 * REPEAT_ATTENTION_SPAN, len(tokens)))
    seen = set()
    # The last completed token has no subsequent prediction in a cached caption.
    for end in range(REPEAT_ATTENTION_SPAN, len(tokens)):
        fragment = tuple(tokens[end - REPEAT_ATTENTION_SPAN:end])
        if nouns.intersection(fragment):
            if fragment in seen:
                return True
            seen.add(fragment)
    return False


class WriteEvidence:
    """Use a fixed image cache and inspect the current causal query only."""

    def __init__(self, model, nouns: Iterable[int], calibration: dict | None = None):
        self.model = model
        self.decoder = model.language_model
        self.nouns = frozenset(nouns)
        self.norm = self.decoder.norm
        self.head = model.get_output_embeddings()
        self.noun_mask = torch.zeros(self.head.weight.shape[0], dtype=torch.bool,
                                     device=self.head.weight.device)
        self.noun_mask[list(self.nouns)] = True
        self.calibration = calibration
        self.calibrated_tokens = self.nouns & frozenset(calibration["token_ids"]) if calibration else frozenset()
        self.readouts: dict[int, dict[int, torch.Tensor]] = {i: {} for i in EVIDENCE_LAYERS}
        self.air_grams: dict[int, torch.Tensor] = {}
        self.spin = SpinAttention()
        self.reset_image((0, 0))

    def reset_image(self, span: tuple[int, int]):
        self.start, self.end = span
        self.visual_states: dict[int, torch.Tensor] = {}
        self.visual_values: dict[int, torch.Tensor] = {}
        self.patch_readouts: dict[int, dict[int, torch.Tensor]] = {i: {} for i in EVIDENCE_LAYERS}
        self.attention_rows: dict[int, torch.Tensor] = {}
        self.ffn_outputs: dict[int, torch.Tensor] = {}
        self.after_attention: dict[int, torch.Tensor] = {}
        self.prediction_hpre: dict[int, torch.Tensor] = {}
        self.sar_ems_visual_costs: dict[int, torch.Tensor] = {}
        self.candidate_steps = 0
        self.ffn_steps: list[torch.Tensor] = []
        self.sar_visual_hpre: dict[int, torch.Tensor] = {}
        self.sar_patch_terms: dict[int, SarPatchTerms] = {}
        self.write_prior_grams: dict[int, torch.Tensor] = {}
        self.reset_sar_ems_steps()

    def reset_sar_ems_steps(self):
        self.sar_ems_budget: torch.Tensor | None = None
        self.patch_mixture: tuple[torch.Tensor, torch.Tensor] | None = None
        self.visual_prior: torch.Tensor | None = None
        self.sar_ems_step = 0
        self.sar_ems_targets: list[tuple[int, int]] = []
        self.ems_comparison_steps = 0
        self.repeat_evidence: dict[tuple[int, ...], torch.Tensor | None] = {}
        self.transfer_previous: torch.Tensor | None = None
        self.transfer_evidence: torch.Tensor | None = None
        self.transfer_mass: list[torch.Tensor] = []
        self.visual_head_masks: dict[int, torch.Tensor] = {}
        self.repeat_donor_offsets: torch.Tensor | None = None
        self.repeat_attention_evidence: torch.Tensor | None = None
        self.sar_ems_recipient: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pad_previous: torch.Tensor | None = None
        self.pad_accumulator: torch.Tensor | None = None

    def _remember_states(self, layer_index, save_sar_hpre):
        def hook(_layer, args, kwargs):
            states = args[0] if args else kwargs["hidden_states"]
            visual = states[0, self.start:self.end, :]
            self.visual_states[layer_index] = F.normalize(
                visual.float(), dim=-1
            ).contiguous()
            if save_sar_hpre:
                self.sar_visual_hpre[layer_index] = visual.contiguous()
        return hook

    def _remember_ffn(self, layer_index):
        def hook(_module, _args, output):
            self.ffn_outputs[layer_index] = output[0, -1, :]
        return hook

    def _remember_prediction(self, layer_index):
        def hook(_layer, args, kwargs):
            states = args[0] if args else kwargs["hidden_states"]
            self.prediction_hpre[layer_index] = states[0, -1, :]
        return hook

    def _remember_mid(self, layer_index):
        def hook(_module, args):
            self.after_attention[layer_index] = args[0][0, -1, :]
        return hook

    @contextmanager
    def capture(self, *, prefill: bool, variant: Intervention, need_prefill: bool = False,
                need_sar_ems: bool = False, need_patch_mixture: bool = False,
                need_write_prior: bool = False, need_sar_ems_selection: bool = False,
                need_transfer_evidence: bool = False, need_sar_ems_allocation: bool = False):
        """Intercept eager attention after key/value projection and KV update."""
        if not prefill and variant.attention in {"sparc", "sparc_write"}:
            with sparc_attention(self.start, self.end, projected_write=variant.attention == "sparc_write"):
                yield
            return
        if not prefill and variant.attention in {"spin", "spin_projected"}:
            with self.spin.capture(self.start, self.end, projected_write=variant.attention == "spin_projected"):
                yield
            return
        if not prefill and variant.attention in {"svar_spin", "svar_spin_projected"}:
            with (
                self.capture(prefill=False, variant=VARIANTS["svar"]),
                self.spin.capture(self.start, self.end,
                                  projected_write=variant.attention == "svar_spin_projected",
                                  layers=range(ATTENTION_LAYERS[-1] + 1, len(self.decoder.layers))),
            ):
                yield
            return
        original = llama.eager_attention_forward
        uses_sar_ems_attention = variant.sar_ems_attention_strength > 0.0
        uses_patch_mixture = variant.sar_ems_patch_strength > 0.0
        uses_allocation = variant.sar_ems_allocation > 0.0
        uses_pade = not prefill and variant.attention == "pade"
        uses_air = not prefill and variant.attention in {"air", "air_projected", "air_agreement"}
        local_vhr = variant.visual_head_gain in {"vhr", "vhr_write"}
        sustained = variant.repeat_attention == "early_write"
        repeat_layers = ATTENTION_LAYERS if sustained else REPEAT_ATTENTION_LAYERS
        if not (need_prefill or variant.rerank or variant.ffn_mode
                or variant.attention != "native" or variant.uses_sar_ems or variant.write_prior_strength
                or variant.template_transfer or variant.visual_head_gain or need_transfer_evidence):
            yield
            return

        def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            layer = int(module.layer_idx)
            last_layer = len(self.decoder.layers) - 1
            guide = (not prefill and variant.attention in {"svar", "sadi", "pai"}
                     and layer in ATTENTION_LAYERS)
            pade_edit = uses_pade and layer == last_layer
            air_edit = uses_air and layer in AIR_LAYERS
            redistribute = (not prefill and uses_sar_ems_attention
                            and layer in SAR_EMS_ATTENTION_LAYERS
                            and self.sar_ems_budget is not None)
            mix = (not prefill and uses_patch_mixture and layer in PATCH_MIXTURE_LAYERS
                   and self.patch_mixture is not None)
            use_write_prior = not prefill and variant.write_prior_strength > 0.0 and layer in ATTENTION_LAYERS
            transfer = not prefill and variant.template_transfer is not None and layer in ATTENTION_LAYERS
            head_gain = (not prefill and variant.visual_head_gain is not None
                         and layer in (VHR_LAYERS if local_vhr else ATTENTION_LAYERS))
            repeat_transfer = (not prefill and variant.repeat_attention is not None
                               and self.repeat_donor_offsets is not None and layer in repeat_layers)
            observe = layer in EVIDENCE_LAYERS and (
                (prefill and need_prefill) or (variant.rerank and variant.repeat_mode is None)
                or ((variant.ffn_mode == "rotate" or uses_sar_ems_attention)
                    and layer == EARLY_WRITE_LAYER)
                or (uses_patch_mixture and layer in PATCH_MIXTURE_ANCHORS)
                or (uses_allocation and layer == ALLOCATION_ANCHOR))
            save_prior_values = prefill and (need_write_prior or need_transfer_evidence) and layer in WRITE_PRIOR_ANCHORS
            active = (guide or pade_edit or air_edit or redistribute or mix
                      or use_write_prior or transfer or head_gain or repeat_transfer)
            if not active and (not observe or prefill):
                if observe or save_prior_values:
                    self.visual_values[layer] = llama.repeat_kv(
                        value[:, :, self.start:self.end, :], module.num_key_value_groups
                    )[0].contiguous()
                output, attention = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
                if uses_pade:
                    self._accumulate_pad(layer, attention[0, :, -1, self.start:self.end])
                return output, attention

            key = llama.repeat_kv(key, module.num_key_value_groups)
            value = llama.repeat_kv(value, module.num_key_value_groups)
            scores = (query @ key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, :, :key.shape[-2]]
            visual = scores[:, :, -1, self.start:self.end]
            if guide:
                if variant.attention == "svar":
                    visual.copy_(visual + 0.5 * visual.abs().mean(dim=1, keepdim=True))
                elif variant.attention == "pai":
                    visual.copy_(visual + 0.5 * visual.abs())
                else:
                    guided = sadi_visual_logits(visual)
                    if variant.attention_strength != 1.0:
                        guided = torch.lerp(visual, guided, variant.attention_strength)
                    if variant.sadi_head_rank is not None:
                        mask = self._sadi_head_mask(layer, scores, value,
                                                    variant.sadi_head_rank)
                        guided = _matched_sadi_head_logits(visual, guided, mask)
                    visual.copy_(guided)
            if pade_edit:
                native = scores[:, :, -1, :].softmax(dim=-1, dtype=torch.float32)
                self._accumulate_pad(layer, native[0, :, self.start:self.end])
                pad = self.pad_accumulator
                pad = pad / pad.amax().clamp_min(1e-12)
                median = visual.median(dim=-1, keepdim=True).values
                mad = (visual - median).abs().median(dim=-1, keepdim=True).values
                boost = 0.1 * mad * pad
                visual.add_(boost)
                scores[:, :, -1, :self.start].sub_(boost.mean(dim=-1, keepdim=True))
            if redistribute:
                visual.copy_(rebalance_visual_logits(
                    visual, self.sar_ems_budget[None, :],
                    permute=variant.permute_attention_guidance,
                ))
            if air_edit:
                gram = None
                weighted_lens = variant.attention == "air_projected" and layer in AIR_LENS_LAYERS
                agreement = variant.attention == "air_agreement" and layer in AIR_TEXT_LAYERS
                if weighted_lens or agreement:
                    gram = self.air_grams.get(layer)
                    if gram is None:
                        gram = air_projection_gram(module.o_proj.weight, scores.shape[1])
                        self.air_grams[layer] = gram
                attention = air_decode_attention(scores, value, self.start, self.end, layer,
                                                 projected_gram=gram if weighted_lens else None,
                                                 agreement_gram=gram if agreement else None)
                return (attention @ value).transpose(1, 2).contiguous(), attention
            attention = F.softmax(scores, dim=-1, dtype=torch.float32)
            if uses_allocation and (pade_edit or (guide and layer > ALLOCATION_ANCHOR)):
                if self.sar_ems_recipient is not None:
                    recipient, fraction = self.sar_ems_recipient
                    regions = attention[:, :, -1, self.start:self.end]
                    previous = regions.clone()
                    if variant.permute_attention_guidance:
                        recipient = recipient.roll(recipient.numel() // 3)
                    regions.lerp_(regions.sum(dim=-1, keepdim=True) * recipient, fraction)
                    self.transfer_mass.append(0.5 * (regions - previous).abs().sum(dim=-1).mean())
            if repeat_transfer:
                repeated_positions = self.repeat_donor_offsets + scores.shape[-1]
                row = attention[:, :, -1, :]
                evidence = self.repeat_attention_evidence
                if sustained:
                    if layer in WRITE_PRIOR_ANCHORS:
                        self._update_template_evidence(layer, row[0, :, self.start:self.end])
                    evidence = self.transfer_evidence
                    row, template_amount = transfer_template_attention(
                        row, self.start, self.end, evidence, template_fraction=0.25, visual_limit=3.0,
                    )
                shifted, amount = transfer_repeat_attention(
                    row, self.start, self.end, repeated_positions, evidence,
                )
                if sustained:
                    amount = amount + template_amount
                attention[:, :, -1, :] = shifted
                self.transfer_mass.append(amount.mean())
            if transfer:
                if variant.transfer_uses_writes and layer in WRITE_PRIOR_ANCHORS:
                    self._update_template_evidence(layer, attention[0, :, -1, self.start:self.end])
                shifted, amount = transfer_template_attention(
                    attention[:, :, -1, :], self.start, self.end, self.transfer_evidence,
                )
                attention[:, :, -1, :] = shifted
                self.transfer_mass.append(amount.mean())
            if use_write_prior:
                visual_attention = attention[:, :, -1, self.start:self.end]
                if layer in WRITE_PRIOR_ANCHORS:
                    self._update_write_prior(layer, visual_attention[0])
                if variant.write_prior_logit_scale:
                    visual.copy_(guide_visual_logits_with_writes(
                        visual, visual_attention, self.visual_prior, variant.write_prior_strength,
                        cap=variant.write_prior_logit_cap,
                    ))
                    attention = F.softmax(scores, dim=-1, dtype=torch.float32)
                else:
                    attention[:, :, -1, :] = mix_visual_write_prior(
                        attention[:, :, -1, :], self.start, self.end, self.visual_prior,
                        variant.write_prior_strength,
                    )
            if mix:
                target, budget = self.patch_mixture
                attention[:, :, -1, :] = mix_supported_visual_attention(
                    attention[:, :, -1, :], self.start, self.end, target, budget,
                )
            attention = attention.to(query.dtype)
            if uses_pade and not pade_edit:
                self._accumulate_pad(layer, attention[0, :, -1, self.start:self.end])
            if observe:
                self.attention_rows[layer] = attention[0, :, -1, self.start:self.end]
            if (not prefill and variant.repeat_attention == "write" and layer == EARLY_WRITE_LAYER
                    and self.repeat_donor_offsets is not None):
                self._prepare_repeat_evidence(layer)
            output = attention @ value
            if head_gain:
                visual_attention = attention[:, :, -1:, self.start:self.end]
                if local_vhr:
                    mask = self._vhr_mask(layer, scores, value, output,
                                          variant.visual_head_gain == "vhr_write")
                    if variant.permute_attention_guidance:
                        mask = mask.roll(mask.numel() // 3)
                    output[:, :, -1:, :] *= 1 + mask[None, :, None, None]
                else:
                    visual_output = visual_attention @ value[:, :, self.start:self.end]
                    mask = self._visual_write_mask(layer, visual_attention, visual_output,
                                                   output[:, :, -1:, :], variant.visual_head_gain)
                    output[:, :, -1:, :] += (VISUAL_HEAD_GAIN - 1) * mask[None, :, None, None] * visual_output
            return output.transpose(1, 2).contiguous(), attention

        with ExitStack() as stack:
            stack.enter_context(patch.object(llama, "eager_attention_forward", forward))
            if (prefill and need_transfer_evidence) or (not prefill and variant.transfer_uses_writes):
                for layer_index in WRITE_PRIOR_ANCHORS:
                    layer = self.decoder.layers[layer_index]
                    capture_states = (self._remember_states(layer_index, False) if prefill
                                      else self._remember_prediction(layer_index))
                    hook = layer.register_forward_pre_hook(capture_states, with_kwargs=True)
                    stack.callback(hook.remove)
                    if not prefill and variant.template_transfer == "write_ffn":
                        hook = layer.mlp.register_forward_hook(self._tilt_template_evidence(layer_index))
                        stack.callback(hook.remove)
            for layer_index in CAPTURE_LAYERS:
                layer = self.decoder.layers[layer_index]
                if prefill and need_prefill:
                    save_sar_hpre = (
                        (need_sar_ems and layer_index == EARLY_WRITE_LAYER)
                        or (need_sar_ems_allocation and layer_index == ALLOCATION_ANCHOR)
                        or (need_patch_mixture and layer_index in PATCH_MIXTURE_ANCHORS)
                        or (need_sar_ems_selection and layer_index in EVIDENCE_LAYERS)
                    )
                    hook = layer.register_forward_pre_hook(
                        self._remember_states(layer_index, save_sar_hpre), with_kwargs=True
                    )
                    stack.callback(hook.remove)
                elif variant.compares_sar_ems and layer_index in EVIDENCE_LAYERS:
                    hook = layer.register_forward_pre_hook(
                        self._remember_prediction(layer_index), with_kwargs=True
                    )
                    stack.callback(hook.remove)
                    hook = layer.mlp.register_forward_hook(self._remember_ffn(layer_index))
                    stack.callback(hook.remove)
                elif variant.drift_strength and layer_index in EVIDENCE_LAYERS:
                    hook = layer.mlp.register_forward_hook(self._remember_ffn(layer_index))
                    stack.callback(hook.remove)
                    hook = layer.post_attention_layernorm.register_forward_pre_hook(
                        self._remember_mid(layer_index)
                    )
                    stack.callback(hook.remove)
                elif variant.repeat_mode == "visual" and layer_index == EARLY_WRITE_LAYER:
                    hook = layer.mlp.register_forward_hook(self._remember_ffn(layer_index))
                    stack.callback(hook.remove)
                elif variant.repeat_attention == "write" and layer_index == EARLY_WRITE_LAYER:
                    hook = layer.register_forward_pre_hook(
                        self._remember_prediction(layer_index), with_kwargs=True
                    )
                    stack.callback(hook.remove)
            if not prefill and variant.ffn_mode:
                layer = self.decoder.layers[FFN_EDIT_LAYER]
                if variant.ffn_mode == "rotate":
                    hook = layer.post_attention_layernorm.register_forward_pre_hook(
                        self._remember_mid(FFN_EDIT_LAYER)
                    )
                    stack.callback(hook.remove)
                hook = layer.mlp.register_forward_hook(self._edit_ffn(variant))
                stack.callback(hook.remove)
            if not prefill and uses_sar_ems_attention:
                layer = self.decoder.layers[FFN_EDIT_LAYER]
                hook = layer.post_attention_layernorm.register_forward_pre_hook(
                    self._remember_mid(FFN_EDIT_LAYER)
                )
                stack.callback(hook.remove)
                hook = layer.mlp.register_forward_hook(self._plan_sar_ems_attention(variant))
                stack.callback(hook.remove)
            if not prefill and uses_patch_mixture:
                for layer_index in PATCH_MIXTURE_ANCHORS:
                    layer = self.decoder.layers[layer_index]
                    hook = layer.post_attention_layernorm.register_forward_pre_hook(
                        self._remember_mid(layer_index)
                    )
                    stack.callback(hook.remove)
                    hook = layer.mlp.register_forward_hook(
                        self._plan_patch_mixture(layer_index, variant.sar_ems_patch_strength)
                    )
                    stack.callback(hook.remove)
            if not prefill and uses_allocation:
                layer = self.decoder.layers[ALLOCATION_ANCHOR]
                hook = layer.post_attention_layernorm.register_forward_pre_hook(
                    self._remember_mid(ALLOCATION_ANCHOR)
                )
                stack.callback(hook.remove)
                hook = layer.mlp.register_forward_hook(self._plan_sar_ems_allocation(variant))
                stack.callback(hook.remove)
            yield

    def _accumulate_pad(self, layer, attention):
        current = attention.float().mean(dim=0)
        if layer == 0:
            self.pad_accumulator = torch.zeros_like(current)
        else:
            self.pad_accumulator.add_((current - self.pad_previous).clamp_min(0))
        self.pad_previous = current

    def _sadi_head_mask(self, layer, scores, value, ranking):
        mask = self.visual_head_masks.get(layer)
        if mask is None:
            last_scores = scores[:, :, -1:, :]
            native_attention = F.softmax(last_scores, dim=-1, dtype=torch.float32).to(value.dtype)
            mask = self._vhr_mask(layer, last_scores, value, native_attention @ value,
                                  project=ranking == "projected")
        return mask

    def sadi_head_selection(self) -> dict[int, list[int]]:
        """Return each layer's selected heads for the current image."""
        return {
            layer: mask.nonzero().flatten().tolist()
            for layer, mask in self.visual_head_masks.items()
        }

    def _vhr_mask(self, layer, scores, value, output, project):
        mask = self.visual_head_masks.get(layer)
        if mask is None:
            text_scores = scores[:, :, -1:, :].clone()
            text_scores[..., self.start:self.end] = -torch.inf
            text_attention = F.softmax(text_scores, dim=-1, dtype=torch.float32).to(value.dtype)
            text = (text_attention @ value)[0, :, 0, :]
            delta = output[0, :, -1, :] - text
            divergence = delta.float().square().sum(dim=-1)
            text_energy = text.float().square().sum(dim=-1)
            outliers = ((divergence > divergence.mean() + divergence.std(correction=0))
                        & (text_energy > text_energy.mean() + text_energy.std(correction=0)))
            score = divergence
            if project:
                weight = self.decoder.layers[layer].self_attn.o_proj.weight
                projection = weight.reshape(weight.shape[0], *delta.shape).permute(1, 2, 0)
                score = torch.bmm(delta[:, None, :], projection).float().square().sum(dim=(1, 2))
            selected = score.masked_fill(outliers, -torch.inf).topk(score.numel() // 2).indices
            mask = torch.zeros_like(score, dtype=output.dtype)
            mask[selected] = 1
            self.visual_head_masks[layer] = mask
        return mask

    def _plan_sar_ems_allocation(self, variant):
        def hook(_module, _args, output):
            self.sar_ems_recipient = None
            step = self.sar_ems_step
            self.sar_ems_step += 1
            mid = self.after_attention[ALLOCATION_ANCHOR]
            ffn = output[0, -1, :]
            post = self.norm(mid + ffn)
            logits, tokens = self.head(post).float().topk(PATCH_MIXTURE_CANDIDATES)
            gains = self.head.weight[tokens].float() @ (post.float() - self.norm(mid).float())
            weights = logits.softmax(dim=0) * (gains.clamp_min(0) / 2).tanh() * self.noun_mask[tokens]
            confidence = weights.sum()
            if confidence.item() == 0.0:
                return
            support = self._get_sar_patch_terms(ALLOCATION_ANCHOR).candidate_mixture(
                self.attention_rows[ALLOCATION_ANCHOR], tokens, weights / confidence,
            )
            states = self.visual_states[ALLOCATION_ANCHOR]
            preference = (states @ F.normalize(ffn.float(), dim=-1) / TAU).softmax(dim=0)
            deficit = (support - preference).clamp_min(0)
            drift = (1 - (support * preference).sqrt().sum()).clamp(0, 1)
            fraction = variant.sar_ems_allocation * drift * confidence
            mass = deficit.sum()
            if mass.item() > 0.0:
                self.sar_ems_recipient = (deficit / mass, fraction)
                self.sar_ems_targets.append((step, int(tokens[weights.argmax()].item())))
        return hook

    def _write_gram(self, layer_index):
        gram = self.write_prior_grams.get(layer_index)
        if gram is None:
            projection = self.decoder.layers[layer_index].self_attn.o_proj.weight
            gram = visual_write_gram(self.visual_values[layer_index], projection)
            self.write_prior_grams[layer_index] = gram
        return gram

    def _update_write_prior(self, layer_index, visual_attention):
        magnitudes = visual_write_norms(self._write_gram(layer_index), visual_attention).sqrt()
        self.visual_prior = magnitudes / magnitudes.sum().clamp_min(1e-12)

    def _update_template_evidence(self, layer_index, attention):
        if layer_index == WRITE_PRIOR_ANCHORS[0]:
            self.transfer_previous = None
        writes = visual_write_norms(self._write_gram(layer_index), attention)
        query = F.normalize(self.prediction_hpre[layer_index].float(), dim=-1)
        similarity = self.visual_states[layer_index] @ query
        self.transfer_previous, self.transfer_evidence = write_evidence_distribution(
            writes, similarity, self.transfer_previous,
        )

    def _tilt_template_evidence(self, layer_index):
        def hook(_module, _args, output):
            ffn = F.normalize(output[0, -1, :].float(), dim=-1)
            preference = (self.visual_states[layer_index] @ ffn / TAU).softmax(dim=0)
            evidence = self.transfer_evidence
            updated = evidence + 0.5 * (evidence - preference).clamp_min(0)
            self.transfer_evidence = updated / updated.sum().clamp_min(1e-12)
        return hook

    def attention_transfer_statistics(self) -> dict:
        if not self.transfer_mass:
            return {}
        return {"layer_steps": len(self.transfer_mass),
                "mean_moved_mass": float(torch.stack(self.transfer_mass).mean())}

    def _visual_write_mask(self, layer_index, visual_attention, visual_output, output, mode):
        mask = self.visual_head_masks.get(layer_index)
        if mask is None:
            if mode == "write":
                visual = visual_output[0, :, 0, :]
                text = output[0, :, 0, :] - visual
                weight = self.decoder.layers[layer_index].self_attn.o_proj.weight
                scores = projected_visual_head_scores(visual, text, weight)
            else:
                scores = visual_attention[0, :, 0, :].float().sum(dim=-1)
            selected = scores.topk(scores.numel() // 2).indices
            mask = torch.zeros_like(scores, dtype=visual_output.dtype)
            mask[selected] = 1
            self.visual_head_masks[layer_index] = mask
        return mask

    def prepare_repeat_attention(self, response: list[int], *, sustained: bool = False) -> None:
        self.repeat_donor_offsets = None
        self.repeat_attention_evidence = None
        starts = _repeat_attention_starts(response, self.nouns, sustained)
        if starts:
            positions = sorted({j - len(response) for i in starts for j in range(i, i + REPEAT_ATTENTION_SPAN)})
            self.repeat_donor_offsets = torch.tensor(positions, device=self.head.weight.device)

    def _prepare_repeat_evidence(self, layer_index):
        terms = self.sar_patch_terms.get(layer_index)
        gram = terms.gram if terms is not None else self._write_gram(layer_index)
        writes = visual_write_norms(gram, self.attention_rows[layer_index])
        query = F.normalize(self.prediction_hpre[layer_index].float(), dim=-1)
        _, self.repeat_attention_evidence = write_evidence_distribution(
            writes, self.visual_states[layer_index] @ query, previous=None,
        )

    def _get_sar_patch_terms(self, layer_index: int) -> SarPatchTerms:
        terms = self.sar_patch_terms.get(layer_index)
        if terms is None:
            layer = self.decoder.layers[layer_index]
            terms = SarPatchTerms(
                self.visual_values[layer_index], layer.self_attn.o_proj.weight,
                self.sar_visual_hpre[layer_index], self.norm, self.head,
                gram=self.write_prior_grams.get(layer_index),
            )
            self.sar_patch_terms[layer_index] = terms
        return terms

    def _plan_sar_ems_attention(self, variant: Intervention):
        def hook(_module, _args, output):
            self.sar_ems_budget = None
            step = self.sar_ems_step
            self.sar_ems_step += 1
            hmid = self.after_attention[FFN_EDIT_LAYER]
            ffn = output[0, -1, :]
            post = self.norm(hmid + ffn)
            token = int(self.head(post).argmax().item())
            if token not in self.nouns:
                return
            gain = self.head.weight[token].float() @ (post.float() - self.norm(hmid).float())
            if gain.item() <= 0.0:
                return
            target = self._get_sar_patch_terms(EARLY_WRITE_LAYER).distribution(
                self.attention_rows[EARLY_WRITE_LAYER], token,
            )
            states = self.visual_states[FFN_EDIT_LAYER]
            preference = (states @ F.normalize(ffn.float(), dim=-1) / TAU).softmax(dim=0)
            drift = (1 - (target * preference).sqrt().sum()).clamp(0, 1)
            gap = (target - preference).clamp_min(0)
            gate = drift * (gain / 2.0).tanh()
            self.sar_ems_budget = variant.sar_ems_attention_strength * gate * gap / gap.amax().clamp_min(1e-6)
            self.sar_ems_targets.append((step, token))
        return hook

    def _plan_patch_mixture(self, layer_index: int, strength: float):
        def hook(_module, _args, output):
            self.patch_mixture = None
            if layer_index == PATCH_MIXTURE_ANCHORS[0]:
                self.sar_ems_step += 1
            mid = self.after_attention[layer_index]
            ffn = output[0, -1, :]
            post = self.norm(mid + ffn)
            logits, tokens = self.head(post).float().topk(PATCH_MIXTURE_CANDIDATES)
            gain = self.head.weight[tokens].float() @ (post.float() - self.norm(mid).float())
            weights = logits.softmax(dim=0) * (gain.clamp_min(0.0) / 2).tanh() * self.noun_mask[tokens]
            confidence = weights.sum()
            if confidence.item() == 0.0:
                return
            target = self._get_sar_patch_terms(layer_index).candidate_mixture(
                self.attention_rows[layer_index], tokens, weights / confidence,
            )
            states = self.visual_states[layer_index]
            preference = (states @ F.normalize(ffn.float(), dim=-1) / TAU).softmax(dim=0)
            budget = strength * confidence * (target - preference).clamp_min(0)
            self.patch_mixture = (target, budget)
            self.sar_ems_targets.append((self.sar_ems_step - 1, int(tokens[weights.argmax()].item())))
        return hook

    def sar_ems_statistics(self, response: list[int]) -> dict:
        return {
            "guided_steps": len(self.sar_ems_targets),
            "target_matches_generated": sum(response[step] == token for step, token in self.sar_ems_targets),
        }

    def _edit_ffn(self, variant: Intervention):
        def hook(_module, _args, output):
            if variant.ffn_mode == "attenuate":
                return output * (1 - variant.ffn_strength)
            hmid = self.after_attention[FFN_EDIT_LAYER]
            ffn = output[0, -1, :]
            post = self.norm(hmid + ffn)
            token = int(self.head(post).argmax().item())
            if token not in self.nouns:
                return output

            value = self._project_per_patch(EARLY_WRITE_LAYER, [token])[:, :, 0]
            weights = self.attention_rows[EARLY_WRITE_LAYER].float()
            contribution = (weights * value).sum(dim=0)
            positive = contribution.clamp_min(0.0)
            positive_mass = positive.sum()
            target = positive / positive_mass.clamp_min(1e-6)

            states = self.visual_states[FFN_EDIT_LAYER]
            u = F.normalize(ffn.float(), dim=-1)
            preference = (states @ u / TAU).softmax(dim=0)
            drift = (1 - (target * preference).sqrt().sum()).clamp(0, 1)
            gain = self.head.weight[token].float() @ (post.float() - self.norm(hmid).float())
            gate = drift * (gain.clamp_min(0.0) / 2.0).tanh()
            gate = gate * (positive_mass > 1e-6)

            gradient = (target - preference) @ states
            tangent = gradient - u * (gradient @ u)
            delta = variant.ffn_strength * gate * F.normalize(tangent, dim=-1, eps=1e-6)
            rotated = F.normalize(u + delta, dim=-1)
            steered = (rotated * ffn.float().norm()).to(ffn.dtype)
            new_preference = (states @ F.normalize(steered.float(), dim=-1) / TAU).softmax(dim=0)
            new_drift = (1 - (target * new_preference).sqrt().sum()).clamp(0, 1)
            accept = (gate > 0.0) & (new_drift < drift) & (steered != ffn).any()
            self.ffn_steps.append(torch.stack((accept.float(), drift, torch.where(accept, new_drift, drift))))
            edited = output.clone()
            edited[0, -1] = torch.where(accept, steered, ffn)
            return edited
        return hook

    def ffn_statistics(self) -> dict:
        if not self.ffn_steps:
            return {}
        stats = torch.stack(self.ffn_steps)
        accepted = stats[:, 0].sum()
        mean_gain = ((stats[:, 1] - stats[:, 2]).sum() / accepted.clamp_min(1)).item()
        return {"draft_noun_steps": len(self.ffn_steps), "accepted_updates": int(accepted.item()),
                "mean_drift_reduction_per_update": mean_gain}

    def _project_per_patch(self, layer_index: int, ids: list[int]):
        """Cache the head readout and its product with each visual value."""
        cached = self.patch_readouts[layer_index]
        missing = [token for token in ids if token not in cached]
        if missing:
            value = self.visual_values[layer_index]
            learned = self.readouts[layer_index]
            new_tokens = [token for token in missing if token not in learned]
            if new_tokens:
                decoder = self.decoder.layers[layer_index]
                unembedding = self.head.weight[new_tokens] * self.norm.weight
                rows = unembedding @ decoder.self_attn.o_proj.weight
                for token, row in zip(new_tokens, rows.view(len(new_tokens), value.shape[0], value.shape[-1])):
                    learned[token] = row
            readouts = torch.stack([learned[token] for token in missing], dim=-1)
            projections = torch.bmm(value, readouts)
            for token, projection in zip(missing, projections.unbind(dim=-1)):
                cached[token] = projection
        return torch.stack([cached[token] for token in ids], dim=-1)

    def select(self, logits, step: int, response: list[int], variant: Intervention):
        scores = logits[0].float()
        if variant.repeat_mode is not None:
            return self._select_repeated_phrase(scores, step, response, variant.repeat_mode)
        if variant.calibrated_penalty > 0.0:
            if variant.calibrated_candidates > 2:
                return self._select_calibrated_pool(scores, step, variant)
            return self._select_calibrated(scores, step, variant.calibrated_penalty)
        top_scores, indices = scores.topk(CANDIDATE_COUNT)
        logit_window = SAR_EMS_LOGIT_WINDOW if variant.sar_ems_selection else LOGIT_WINDOW
        eligible = indices.masked_fill(top_scores < top_scores[0] - logit_window, -1)
        # Keep torch.argmax's tie policy even if topk orders a tie differently.
        winner, *alternatives = torch.cat((scores.argmax().reshape(1), eligible)).tolist()
        if winner not in self.nouns:
            return winner, None
        ids = [winner] + [token for token in alternatives
                         if token != winner and token in self.nouns and token not in response]
        if len(ids) < 2:
            return winner, None

        self.candidate_steps += 1
        if variant.sar_ems_selection:
            return self._select_sar_ems(scores, ids[:1 + SAR_EMS_ALTERNATIVES], step)
        original = scores[ids]
        support = []
        penalties = []
        for layer in EVIDENCE_LAYERS:
            projection = self._project_per_patch(layer, ids)
            weight = self.attention_rows[layer].float()
            contribution = (weight.unsqueeze(-1) * projection).sum(dim=0)
            support.append(contribution.sum(dim=0) / contribution.abs().sum(dim=0).clamp_min(1e-6))
            if variant.drift_strength:
                positive = contribution.clamp_min(0.0)
                total_positive = positive.sum(dim=0)
                target = positive / total_positive.clamp_min(1e-6)
                ffn = self.ffn_outputs[layer]
                preference = (self.visual_states[layer] @ F.normalize(ffn.float(), dim=-1) / TAU).softmax(dim=0)
                disagreement = (1 - (target * preference[:, None]).sqrt().sum(dim=0)).clamp(0, 1)
                disagreement = torch.where(total_positive > 1e-6, disagreement, torch.zeros_like(disagreement))
                hmid = self.after_attention[layer]
                # Same-layer logit lens measures which candidate this FFN promoted.
                gain = (self.head.weight[ids].float() @
                        (self.norm(hmid + ffn).float() - self.norm(hmid).float()))
                penalties.append(disagreement * (gain.clamp_min(0.0) / 2.0).tanh())

        support = torch.stack(support).mean(dim=0)
        drift = torch.stack(penalties).mean(dim=0) if penalties else torch.zeros_like(support)
        shifted = original + variant.support_strength * support - variant.drift_strength * drift
        selected = int(shifted.argmax().item())
        if selected == 0:
            return winner, None
        choice = ids[selected]
        return choice, {
            "step": step, "old_token": winner, "new_token": choice,
            "original_gap": float(original[0] - original[selected]),
            "old_support": float(support[0]), "new_support": float(support[selected]),
            "old_ffn_penalty": float(drift[0]), "new_ffn_penalty": float(drift[selected]),
        }

    def _candidate_sar(self, tokens: torch.Tensor):
        evidence = []
        support = []
        sar = []
        for layer in EVIDENCE_LAYERS:
            terms = self._get_sar_patch_terms(layer)
            semantics = terms.log_semantics(tokens)
            writes = visual_write_norms(terms.gram, self.attention_rows[layer])
            semantic_support = writes @ semantics
            write_mass = writes.sum()
            support.append(semantic_support / write_mass.clamp_min(1e-12))
            sar.append(semantic_support + write_mass)
            evidence.append(semantics)
        return torch.stack(support), torch.stack(sar), evidence

    def _select_sar_ems(self, scores, ids: list[int], step: int):
        tokens = torch.tensor(ids, device=scores.device)
        support, _, evidence = self._candidate_sar(tokens)
        support = support.mean(dim=0).tolist()
        comparable = [0] + [col for col, value in enumerate(support[1:], 1)
                             if value > support[0] + SAR_SUPPORT_GAIN]
        if len(comparable) == 1:
            return ids[0], None

        risk_layers = self._candidate_risks(evidence, comparable)
        risks = [sum(layers) / len(layers) for layers in risk_layers]
        for column, risk in zip(comparable[1:], risks[1:], strict=True):
            if risk < risks[0] - EMS_RISK_DROP:
                return ids[column], {
                    "step": step, "old_token": ids[0], "new_token": ids[column],
                    "original_gap": float(scores[ids[0]] - scores[ids[column]]),
                    "old_support": support[0], "new_support": support[column],
                    "old_ems_risk": risks[0], "new_ems_risk": risk,
                }
        return ids[0], None

    def _candidate_risks(self, evidence, columns):
        self.ems_comparison_steps += 1
        problems = []
        for layer, semantics in zip(EVIDENCE_LAYERS, evidence, strict=True):
            states = self.sar_visual_hpre[layer]
            cost = self.sar_ems_visual_costs.get(layer)
            if cost is None:
                cost = self.sar_ems_visual_costs[layer] = _cosine_distance_matrix(states)
            problems.append(candidate_ems_problems(
                visual_hpre=states, prediction_hpre=self.prediction_hpre[layer],
                ffn=self.ffn_outputs[layer], attention=self.attention_rows[layer],
                log_semantics=semantics[:, columns], visual_cost=cost,
            ))
        return layer_candidate_risks(problems, len(columns))

    def _calibrated_scores(self, ids):
        tokens = torch.tensor(ids, device=self.head.weight.device)
        support, sar, evidence = self._candidate_sar(tokens)
        risk = self._candidate_risks(evidence, list(range(len(ids))))
        calibration = self.calibration
        semantic = support if calibration["evidence"] == "relative_SAR" else sar
        scores = []
        for evidence_values, risk_values in zip(semantic.T.tolist(), risk, strict=True):
            values = evidence_values + risk_values
            margin = calibration["intercept"] + sum(
                weight * value for weight, value in zip(calibration["weights"], values, strict=True)
            )
            scores.append(0.5 + 0.5 * math.tanh(margin / 2))
        return scores

    def _select_calibrated_pool(self, scores, step, variant):
        winner = int(scores.argmax().item())
        if winner not in self.calibrated_tokens:
            return winner, None
        n = variant.calibrated_candidates
        top_scores, top_ids = scores.topk(n)
        gap = float(top_scores[0] - top_scores[1])
        if gap >= variant.calibrated_penalty:
            return winner, None
        self.candidate_steps += 1
        threshold = self.calibration["threshold"]
        old_risk = self._calibrated_scores([winner])[0]
        penalty = variant.calibrated_penalty * max(0.0, old_risk - threshold) / (1 - threshold)
        if penalty <= gap:
            return winner, None

        best_score = float(scores[winner]) - penalty
        alternatives = [token for score, token in zip(top_scores.tolist(), top_ids.tolist(), strict=True)
                        if token != winner and score > best_score]
        scored = [token for token in alternatives if token in self.calibrated_tokens]
        risks = dict(zip(scored, self._calibrated_scores(scored), strict=True)) if scored else {}
        chosen = winner
        for token, original in zip(top_ids.tolist(), top_scores.tolist(), strict=True):
            if token not in alternatives:
                continue
            risk = risks.get(token, 0.0)
            adjusted = original - variant.calibrated_penalty * max(0.0, risk - threshold) / (1 - threshold)
            if adjusted > best_score:
                chosen, best_score = token, adjusted
        if chosen == winner:
            return winner, None
        return chosen, {
            "step": step, "old_token": winner, "new_token": chosen,
            "original_gap": float(scores[winner] - scores[chosen]),
            "old_hall_score": old_risk, "new_hall_score": risks.get(chosen),
            "old_penalty": penalty,
        }

    def _select_calibrated(self, scores, step: int, max_penalty: float):
        winner = int(scores.argmax().item())
        if winner not in self.calibrated_tokens:
            return winner, None
        alternatives = scores.clone()
        alternatives[winner] = -torch.inf
        second = int(alternatives.argmax().item())
        gap = float(scores[winner] - scores[second])
        if gap >= max_penalty:
            return winner, None
        self.candidate_steps += 1
        ids = [winner, second] if second in self.calibrated_tokens else [winner]
        threshold = self.calibration["threshold"]
        hall_scores = self._calibrated_scores(ids)
        if len(hall_scores) == 1:
            hall_scores.append(0.0)
        penalties = [max_penalty * max(0.0, score - threshold) / (1 - threshold) for score in hall_scores]
        if penalties[0] - penalties[1] <= gap:
            return winner, None
        return second, {
            "step": step, "old_token": winner, "new_token": second, "original_gap": gap,
            "old_hall_score": hall_scores[0], "new_hall_score": hall_scores[1] if len(ids) > 1 else None,
            "old_penalty": penalties[0], "new_penalty": penalties[1],
        }

    def _select_repeated_phrase(self, scores, step: int, response: list[int], mode: str):
        winner = int(scores.argmax().item())
        if winner not in self.nouns or len(response) < REPEAT_PHRASE_LENGTH - 1:
            return winner, None

        prefix = tuple(response[-(REPEAT_PHRASE_LENGTH - 1):])
        phrase = (*prefix, winner)
        current = None
        penalty = 0.0
        overlap = None
        drift = None
        if mode == "visual":
            layer = EARLY_WRITE_LAYER
            terms = self._get_sar_patch_terms(layer)
            attention = self.attention_rows[layer]
            current = terms.distribution(attention, winner)
            if phrase in self.repeat_evidence:
                self.candidate_steps += 1
                previous = self.repeat_evidence[phrase]
                overlap = float((current * previous).sqrt().sum().clamp(0, 1).item())
                if overlap >= REPEAT_VISUAL_OVERLAP:
                    states = self.visual_states[layer]
                    ffn = F.normalize(self.ffn_outputs[layer].float(), dim=-1)
                    preference = ((states @ ffn) / TAU).softmax(dim=0)
                    drift = float((1 - (preference * current).sqrt().sum()).clamp(0, 1).item())
                    penalty = REPEAT_LOGIT_PENALTY * (0.5 + 0.5 * drift)
        elif phrase in self.repeat_evidence:
            self.candidate_steps += 1
            penalty = REPEAT_LOGIT_PENALTY

        chosen = winner
        if penalty:
            adjusted = scores.clone()
            adjusted[winner] -= penalty
            chosen = int(adjusted.argmax().item())

        if chosen in self.nouns:
            if mode == "visual" and chosen != winner:
                current = terms.distribution(attention, chosen)
            self.repeat_evidence[(*prefix, chosen)] = current
        if chosen == winner:
            return chosen, None
        return chosen, {
            "step": step, "old_token": winner, "new_token": chosen,
            "original_gap": float(scores[winner] - scores[chosen]),
            "repeat_penalty": penalty, "sar_overlap": overlap, "ffn_preference_drift": drift,
        }
