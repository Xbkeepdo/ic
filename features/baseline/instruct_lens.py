"""Training-free Instruction Lens Score for one aligned object token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn.functional as F

from models.dgst_capture import target_probabilities_multi


@dataclass(frozen=True)
class InstructionLensFeatures:
    score: float
    s_cafe: float
    s_local: float
    s_cls: float
    s_con: float
    mean_context_prob: float
    s_ccs: float
    instruction_layer: int
    object_layer: int
    visual_layer: int
    target_token_id: int
    instruction_token_count: int
    visual_token_count: int
    object_state_definition: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "s_cafe": float(self.s_cafe),
            "s_local": float(self.s_local),
            "s_cls": float(self.s_cls),
            "s_con": float(self.s_con),
            "mean_context_prob": float(self.mean_context_prob),
            "s_ccs": float(self.s_ccs),
            "k": 16,
            "m": 4,
            "tau": 10.0,
            "alpha": 2.0,
            "w": 0.4,
            "score_formula": "0.6*s_cls+0.4*s_ccs",
            "score_orientation": "higher_is_more_truthful",
            "instruction_layer": int(self.instruction_layer),
            "object_layer": int(self.object_layer),
            "visual_layer": int(self.visual_layer),
            "target_token_id": int(self.target_token_id),
            "instruction_token_count": int(self.instruction_token_count),
            "visual_token_count": int(self.visual_token_count),
            "object_state_definition": str(self.object_state_definition),
        }


@dataclass(frozen=True)
class InstructionLensProbabilityCache:
    target_token_ids: tuple[int, ...]
    instruction_probabilities: torch.Tensor
    visual_probabilities: torch.Tensor
    instruction_layer: int
    visual_layer: int

    def column(self, target_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            index = self.target_token_ids.index(int(target_token_id))
        except ValueError as exc:
            raise KeyError(
                f"InsLen probability cache lacks token {target_token_id}"
            ) from exc
        return (
            self.instruction_probabilities[:, index],
            self.visual_probabilities[:, index],
        )


def _layer_index(value: int, count: int, *, name: str) -> int:
    index = int(value)
    resolved = index if index >= 0 else int(count) + index
    if resolved < 0 or resolved >= int(count):
        raise ValueError(f"{name}={index} is outside {count} captured layers")
    return resolved


@torch.no_grad()
def build_instruction_lens_probability_cache(
    *,
    output_layer: Any,
    instruction_hidden_states: torch.Tensor,
    patch_hidden_states: torch.Tensor,
    target_token_ids: list[int] | tuple[int, ...],
    instruction_layer: int,
    visual_layer: int,
    tau: float = 10.0,
    chunk_size: int = 64,
) -> InstructionLensProbabilityCache:
    """Project shared image/instruction states once for every image target."""
    instruction = torch.as_tensor(instruction_hidden_states).detach()
    patches = torch.as_tensor(patch_hidden_states).detach()
    if instruction.ndim != 3 or patches.ndim != 3:
        raise ValueError("InsLen shared states must have shapes [L,M,D] and [L,P,D]")
    if instruction.shape[0] != patches.shape[0]:
        raise ValueError("InsLen shared layer counts differ")
    layer_count = int(instruction.shape[0])
    instruction_index = _layer_index(
        instruction_layer, layer_count, name="instruction_layer"
    )
    visual_index = _layer_index(visual_layer, layer_count, name="visual_layer")
    targets = tuple(dict.fromkeys(int(value) for value in target_token_ids))
    if not targets:
        raise ValueError("InsLen cache requires at least one target token")
    instruction_probs = target_probabilities_multi(
        output_layer=output_layer,
        states=instruction[instruction_index],
        target_token_ids=targets,
        chunk_size=int(chunk_size),
        temperature=float(tau),
    ).float().cpu()
    visual_probs = target_probabilities_multi(
        output_layer=output_layer,
        states=patches[visual_index],
        target_token_ids=targets,
        chunk_size=int(chunk_size),
        temperature=float(tau),
    ).float().cpu()
    return InstructionLensProbabilityCache(
        target_token_ids=targets,
        instruction_probabilities=instruction_probs,
        visual_probabilities=visual_probs,
        instruction_layer=instruction_index,
        visual_layer=visual_index,
    )


@torch.no_grad()
def compute_instruction_lens_features(
    *,
    output_layer: Any,
    instruction_hidden_states: torch.Tensor,
    token_hidden_states: torch.Tensor,
    patch_hidden_states: torch.Tensor,
    target_token_id: int,
    instruction_layer: int = -2,
    object_layer: int = -2,
    visual_layer: int = -1,
    k: int = 16,
    m: int = 4,
    tau: float = 10.0,
    alpha: float = 2.0,
    w: float = 0.4,
    chunk_size: int = 64,
    probability_cache: InstructionLensProbabilityCache | None = None,
    object_hidden_state: torch.Tensor | None = None,
) -> InstructionLensFeatures:
    """Compute the repository-weighted InsLen score for one object target."""

    instruction = torch.as_tensor(instruction_hidden_states).detach()
    token = torch.as_tensor(token_hidden_states).detach()
    patches = torch.as_tensor(patch_hidden_states).detach()
    if instruction.ndim != 3:
        raise ValueError("instruction_hidden_states must have shape [L,M,D]")
    if token.ndim != 2:
        raise ValueError("token_hidden_states must have shape [L,D]")
    if patches.ndim != 3:
        raise ValueError("patch_hidden_states must have shape [L,P,D]")
    layer_count = int(token.shape[0])
    if int(instruction.shape[0]) != layer_count or int(patches.shape[0]) != layer_count:
        raise ValueError("InsLen instruction/object/visual layer counts differ")
    if not (instruction.shape[-1] == token.shape[-1] == patches.shape[-1]):
        raise ValueError("InsLen hidden dimensions differ")
    instruction_index = _layer_index(instruction_layer, layer_count, name="instruction_layer")
    # The released InsLen LLaVA setup uses hidden_states[32] for a 32-block
    # decoder.  That entry is the final decoder-normalized state, while this
    # project's per-layer token tensor contains only the 32 raw post-block
    # states (indices 0..31).  A wrapper-supplied final-normalized state may
    # therefore use object_layer == layer_count as its faithful HF index.
    object_index = (
        int(object_layer)
        if object_hidden_state is not None
        else _layer_index(object_layer, layer_count, name="object_layer")
    )
    if object_hidden_state is not None and object_index != layer_count:
        raise ValueError(
            "InsLen final-normalized object state must use object_layer equal "
            f"to decoder layer count ({layer_count}), got {object_index}"
        )
    visual_index = _layer_index(visual_layer, layer_count, name="visual_layer")
    k_value, m_value = int(k), int(m)
    if k_value != 16 or m_value != 4:
        raise ValueError("This controlled baseline fixes InsLen at K=16 and m=4")
    if int(patches.shape[1]) < k_value:
        raise ValueError(f"InsLen needs at least {k_value} visual tokens")
    if int(instruction.shape[1]) < m_value:
        raise ValueError(f"InsLen needs at least {m_value} instruction tokens")
    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError("tau must be finite and positive")

    instruction_states = instruction[instruction_index]
    visual_states = patches[visual_index]
    if object_hidden_state is None:
        object_state = token[object_index].float()
        object_state_definition = "raw_post_block"
    else:
        object_state = torch.as_tensor(object_hidden_state).detach().float()
        if object_state.ndim != 1 or object_state.shape[0] != token.shape[-1]:
            raise ValueError("InsLen final-normalized object state must have shape [D]")
        object_state_definition = "final_decoder_norm"
    if probability_cache is None:
        instruction_probs = target_probabilities_multi(
            output_layer=output_layer,
            states=instruction_states,
            target_token_ids=[int(target_token_id)],
            chunk_size=int(chunk_size),
            temperature=float(tau),
        ).squeeze(-1).float().cpu()
        visual_probs = target_probabilities_multi(
            output_layer=output_layer,
            states=visual_states,
            target_token_ids=[int(target_token_id)],
            chunk_size=int(chunk_size),
            temperature=float(tau),
        ).squeeze(-1).float().cpu()
    else:
        if probability_cache.instruction_layer != instruction_index:
            raise ValueError("InsLen cached instruction layer differs")
        if probability_cache.visual_layer != visual_index:
            raise ValueError("InsLen cached visual layer differs")
        instruction_probs, visual_probs = probability_cache.column(
            int(target_token_id)
        )
    top_instruction_probs, top_instruction_indices = torch.topk(
        instruction_probs, k=m_value
    )
    top_visual_indices = torch.topk(visual_probs, k=k_value).indices

    instruction_cpu = instruction_states.float().cpu()
    visual_cpu = visual_states.float().cpu()
    object_cpu = object_state.cpu()
    selected_visual = visual_cpu.index_select(0, top_visual_indices)
    s_local = F.cosine_similarity(
        selected_visual,
        object_cpu.unsqueeze(0).expand_as(selected_visual),
        dim=-1,
    ).mean()
    s_cafe = instruction_probs.max()
    s_cls = s_cafe * s_local
    context = instruction_cpu.index_select(0, top_instruction_indices).mean(dim=0)
    object_norm = object_cpu.norm(p=2)
    if float(object_norm.item()) <= 0.0:
        raise ValueError("InsLen object hidden state has zero norm")
    s_con = float(alpha) - (object_cpu - context).norm(p=2) / object_norm
    mean_context_prob = top_instruction_probs.mean()
    s_ccs = s_con * mean_context_prob
    score = (1.0 - float(w)) * s_cls + float(w) * s_ccs
    values = (s_cafe, s_local, s_cls, s_con, mean_context_prob, s_ccs, score)
    if not all(math.isfinite(float(value.item())) for value in values):
        raise ValueError("InsLen produced a non-finite value")
    return InstructionLensFeatures(
        score=float(score.item()),
        s_cafe=float(s_cafe.item()),
        s_local=float(s_local.item()),
        s_cls=float(s_cls.item()),
        s_con=float(s_con.item()),
        mean_context_prob=float(mean_context_prob.item()),
        s_ccs=float(s_ccs.item()),
        instruction_layer=instruction_index,
        object_layer=object_index,
        visual_layer=visual_index,
        target_token_id=int(target_token_id),
        instruction_token_count=int(instruction.shape[1]),
        visual_token_count=int(patches.shape[1]),
        object_state_definition=object_state_definition,
    )
