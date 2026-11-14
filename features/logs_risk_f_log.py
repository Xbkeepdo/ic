"""Exact Log-S hpre transport risk plus full-visual F_log."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from features.attention_writes import visual_attention_writes
from models.dgst_capture import (
    apply_decoder_final_norm,
    attention_row_from_capture,
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
    target_probabilities_multi,
)


PROFILE = "logs_risk_f_log_hpre_v1"


def _validate_settings(tau: float, transport_top_k: int, ot_solver: str) -> None:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("Log-S source tau must be finite and positive")
    if int(transport_top_k) <= 0 or int(transport_top_k) % 2:
        raise ValueError("Log-S transport_top_k must be a positive even integer")
    if str(ot_solver).strip().lower() not in {
        "emd",
        "exact",
        "network_simplex",
        "network-simplex",
    }:
        raise ValueError("Log-S risk requires exact EMD/network-simplex")


def _transport_problem(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    visual_states: torch.Tensor,
    prediction_state: torch.Tensor,
    top_k: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, int]:
    from features.dgst_t import (
        _cosine_distance_matrix,
        _renormalize,
        _topk_union_indices,
    )

    support = _topk_union_indices(source, target, int(top_k))
    local_source = _renormalize(source.index_select(0, support))
    local_target = _renormalize(target.index_select(0, support))
    visual_cost = _cosine_distance_matrix(visual_states)
    local_cost = visual_cost.index_select(0, support).index_select(1, support)
    endpoint_cosine = F.cosine_similarity(
        prediction_state.float().unsqueeze(0),
        visual_states.float(),
        dim=-1,
    )
    endpoint_penalty = 1.0 - torch.nan_to_num(
        endpoint_cosine, nan=0.0, posinf=1.0, neginf=-1.0
    ).clamp(-1.0, 1.0)
    local_penalty = endpoint_penalty.index_select(0, support)
    shift = float(
        torch.dot(local_source.double(), local_penalty.double())
        + torch.dot(local_target.double(), local_penalty.double())
    )
    problem = (
        local_source.detach().float().cpu().numpy().copy(),
        local_target.detach().float().cpu().numpy().copy(),
        local_cost.detach().float().cpu().numpy().copy(),
    )
    return problem, shift, int(support.numel())


@torch.inference_mode()
def compute_logs_risk_f_log_batch(
    *,
    model: Any,
    captures: Sequence[dict[str, Any]],
    visual_start: int,
    visual_end: int,
    target_token_ids: Sequence[int],
    prediction_positions: Sequence[int],
    tau: float,
    transport_top_k: int,
    semantic_chunk_size: int,
    ot_solver: str,
    release_layer_captures: bool = False,
) -> list[dict[str, Any]]:
    """Return one exact ``[R_1..R_L, F_log_1..F_log_L]`` record per target."""

    _validate_settings(tau, transport_top_k, ot_solver)
    target_ids = [int(value) for value in target_token_ids]
    positions = [int(value) for value in prediction_positions]
    if len(target_ids) != len(positions):
        raise ValueError("Log-S targets and prediction positions must have equal length")
    if not target_ids:
        return []
    if int(visual_end) <= int(visual_start):
        raise ValueError("Log-S requires a non-empty visual-token span")

    layers = resolve_decoder_layers(model)
    if len(layers) != len(captures):
        raise ValueError(
            f"Decoder layer/capture mismatch: {len(layers)} != {len(captures)}"
        )
    output_layer = resolve_output_embedding_layer(model)
    if getattr(output_layer, "bias", None) is not None:
        raise ValueError(
            "The reference Log-S probability excludes LM-head bias; "
            "this model exposes a biased output head"
        )
    final_norm = resolve_decoder_final_norm(model)
    vocabulary_size = int(output_layer.weight.shape[0])
    problem_series = {str(index): [] for index in range(len(target_ids))}
    shifts = {str(index): [] for index in range(len(target_ids))}
    support_sizes = {str(index): [] for index in range(len(target_ids))}
    f_log_values = {str(index): [] for index in range(len(target_ids))}
    semantic_sum_values = {str(index): [] for index in range(len(target_ids))}
    semantic_mean_values = {str(index): [] for index in range(len(target_ids))}
    visual_write_sum_values = {str(index): [] for index in range(len(target_ids))}

    for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
        missing = [
            key
            for key in ("h_prev", "o_ffn", "attn_weights")
            if capture.get(key) is None
        ]
        if missing:
            raise KeyError(f"Log-S layer {layer_index} missing captures: {missing}")
        hpre = capture["h_prev"][0].float()
        visual_hpre = hpre[int(visual_start) : int(visual_end)]
        normalized_visual = apply_decoder_final_norm(final_norm, visual_hpre)
        probability = target_probabilities_multi(
            output_layer=output_layer,
            states=normalized_visual,
            target_token_ids=target_ids,
            chunk_size=int(semantic_chunk_size),
        )
        log_s = torch.log1p(float(vocabulary_size) * probability.float())
        attention = torch.stack(
            [
                attention_row_from_capture(capture, position)
                .float()
                .mean(dim=0)[int(visual_start) : int(visual_end)]
                for position in positions
            ],
            dim=0,
        )
        from features.dgst_t import _renormalize

        attention = torch.stack([_renormalize(row) for row in attention], dim=0)
        writes = visual_attention_writes(
            layer,
            capture,
            positions,
            int(visual_start),
            int(visual_end),
        )
        write_norm = writes.norm(dim=-1).float()

        for target_index, prediction_position in enumerate(positions):
            source_scores = F.cosine_similarity(
                capture["o_ffn"][0, prediction_position].float().unsqueeze(0),
                visual_hpre,
                dim=-1,
            )
            source = torch.softmax(
                torch.nan_to_num(
                    source_scores, nan=0.0, posinf=0.0, neginf=0.0
                )
                / float(tau),
                dim=-1,
            )
            target = _renormalize(
                attention[target_index] * log_s[:, target_index]
            )
            problem, shift, support_size = _transport_problem(
                source=source,
                target=target,
                visual_states=visual_hpre,
                prediction_state=hpre[prediction_position],
                top_k=int(transport_top_k),
            )
            key = str(target_index)
            problem_series[key].append(problem)
            shifts[key].append(shift)
            support_sizes[key].append(support_size)
            f_log_values[key].append(
                float(
                    (
                        write_norm[target_index]
                        * (1.0 + log_s[:, target_index])
                    ).sum().item()
                )
            )
            semantic_sum_values[key].append(float(probability[:, target_index].sum().item()))
            semantic_mean_values[key].append(float(probability[:, target_index].mean().item()))
            visual_write_sum_values[key].append(float(write_norm[target_index].sum().item()))

        if release_layer_captures:
            for key in ("h_prev", "o_attn", "h_mid", "o_ffn", "attn_weights"):
                capture[key] = None

    from features.dgst_t import _solve_exact_emd_problem_series

    solved = _solve_exact_emd_problem_series(problem_series)
    records: list[dict[str, Any]] = []
    for target_index in range(len(target_ids)):
        key = str(target_index)
        risk = np.asarray(
            [
                float(value) + float(shift)
                for value, shift in zip(solved[key], shifts[key])
            ],
            dtype=np.float32,
        )
        f_log = np.asarray(f_log_values[key], dtype=np.float32)
        if risk.shape != (len(layers),) or f_log.shape != (len(layers),):
            raise ValueError("Log-S/F_log output does not have one value per layer")
        if not np.isfinite(risk).all() or not np.isfinite(f_log).all():
            raise ValueError("Log-S/F_log output contains non-finite values")
        records.append(
            {
                "logs_profile": PROFILE,
                "logs_risk_per_layer": risk,
                "f_log_per_layer": f_log,
                "semantic_sum_per_layer": np.asarray(semantic_sum_values[key], dtype=np.float32),
                "semantic_mean_per_layer": np.asarray(semantic_mean_values[key], dtype=np.float32),
                "visual_write_sum_per_layer": np.asarray(visual_write_sum_values[key], dtype=np.float32),
                "logs_union_support_size_per_layer": np.asarray(
                    support_sizes[key], dtype=np.int16
                ),
                "logs_tau": float(tau),
                "logs_transport_top_k": int(transport_top_k),
                "logs_source_top_k": int(transport_top_k) // 2,
                "logs_target_top_k": int(transport_top_k) // 2,
                "logs_ot_solver": "exact_emd_network_simplex",
                "logs_state": "decoder_layer_hpre",
                "logs_semantic_probability": "final_norm_vocabulary_softmax",
                "logs_semantic_transform": "log1p(vocabulary_size * probability)",
                "logs_source_distribution": "softmax(cos(o_ffn,hpre_visual)/tau)",
                "logs_target_distribution": "renorm(mean_head_attention * log_s)",
                "logs_support": "top_k_over_2_source_union_top_k_over_2_target",
                "logs_cost": (
                    "1-cos(v_i,v_j) + 1-cos(q,v_i) + 1-cos(q,v_j)"
                ),
                "logs_f_log_definition": (
                    "sum_all_visual ||attention_write_m||_2 * (1 + log_s_m)"
                ),
                "logs_visual_token_count": int(visual_end) - int(visual_start),
            }
        )
    return records
