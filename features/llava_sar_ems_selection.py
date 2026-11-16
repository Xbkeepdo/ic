"""Exact Log-S EMS comparison for candidates of the same causal query."""

import torch
from torch.nn import functional as F

from features.dgst_t import _renormalize, _solve_exact_emd_problem_series
from features.logs_risk_f_log import _transport_problem_from_costs, _visual_endpoint_penalty


def candidate_ems_problems(
    *, visual_hpre, prediction_hpre, ffn, attention, log_semantics, visual_cost,
):
    source_similarity = F.cosine_similarity(ffn.float()[None, :], visual_hpre.float(), dim=-1)
    source = (torch.nan_to_num(source_similarity, nan=0.0, posinf=0.0, neginf=0.0) / 0.03).softmax(dim=0)
    endpoint = _visual_endpoint_penalty(prediction_hpre, visual_hpre)
    attn = _renormalize(attention.float().mean(dim=0))
    for semantics in log_semantics.T:
        problem, shift, _ = _transport_problem_from_costs(
            source=source, target=_renormalize(attn * semantics), visual_cost=visual_cost,
            endpoint_penalty=endpoint, top_k=32,
        )
        yield problem, shift


def layer_candidate_risks(layer_problems, candidate_count: int) -> list[list[float]]:
    problems = []
    shifts = []
    for layer in layer_problems:
        for problem, shift in layer:
            problems.append(problem)
            shifts.append(shift)
    costs = _solve_exact_emd_problem_series({"risk": problems})["risk"]
    risk = [cost + shift for cost, shift in zip(costs, shifts, strict=True)]
    return [risk[offset::candidate_count] for offset in range(candidate_count)]
