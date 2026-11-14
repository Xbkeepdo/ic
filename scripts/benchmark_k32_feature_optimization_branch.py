#!/usr/bin/env python3
"""Compare three extraction optimizations against the original K=32 SAR+EMS path.

Each variant repeats image loading and the teacher-forced model forward. The
variants rotate order across the same images to reduce order/cache bias.
Serialization and result comparison happen outside the timed region.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts import benchmark_k32_feature_complexity as original

from features.dgst_t import (  # noqa: E402
    _cosine_distance_matrix,
    _renormalize,
    _solve_exact_emd_problem_series,
    _topk_union_indices,
)
from features.semantic_attention_topk import target_probabilities  # noqa: E402
from features.visual_ffn_jacobian import (  # noqa: E402
    _project_values,
    reconstruct_visual_directions,
    resolve_decoder_layer_adapter,
)
from models import build_model  # noqa: E402
from models.dgst_capture import (  # noqa: E402
    apply_decoder_final_norm,
    attention_row_from_capture,
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
)
from scripts.extract_endac_811 import _capture, _targets  # noqa: E402
from scripts.extract_linear_cost_log_s_risk_811 import (  # noqa: E402
    _mean_head_visual_attention,
    _penalty,
)
from scripts.extract_log_s_hpre_k_tau_ablation_811 import (  # noqa: E402
    _problem_from_full_cost,
    _source_from_scores,
)
from utils.config_utils import (  # noqa: E402
    get_dataset_cfg,
    get_extraction_model_cfg,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/complexity_k032_50_optimization_branch"
VARIANTS = (
    "reference",
    "norm_probability_only",
    "plus_local_cost",
    "plus_visual_writes",
)


def _norm_only_probabilities(
    visual_states: torch.Tensor,
    target_ids: list[int],
    output_layer: Any,
    final_norm: Any,
    chunk_size: int,
) -> torch.Tensor:
    """Compute only the final-norm target probability, preserving full softmax."""
    weight = output_layer.weight
    targets = torch.tensor(target_ids, dtype=torch.long, device=weight.device)
    chunks = []
    for start in range(0, len(visual_states), max(1, int(chunk_size))):
        raw = visual_states[start : start + int(chunk_size)].to(
            device=weight.device, dtype=weight.dtype,
        )
        normed = apply_decoder_final_norm(final_norm, raw)
        logits = F.linear(normed, weight, bias=None)
        log_partition = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
        selected = logits.index_select(-1, targets).float()
        chunks.append(torch.exp(selected - log_partition).transpose(0, 1))
    return torch.cat(chunks, dim=1)


def _visual_only_write_norms(
    layer: Any,
    capture: dict[str, Any],
    positions: list[int],
    visual_start: int,
    visual_end: int,
) -> torch.Tensor:
    """Compute the same bias-free a_m, projecting only visual source tokens."""
    adapter = resolve_decoder_layer_adapter(layer)
    device = adapter.output_projection.weight.device
    dtype = adapter.output_projection.weight.dtype
    visual_states = capture["h_prev"][0, visual_start:visual_end].to(
        device=device, dtype=dtype,
    )
    values = _project_values(adapter, adapter.attention_norm(visual_states))
    attention = torch.stack(
        [attention_row_from_capture(capture, position).to(device=device, dtype=dtype)
         for position in positions],
        dim=0,
    )[:, :, visual_start:visual_end]
    visual_count = visual_end - visual_start
    if attention.shape != (len(positions), adapter.num_attention_heads, visual_count):
        raise ValueError(f"Unexpected visual attention shape: {attention.shape}")
    payload = (
        attention.permute(2, 0, 1).unsqueeze(-1)
        * values.permute(1, 0, 2).unsqueeze(1)
    ).reshape(visual_count, len(positions), -1)
    writes = F.linear(payload, adapter.output_projection.weight, bias=None)
    return writes.float().norm(dim=-1).transpose(0, 1)


def _local_problem(
    source: torch.Tensor,
    target: torch.Tensor,
    visual_states: torch.Tensor,
    penalty: torch.Tensor,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """Form the same exact-EMD input after selecting the K<=32 support."""
    support = _topk_union_indices(source, target, original.TOP_K)
    local_source = _renormalize(source.index_select(0, support))
    local_target = _renormalize(target.index_select(0, support))
    selected_states = visual_states.index_select(0, support)
    local_cost = _cosine_distance_matrix(selected_states)
    local_penalty = penalty.index_select(0, support)
    shift = float(
        torch.dot(local_source.double(), local_penalty.double())
        + torch.dot(local_target.double(), local_penalty.double())
    )
    return (
        local_source.detach().float().cpu().numpy().copy(),
        local_target.detach().float().cpu().numpy().copy(),
        local_cost.detach().float().cpu().numpy().copy(),
    ), shift


@torch.inference_mode()
def extract_features(
    *,
    wrapper: Any,
    model: str,
    config: dict[str, Any],
    image_id: int,
    label: dict[str, Any],
    generation: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    use_norm_only = variant != "reference"
    use_local_cost = variant in VARIANTS[2:]
    use_visual_writes = variant == "plus_visual_writes"
    response_ids = [int(value) for value in generation["response_token_ids"]]
    targets = _targets(label, response_ids)
    if not targets:
        raise ValueError(f"Image {image_id} has no object targets")
    image_path = (
        Path(get_dataset_cfg(config)["coco_root"])
        / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )
    with Image.open(image_path) as image:
        captures, positions, visual_start, visual_end, _grid, _stats = _capture(
            wrapper,
            model,
            image.convert("RGB"),
            response_ids,
            targets,
            str(config.get("run", {}).get("prompt") or "Describe this image."),
            collect_response_statistics=False,
        )
    layers = resolve_decoder_layers(wrapper.model)
    expected_layers = int(config["endac_811"]["models"][model]["num_layers"])
    if len(layers) != expected_layers or len(captures) != expected_layers:
        raise ValueError(f"{model}: unexpected decoder layer count")
    output_layer = resolve_output_embedding_layer(wrapper.model)
    final_norm = resolve_decoder_final_norm(wrapper.model)
    vocabulary_size = int(output_layer.weight.shape[0])
    target_ids = [int(target["target_token_id"]) for target in targets]
    vocab_chunk_size = int(config["semantic_attention_topk"]["vocab_chunk_size"])
    problems = {str(index): [] for index in range(len(targets))}
    shifts = {str(index): [] for index in range(len(targets))}
    sar_rows: list[list[float]] = [[] for _ in targets]

    for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
        hpre = capture["h_prev"][0].float()
        visual_hpre = hpre[visual_start:visual_end]
        if use_norm_only:
            probability = _norm_only_probabilities(
                visual_hpre, target_ids, output_layer, final_norm, vocab_chunk_size,
            )
        else:
            _unused_raw, probability = target_probabilities(
                visual_hpre, target_ids, output_layer, final_norm, vocab_chunk_size,
            )
        full_cost = None if use_local_cost else _cosine_distance_matrix(visual_hpre)
        if use_visual_writes:
            am_norm = _visual_only_write_norms(
                layer, capture, positions, visual_start, visual_end,
            )
        else:
            writes = reconstruct_visual_directions(
                layer=layer,
                capture=capture,
                prediction_positions=positions,
                visual_start=visual_start,
                visual_end=visual_end,
            )["a_tokens"].float()
            am_norm = writes.norm(dim=-1).transpose(0, 1)
        if am_norm.shape != probability.shape:
            raise ValueError(f"{model}/{image_id}: write/probability shape mismatch")
        log_support = torch.log1p(float(vocabulary_size) * probability.float())
        sar = (am_norm * (1.0 + log_support)).sum(dim=-1)

        for index, position in enumerate(positions):
            attention = _mean_head_visual_attention(
                capture, position, visual_start, visual_end,
            )
            target_dist = _renormalize(attention * log_support[index])
            update_scores = F.cosine_similarity(
                capture["o_ffn"][0, position].float().unsqueeze(0),
                visual_hpre,
                dim=-1,
            )
            source_dist = _source_from_scores(update_scores, original.TAU)
            penalty = _penalty(hpre[position], visual_hpre)
            if use_local_cost:
                problem, shift = _local_problem(
                    source_dist, target_dist, visual_hpre, penalty,
                )
            else:
                assert full_cost is not None
                problem, shift = _problem_from_full_cost(
                    source=source_dist,
                    target=target_dist,
                    full_cost=full_cost,
                    penalty=penalty,
                    top_k=original.TOP_K,
                )
            problems[str(index)].append(problem)
            shifts[str(index)].append(shift)
            sar_rows[index].append(float(sar[index].item()))

        captures[layer_index] = None
        del capture, hpre, visual_hpre, probability, full_cost, am_norm, log_support, sar

    solved = _solve_exact_emd_problem_series(problems)
    risk_values, sar_values = [], []
    for index in range(len(targets)):
        risk = np.asarray(
            [value + shift for value, shift in zip(solved[str(index)], shifts[str(index)])],
            dtype=np.float32,
        )
        sar = np.asarray(sar_rows[index], dtype=np.float32)
        if risk.shape != (expected_layers,) or sar.shape != risk.shape:
            raise ValueError(f"{model}/{image_id}: invalid feature shape")
        if not np.isfinite(risk).all() or not np.isfinite(sar).all():
            raise ValueError(f"{model}/{image_id}: nonfinite features")
        risk_values.append(risk)
        sar_values.append(sar)
    return {
        "keys": [(int(t["response_index"]), int(t["target_token_id"])) for t in targets],
        "risk": np.stack(risk_values),
        "sar": np.stack(sar_values),
    }


def run_model(model: str, device_name: str, num_images: int, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the feature benchmark")
    device = torch.device(device_name)
    image_ids, warmup_id = original.benchmark_image_ids(num_images)
    source = original._source_dir(model)
    labels = {int(key): value for key, value in original._json(source / "labeling.json").items()}
    generations = {
        int(key): value for key, value in original._json(source / "generations.json").items()
    }
    config = load_config(str(original.OURS_CONFIG))
    baseline_config = load_config(str(original.BASELINE_CONFIG))
    wrapper = build_model(
        model, get_extraction_model_cfg(baseline_config, model), device=device_name,
    )
    wrapper.model.eval().requires_grad_(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    times: dict[str, list[float]] = {name: [] for name in VARIANTS}
    per_image = []
    all_differences: dict[str, dict[str, list[np.ndarray]]] = {
        name: {feature: [] for feature in ("risk", "sar")}
        for name in VARIANTS[1:]
    }
    incremental_differences = {
        name: {feature: [] for feature in ("risk", "sar")}
        for name in VARIANTS[1:]
    }
    reference_magnitudes = {feature: [] for feature in ("risk", "sar")}
    object_count = 0
    try:
        for variant in VARIANTS:
            extract_features(
                wrapper=wrapper, model=model, config=config, image_id=warmup_id,
                label=labels[warmup_id], generation=generations[warmup_id], variant=variant,
            )
            torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        for image_offset, image_id in enumerate(image_ids):
            results = {}
            elapsed_by_variant = {}
            order = VARIANTS[image_offset % len(VARIANTS):] + VARIANTS[:image_offset % len(VARIANTS)]
            for variant in order:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = extract_features(
                    wrapper=wrapper, model=model, config=config, image_id=image_id,
                    label=labels[image_id], generation=generations[image_id], variant=variant,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                results[variant] = result
                times[variant].append(elapsed)
                elapsed_by_variant[variant] = elapsed
            reference = results["reference"]
            object_count += len(reference["keys"])
            row = {"image_id": image_id, "objects": len(reference["keys"]), "seconds": elapsed_by_variant}
            for feature in ("risk", "sar"):
                reference_magnitudes[feature].append(np.abs(reference[feature].astype(np.float64)))
            for variant_index, variant in enumerate(VARIANTS[1:], 1):
                if results[variant]["keys"] != reference["keys"]:
                    raise ValueError(f"{model}/{image_id}/{variant}: target key mismatch")
                for feature in ("risk", "sar"):
                    difference = np.abs(
                        reference[feature].astype(np.float64)
                        - results[variant][feature].astype(np.float64)
                    )
                    all_differences[variant][feature].append(difference)
                    previous = results[VARIANTS[variant_index - 1]][feature]
                    incremental_differences[variant][feature].append(
                        np.abs(previous.astype(np.float64) - results[variant][feature].astype(np.float64))
                    )
            per_image.append(row)
            print(
                f"[{model}] {image_offset + 1}/{len(image_ids)} "
                + " ".join(f"{name}={elapsed_by_variant[name]:.3f}s" for name in VARIANTS),
                flush=True,
            )
        summaries = {}
        reference_seconds = sum(times["reference"])
        for variant in VARIANTS:
            total = sum(times[variant])
            item: dict[str, Any] = {
                "total_seconds": total,
                "seconds_per_image": total / len(image_ids),
                "median_seconds_per_image": statistics.median(times[variant]),
                "milliseconds_per_object": 1000 * total / object_count,
                "speedup_vs_reference": reference_seconds / total,
            }
            if variant != "reference":
                item["numeric_difference_vs_reference"] = {}
                for feature in ("risk", "sar"):
                    values = np.concatenate([d.reshape(-1) for d in all_differences[variant][feature]])
                    previous_values = np.concatenate(
                        [d.reshape(-1) for d in incremental_differences[variant][feature]]
                    )
                    reference_values = np.concatenate(
                        [d.reshape(-1) for d in reference_magnitudes[feature]]
                    )
                    item["numeric_difference_vs_reference"][feature] = {
                        "elements": int(values.size),
                        "max_abs": float(values.max()),
                        "mean_abs": float(values.mean()),
                        "p99_abs": float(np.quantile(values, 0.99)),
                        "reference_mean_abs": float(reference_values.mean()),
                        "max_relative_to_reference": float(
                            (values / np.maximum(reference_values, 1e-6)).max()
                        ),
                        "count_gt_1e_5": int(np.count_nonzero(values > 1e-5)),
                        "count_gt_1e_4": int(np.count_nonzero(values > 1e-4)),
                        "incremental_max_abs": float(previous_values.max()),
                        "incremental_mean_abs": float(previous_values.mean()),
                    }
            summaries[variant] = item
        payload = {
            "schema": "k32-sar-ems-optimization-branch-v1",
            "model": model,
            "device": device_name,
            "image_ids": image_ids,
            "warmup_image_id": warmup_id,
            "top_k": original.TOP_K,
            "source_tau": original.TAU,
            "object_tokens": object_count,
            "timing_scope": "image loading + teacher-forced forward + postprocessing, excluding model load, generation, serialization, comparison",
            "reference_source": str(original.__file__),
            "variants": summaries,
            "per_image": per_image,
        }
        destination = output_dir / f"{model}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {destination}", flush=True)
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def summarize(output_dir: Path) -> None:
    payloads = [json.loads((output_dir / f"{model}.json").read_text()) for model in original.MODELS]
    if payloads[0]["image_ids"] != payloads[1]["image_ids"]:
        raise ValueError("Models did not use the same 50 images")
    names = {
        "reference": "Original",
        "norm_probability_only": "+ norm probability only",
        "plus_local_cost": "+ local cost after Top-K",
        "plus_visual_writes": "+ visual-only writes (all three)",
    }
    lines = [
        "# K=32 SAR+EMS optimization branch: 50-image comparison", "",
        "Same 50 COCO images per model, one additional warm-up image. Each variant repeats image loading, teacher-forced forward and postprocessing. Timed region excludes model load, generation, serialization, and numeric comparison. Variant order rotates by image. K=32 means Top-16 source union Top-16 target; source tau=0.03.", "",
        "| Model | Variant | s/image (median) | Speedup vs original | Speedup vs previous | EMS risk max / mean abs Δ vs original | SAR max / mean abs Δ vs original |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        model_name = "LLaVA-1.5-7B" if payload["model"] == "llava_1_5_7b" else "Qwen3-VL-8B"
        for variant_index, variant in enumerate(VARIANTS):
            row = payload["variants"][variant]
            if variant_index == 0:
                incremental_speedup = "—"
            else:
                previous = payload["variants"][VARIANTS[variant_index - 1]]
                incremental_speedup = f"{previous['total_seconds'] / row['total_seconds']:.2f}×"
            if variant == "reference":
                risk_text = sar_text = "—"
            else:
                differences = row["numeric_difference_vs_reference"]
                risk_text = f"{differences['risk']['max_abs']:.3g} / {differences['risk']['mean_abs']:.3g}"
                sar_text = f"{differences['sar']['max_abs']:.3g} / {differences['sar']['mean_abs']:.3g}"
            lines.append(
                f"| {model_name} | {names[variant]} | {row['seconds_per_image']:.3f} "
                f"({row['median_seconds_per_image']:.3f}) | "
                f"{row['speedup_vs_reference']:.2f}× | {incremental_speedup} | {risk_text} | {sar_text} |"
            )
    lines += [
        "", "## Incremental numerical changes", "",
        "Each row below compares with the preceding variant on identical object tokens and layers.", "",
        "| Model | Added optimization | EMS risk max / mean abs Δ | SAR max / mean abs Δ |",
        "|---|---|---:|---:|",
    ]
    for payload in payloads:
        model_name = "LLaVA-1.5-7B" if payload["model"] == "llava_1_5_7b" else "Qwen3-VL-8B"
        for variant in VARIANTS[1:]:
            differences = payload["variants"][variant]["numeric_difference_vs_reference"]
            risk = differences["risk"]
            sar = differences["sar"]
            lines.append(
                f"| {model_name} | {names[variant]} | "
                f"{risk['incremental_max_abs']:.3g} / {risk['incremental_mean_abs']:.3g} | "
                f"{sar['incremental_max_abs']:.3g} / {sar['incremental_mean_abs']:.3g} |"
            )
    lines += [
        "", "Numeric differences compare aligned object tokens and decoder layers. The JSON files retain per-image timings, p99 absolute differences, relative errors, and counts above 1e-5/1e-4. Variants are cumulative. The reference and each variant repeat the same teacher-forced model forward, so speedup reflects the full extraction path and may be modest. The probability optimization still evaluates the full vocabulary for the softmax denominator; it skips the unused raw-state probability branch. Timing differences between adjacent rows are subject to run noise. This is a 50-image feature comparison, not a retrained detector AUC comparison.", "",
    ]
    destination = output_dir / "complete_results.md"
    destination.write_text("\n".join(lines))
    print(f"Wrote {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=original.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_dir)
    elif args.model:
        torch.set_num_threads(1)
        run_model(args.model, args.device, args.num_images, args.output_dir)
    else:
        parser.error("Provide --model or --summarize")


if __name__ == "__main__":
    main()
