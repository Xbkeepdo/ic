#!/usr/bin/env python3
"""Profile the unoptimized K=32 SAR/EMS extractor and JS versus exact OT.

Uses the same 50-image selection, teacher-forced capture, raw+norm probability
readout, full visual cost matrix, full-sequence visual-write reconstruction,
and exact EMD solver as benchmark_k32_feature_complexity.py.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts import benchmark_k32_feature_complexity as original
from features.dgst_t import (
    _cosine_distance_matrix,
    _renormalize,
    _solve_exact_emd_problem_series,
    _topk_union_indices,
)
from features.semantic_attention_topk import target_probabilities
from features.visual_ffn_jacobian import reconstruct_visual_directions
from models import build_model
from models.dgst_capture import (
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
)
from scripts.extract_endac_811 import _capture, _targets
from scripts.extract_linear_cost_log_s_risk_811 import (
    _mean_head_visual_attention,
    _penalty,
)
from scripts.extract_log_s_hpre_k_tau_ablation_811 import (
    _problem_from_full_cost,
    _source_from_scores,
)
from utils.config_utils import (
    get_dataset_cfg,
    get_extraction_model_cfg,
    load_config,
)


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_k032_50_stage_profile"
STAGES = (
    "image_load", "teacher_forward", "semantic_probability", "sar_write",
    "source_target", "ot_cost", "ot_problem", "ot_solve", "js_support", "js_core",
)


def _measure(device: torch.device, times: dict[str, float], name: str, fn: Callable[[], Any]) -> Any:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize(device)
    times[name] += time.perf_counter() - start
    return value


def _js_natural(source: np.ndarray, target: np.ndarray) -> float:
    """Natural-log JS on the same normalized Top-K union used by exact EMD."""
    middle = 0.5 * (source + target)
    floor = 1e-12
    value = 0.5 * np.sum(
        source * (np.log(np.maximum(source, floor)) - np.log(np.maximum(middle, floor)))
    )
    value += 0.5 * np.sum(
        target * (np.log(np.maximum(target, floor)) - np.log(np.maximum(middle, floor)))
    )
    return float(value)


@torch.inference_mode()
def profile_image(
    *, wrapper: Any, model: str, config: dict[str, Any], image_id: int,
    label: dict[str, Any], generation: dict[str, Any], device: torch.device,
    js_top_each: int = 16,
) -> dict[str, Any]:
    times = defaultdict(float)
    response_ids = [int(v) for v in generation["response_token_ids"]]
    targets = _targets(label, response_ids)
    if not targets:
        raise ValueError(f"No object targets for {image_id}")
    image_path = (
        Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )

    def load_image() -> Image.Image:
        with Image.open(image_path) as opened:
            return opened.convert("RGB")

    image = _measure(device, times, "image_load", load_image)
    captures, positions, visual_start, visual_end, _grid, _stats = _measure(
        device, times, "teacher_forward",
        lambda: _capture(
            wrapper, model, image, response_ids, targets,
            str(config.get("run", {}).get("prompt") or "Describe this image."),
            collect_response_statistics=False,
        ),
    )
    del image
    layers = resolve_decoder_layers(wrapper.model)
    expected_layers = int(config["endac_811"]["models"][model]["num_layers"])
    if len(layers) != expected_layers or len(captures) != expected_layers:
        raise ValueError("Decoder-layer count mismatch")
    output_layer = resolve_output_embedding_layer(wrapper.model)
    final_norm = resolve_decoder_final_norm(wrapper.model)
    vocab_size = int(output_layer.weight.shape[0])
    chunk_size = int(config["semantic_attention_topk"]["vocab_chunk_size"])
    target_ids = [int(t["target_token_id"]) for t in targets]
    problems = {str(index): [] for index in range(len(targets))}
    shifts = {str(index): [] for index in range(len(targets))}
    js_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    sar_values = []

    for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
        hpre = capture["h_prev"][0].float()
        visual = hpre[visual_start:visual_end]

        def compute_semantic() -> torch.Tensor:
            _raw, norm = target_probabilities(
                visual, target_ids, output_layer, final_norm, chunk_size,
            )
            return torch.log1p(float(vocab_size) * norm.float())

        support = _measure(device, times, "semantic_probability", compute_semantic)
        cost = _measure(
            device, times, "ot_cost", lambda: _cosine_distance_matrix(visual),
        )

        def compute_sar() -> torch.Tensor:
            writes = reconstruct_visual_directions(
                layer=layer, capture=capture, prediction_positions=positions,
                visual_start=visual_start, visual_end=visual_end,
            )["a_tokens"].float()
            magnitudes = writes.norm(dim=-1).transpose(0, 1)
            if magnitudes.shape != support.shape:
                raise ValueError(f"{model}/{image_id}/layer{layer_index}: SAR shape mismatch")
            return (magnitudes * (1.0 + support)).sum(dim=-1)

        sar = _measure(device, times, "sar_write", compute_sar)
        sar_values.extend(float(v) for v in sar.tolist())

        for index, position in enumerate(positions):
            def compute_distributions() -> tuple[torch.Tensor, torch.Tensor]:
                attention = _mean_head_visual_attention(
                    capture, position, visual_start, visual_end,
                )
                target = _renormalize(attention * support[index])
                scores = F.cosine_similarity(
                    capture["o_ffn"][0, position].float().unsqueeze(0),
                    visual, dim=-1,
                )
                source = _source_from_scores(scores, original.TAU)
                return source, target

            source, target = _measure(
                device, times, "source_target", compute_distributions,
            )

            def prepare_ot() -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
                penalty = _penalty(hpre[position], visual)
                return _problem_from_full_cost(
                    source=source, target=target, full_cost=cost,
                    penalty=penalty, top_k=original.TOP_K,
                )

            problem, shift = _measure(device, times, "ot_problem", prepare_ot)
            problems[str(index)].append(problem)
            shifts[str(index)].append(shift)

            def prepare_js() -> tuple[np.ndarray, np.ndarray]:
                # _topk_union_indices divides its budget equally across P and T.
                union = _topk_union_indices(source, target, 2 * js_top_each)
                local_source = _renormalize(source.index_select(0, union))
                local_target = _renormalize(target.index_select(0, union))
                return (
                    local_source.detach().float().cpu().numpy().copy(),
                    local_target.detach().float().cpu().numpy().copy(),
                )

            js_pairs.append(_measure(device, times, "js_support", prepare_js))
        captures[layer_index] = None
        del capture, hpre, visual, support, cost, sar

    solved = _measure(
        device, times, "ot_solve", lambda: _solve_exact_emd_problem_series(problems),
    )
    js_values = _measure(
        device, times, "js_core",
        lambda: [_js_natural(left, right) for left, right in js_pairs],
    )
    if len(js_values) != len(targets) * expected_layers:
        raise ValueError("JS count mismatch")
    if not np.isfinite(js_values).all() or not np.isfinite(sar_values).all():
        raise ValueError("Nonfinite JS or SAR result")
    ot_values = [
        value + shift
        for index in range(len(targets))
        for value, shift in zip(solved[str(index)], shifts[str(index)])
    ]
    if len(ot_values) != len(js_values) or not np.isfinite(ot_values).all():
        raise ValueError("Nonfinite or missing OT result")
    return {
        "image_id": image_id, "objects": len(targets),
        "layers": expected_layers, "visual_tokens": visual_end - visual_start,
        "distribution_pairs": len(js_values),
        "seconds": {stage: times[stage] for stage in STAGES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=original.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--js-top-each", type=int, default=16)
    args = parser.parse_args()
    if args.js_top_each <= 0:
        parser.error("--js-top-each must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    image_ids, warmup_id = original.benchmark_image_ids(args.num_images)
    source = original._source_dir(args.model)
    labels = {int(k): v for k, v in original._json(source / "labeling.json").items()}
    generations = {
        int(k): v for k, v in original._json(source / "generations.json").items()
    }
    config = load_config(str(original.OURS_CONFIG))
    baseline_config = load_config(str(original.BASELINE_CONFIG))
    wrapper = build_model(
        args.model, get_extraction_model_cfg(baseline_config, args.model),
        device=args.device,
    )
    wrapper.model.eval().requires_grad_(False)
    kwargs = {"wrapper": wrapper, "model": args.model, "config": config, "device": device,
              "js_top_each": args.js_top_each}
    profile_image(
        image_id=warmup_id, label=labels[warmup_id],
        generation=generations[warmup_id], **kwargs,
    )
    rows = []
    for offset, image_id in enumerate(image_ids, 1):
        row = profile_image(
            image_id=image_id, label=labels[image_id],
            generation=generations[image_id], **kwargs,
        )
        rows.append(row)
        print(f"[{args.model}] {offset}/{len(image_ids)}", flush=True)
    summary = {}
    for stage in STAGES:
        values = [row["seconds"][stage] for row in rows]
        summary[stage] = {
            "total_seconds": sum(values),
            "mean_seconds_per_image": statistics.mean(values),
            "median_seconds_per_image": statistics.median(values),
        }
    payload = {
        "schema": "k32-sar-ems-js-ot-stage-profile-v1",
        "model": args.model, "device": args.device,
        "image_ids": image_ids, "warmup_image_id": warmup_id,
        "top_k": original.TOP_K, "tau": original.TAU,
        "js_top_each": args.js_top_each,
        "object_tokens": sum(row["objects"] for row in rows),
        "distribution_pairs": sum(row["distribution_pairs"] for row in rows),
        "stage_summary": summary, "per_image": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.model}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
