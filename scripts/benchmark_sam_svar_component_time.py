#!/usr/bin/env python3
"""Paired optimized SAM and baseline extraction timing with stage boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from PIL import Image
import torch

from scripts import benchmark_k32_feature_complexity as original
from scripts import benchmark_k32_final_only_vocab as optimized
from scripts import benchmark_k32_sar_ems_js_ot_stages as sam_stages
from features.baseline import BaselineRuntime, baseline_config
from models import build_model
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_sam_svar_paired_optimized_50"


def _measure(device: torch.device, function: Any) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - start


def _profile_memory(device: torch.device, function: Any) -> tuple[Any, dict[str, int]]:
    """Measure PyTorch memory for one complete method call after warm-up."""
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    value = function()
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    return value, {
        "baseline_allocated_bytes": int(baseline_allocated),
        "peak_allocated_bytes": int(peak_allocated),
        "incremental_peak_bytes": int(peak_allocated - baseline_allocated),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


@torch.inference_mode()
def profile_baseline(
    *, wrapper: Any, runtime: BaselineRuntime, config: dict[str, Any],
    image_id: int, label: dict[str, Any], generation: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    response_ids = [int(value) for value in generation["response_token_ids"]]
    targets = original._targets(label, response_ids)
    spans = original._spans_from_targets(targets)
    indices = list(dict.fromkeys(int(target["response_index"]) for target in targets))
    target_ids = [response_ids[index] for index in indices]
    path = (
        Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )

    def load_image() -> Image.Image:
        with Image.open(path) as opened:
            return opened.convert("RGB")

    image, image_load = _measure(device, load_image)
    outputs, capture = _measure(
        device,
        lambda: wrapper.extract_token_features_batch(
            image=image,
            response_token_ids=response_ids,
            response_token_indices=indices,
            target_token_ids=target_ids,
            cfg_dgst_t=None,
            prompt=str(config.get("run", {}).get("prompt") or "Describe this image."),
            requirements=runtime.requirements_for(controlled=True, official=False),
        ),
    )
    output_by_index = dict(zip(indices, outputs))
    records, postcapture = _measure(
        device,
        lambda: runtime.build_image_records(
            image=image,
            image_id=image_id,
            response_token_ids=response_ids,
            spans=spans,
            model_outputs=[
                output_by_index[int(target["response_index"])] for target in targets
            ],
        ),
    )
    method = runtime.methods[0]
    if len(records) != len(targets):
        raise ValueError(f"{image_id}: {method} records/object count mismatch")
    if method not in records[0]["baselines"]:
        raise ValueError(f"{image_id}: missing {method} feature")
    return {
        "image_id": image_id,
        "object_tokens": len(targets),
        "image_load": image_load,
        "capture": capture,
        "postcapture": postcapture,
        "full_extraction": image_load + capture + postcapture,
        "capture_plus_postcapture": capture + postcapture,
    }


def run(
    model: str, device_name: str, num_images: int, output_dir: Path,
    include_token_grounding: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    image_ids, warmup_id = original.benchmark_image_ids(num_images)
    source = original._source_dir(model)
    labels = {int(k): v for k, v in original._json(source / "labeling.json").items()}
    generations = {int(k): v for k, v in original._json(source / "generations.json").items()}
    ours_config = load_config(str(original.OURS_CONFIG))
    ours_config["semantic_attention_topk"]["vocab_chunk_size"] = optimized.ALL_CHUNK
    base_config = load_config(str(original.BASELINE_CONFIG))
    wrapper = build_model(model, get_extraction_model_cfg(base_config, model), device=device_name)
    wrapper.model.eval().requires_grad_(False)
    methods = ("sam", "svar", "token_grounding") if include_token_grounding else ("sam", "svar")
    runtimes = {
        method: BaselineRuntime(
            wrapper=wrapper, methods=(method,),
            baseline_dir=output_dir / "_runtime_scratch" / model / method,
            config=baseline_config(base_config), device=device_name,
            resume=False, inline_dhcp=True,
        )
        for method in methods if method != "sam"
    }
    production_fn = sam_stages.target_probabilities
    sam_stages.target_probabilities = optimized.target_probabilities_final_only
    sam_kwargs = {"wrapper": wrapper, "model": model, "config": ours_config, "device": device}
    try:
        sam_stages.profile_image(
            image_id=warmup_id, label=labels[warmup_id],
            generation=generations[warmup_id], **sam_kwargs,
        )
        for method, runtime in runtimes.items():
            profile_baseline(
                image_id=warmup_id, label=labels[warmup_id],
                generation=generations[warmup_id], wrapper=wrapper,
                runtime=runtime, config=base_config, device=device,
            )
        rows = []
        for offset, image_id in enumerate(image_ids, 1):
            order = methods[offset % len(methods):] + methods[:offset % len(methods)]
            measured = {}
            for method in order:
                if method == "sam":
                    function = lambda: sam_stages.profile_image(
                        image_id=image_id, label=labels[image_id],
                        generation=generations[image_id], **sam_kwargs,
                    )
                else:
                    function = lambda method=method: profile_baseline(
                        image_id=image_id, label=labels[image_id],
                        generation=generations[image_id], wrapper=wrapper,
                        runtime=runtimes[method], config=base_config, device=device,
                    )
                measured[method] = _profile_memory(device, function)
            sam, sam_memory = measured["sam"]
            if any(sam["objects"] != measured[method][0]["object_tokens"] for method in runtimes):
                raise ValueError(f"{model}/{image_id}: object count mismatch")
            t = sam["seconds"]
            sam_sar = t["semantic_probability"] + t["sar_write"]
            sam_ems = t["source_target"] + t["ot_cost"] + t["ot_problem"] + t["ot_solve"]
            row = {
                "image_id": image_id,
                "object_tokens": sam["objects"],
                "sam": {
                    "memory": sam_memory,
                    "image_load": t["image_load"],
                    "capture": t["teacher_forward"],
                    "sar": sam_sar,
                    "ems_after_shared_probability": sam_ems,
                    "postcapture": sam_sar + sam_ems,
                    "capture_plus_postcapture": t["teacher_forward"] + sam_sar + sam_ems,
                    "full_extraction": t["image_load"] + t["teacher_forward"] + sam_sar + sam_ems,
                },
            }
            for method in runtimes:
                value, memory = measured[method]
                row[method] = {
                    "memory": memory,
                    **{key: value[key] for key in (
                        "image_load", "capture", "postcapture", "capture_plus_postcapture", "full_extraction"
                    )},
                }
            rows.append(row)
            print(f"[{model}] {offset}/{num_images}", flush=True)

        summary = {}
        for method in methods:
            summary[method] = {
                key: {
                    "mean_ms_per_image": 1000 * statistics.mean(row[method][key] for row in rows),
                    "median_ms_per_image": 1000 * statistics.median(row[method][key] for row in rows),
                }
                for key in rows[0][method] if key != "memory"
            }
            summary[method]["memory"] = {
                "peak_allocated_gib": max(row[method]["memory"]["peak_allocated_bytes"] for row in rows) / 1024**3,
                "peak_reserved_gib": max(row[method]["memory"]["peak_reserved_bytes"] for row in rows) / 1024**3,
                "max_incremental_peak_gib": max(row[method]["memory"]["incremental_peak_bytes"] for row in rows) / 1024**3,
                "mean_incremental_peak_gib": statistics.mean(row[method]["memory"]["incremental_peak_bytes"] for row in rows) / 1024**3,
                "mean_baseline_allocated_gib": statistics.mean(row[method]["memory"]["baseline_allocated_bytes"] for row in rows) / 1024**3,
            }
        payload = {
            "schema": "sam-baselines-paired-component-time-v2",
            "model": model,
            "device": device_name,
            "methods": methods,
            "image_ids": image_ids,
            "warmup_image_id": warmup_id,
            "object_tokens": sum(row["object_tokens"] for row in rows),
            "sam_vocab_chunk_size": optimized.ALL_CHUNK,
            "sam_probability_readout": "final_norm_only",
            "sam_top_k": original.TOP_K,
            "sam_source_tau": original.TAU,
            "timing_note": "Paired same-image rotating measurement. Postcapture excludes image loading and method-specific state capture. Capture_plus_postcapture includes each method's own capture but excludes image file read. Model load, generation, serialization, training excluded.",
            "summary": summary,
            "per_image": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {path}", flush=True)
    finally:
        sam_stages.target_probabilities = production_fn
        for runtime in runtimes.values():
            runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=original.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--include-token-grounding", action="store_true")
    args = parser.parse_args()
    run(args.model, args.device, args.num_images, args.output_dir, args.include_token_grounding)


if __name__ == "__main__":
    main()
