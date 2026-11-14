#!/usr/bin/env python3
"""Measure original raw+norm target-probability readout at several chunk sizes.

Model captures are shared among chunk sizes; only the per-layer vocabulary
readout is timed. The original probability implementation is called unchanged.
"""

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
from features.semantic_attention_topk import target_probabilities
from models import build_model
from models.dgst_capture import (
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
)
from scripts.extract_endac_811 import _capture, _targets
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_k032_50_vocab_chunk_sweep"
CHUNKS = (32, 64, 128, 256)
GIB = 1024**3


@torch.inference_mode()
def profile_image(
    *, wrapper: Any, model: str, config: dict[str, Any], image_id: int,
    label: dict[str, Any], generation: dict[str, Any], device: torch.device,
    image_index: int,
) -> dict[str, Any]:
    response_ids = [int(value) for value in generation["response_token_ids"]]
    targets = _targets(label, response_ids)
    if not targets:
        raise ValueError(f"Image {image_id} has no object tokens")
    image_path = (
        Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )
    with Image.open(image_path) as opened:
        captures, _positions, visual_start, visual_end, _grid, _stats = _capture(
            wrapper, model, opened.convert("RGB"), response_ids, targets,
            str(config.get("run", {}).get("prompt") or "Describe this image."),
            collect_response_statistics=False,
        )
    layers = resolve_decoder_layers(wrapper.model)
    if len(captures) != len(layers):
        raise ValueError("Unexpected decoder-layer count")
    output_layer = resolve_output_embedding_layer(wrapper.model)
    final_norm = resolve_decoder_final_norm(wrapper.model)
    token_ids = [int(target["target_token_id"]) for target in targets]
    seconds = {str(chunk): 0.0 for chunk in CHUNKS}
    peak_allocated = {str(chunk): 0 for chunk in CHUNKS}
    peak_reserved = {str(chunk): 0 for chunk in CHUNKS}
    extra_allocated = {str(chunk): 0 for chunk in CHUNKS}
    numerical = {str(chunk): {"max_abs": 0.0, "abs_sum": 0.0, "elements": 0} for chunk in CHUNKS[1:]}

    for layer_index, capture in enumerate(captures):
        visual = capture["h_prev"][0, visual_start:visual_end].float()
        order = CHUNKS[(image_index + layer_index) % len(CHUNKS):] + CHUNKS[:(image_index + layer_index) % len(CHUNKS)]
        norm_by_chunk = {}
        for chunk in order:
            key = str(chunk)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            raw_probability, norm_probability = target_probabilities(
                visual, token_ids, output_layer, final_norm, chunk,
            )
            torch.cuda.synchronize(device)
            seconds[key] += time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device)
            peak_allocated[key] = max(peak_allocated[key], peak)
            peak_reserved[key] = max(peak_reserved[key], torch.cuda.max_memory_reserved(device))
            extra_allocated[key] = max(extra_allocated[key], peak - before)
            norm_by_chunk[key] = norm_probability
            del raw_probability
        reference = norm_by_chunk[str(CHUNKS[0])]
        for chunk in CHUNKS[1:]:
            key = str(chunk)
            delta = (reference.float() - norm_by_chunk[key].float()).abs()
            numerical[key]["max_abs"] = max(numerical[key]["max_abs"], float(delta.max().item()))
            numerical[key]["abs_sum"] += float(delta.sum().item())
            numerical[key]["elements"] += delta.numel()
        captures[layer_index] = None
        del capture, visual, norm_by_chunk
    return {
        "image_id": image_id,
        "object_tokens": len(targets),
        "visual_tokens": visual_end - visual_start,
        "layers": len(layers),
        "seconds": seconds,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "extra_allocated_bytes": extra_allocated,
        "numeric_vs_32": numerical,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=original.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    image_ids, warmup_id = original.benchmark_image_ids(args.num_images)
    source = original._source_dir(args.model)
    labels = {int(key): value for key, value in original._json(source / "labeling.json").items()}
    generations = {
        int(key): value for key, value in original._json(source / "generations.json").items()
    }
    config = load_config(str(original.OURS_CONFIG))
    baseline_config = load_config(str(original.BASELINE_CONFIG))
    wrapper = build_model(
        args.model, get_extraction_model_cfg(baseline_config, args.model),
        device=args.device,
    )
    wrapper.model.eval().requires_grad_(False)
    torch.cuda.synchronize(device)
    loaded_allocated = torch.cuda.memory_allocated(device)
    loaded_reserved = torch.cuda.memory_reserved(device)
    kwargs = {"wrapper": wrapper, "model": args.model, "config": config, "device": device}
    profile_image(
        image_id=warmup_id, label=labels[warmup_id],
        generation=generations[warmup_id], image_index=-1, **kwargs,
    )
    rows = []
    for index, image_id in enumerate(image_ids):
        row = profile_image(
            image_id=image_id, label=labels[image_id],
            generation=generations[image_id], image_index=index, **kwargs,
        )
        rows.append(row)
        print(f"[{args.model}] {index + 1}/{len(image_ids)}", flush=True)
    variants = {}
    for chunk in CHUNKS:
        key = str(chunk)
        values = [row["seconds"][key] for row in rows]
        item = {
            "seconds_per_image": statistics.mean(values),
            "median_seconds_per_image": statistics.median(values),
            "total_seconds": sum(values),
            "peak_allocated_gib": max(row["peak_allocated_bytes"][key] for row in rows) / GIB,
            "peak_reserved_gib": max(row["peak_reserved_bytes"][key] for row in rows) / GIB,
            "max_extra_allocated_gib": max(row["extra_allocated_bytes"][key] for row in rows) / GIB,
        }
        if chunk != CHUNKS[0]:
            parts = [row["numeric_vs_32"][key] for row in rows]
            item["norm_probability_vs_32"] = {
                "max_abs": max(part["max_abs"] for part in parts),
                "mean_abs": sum(part["abs_sum"] for part in parts) / sum(part["elements"] for part in parts),
                "elements": sum(part["elements"] for part in parts),
            }
        variants[key] = item
    payload = {
        "schema": "original-raw-norm-vocab-chunk-sweep-v1",
        "model": args.model, "device": args.device,
        "image_ids": image_ids, "warmup_image_id": warmup_id,
        "chunks": CHUNKS,
        "vocabulary_size": int(resolve_output_embedding_layer(wrapper.model).weight.shape[0]),
        "loaded_allocated_gib": loaded_allocated / GIB,
        "loaded_reserved_gib": loaded_reserved / GIB,
        "object_tokens": sum(row["object_tokens"] for row in rows),
        "variants": variants,
        "per_image": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.model}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
