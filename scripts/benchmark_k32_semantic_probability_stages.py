#!/usr/bin/env python3
"""Profile the original raw+norm target-probability readout on 50 COCO images.

CUDA events separate the full-vocabulary projection, FP32 logits conversion,
and logsumexp. The complete operation order matches target_probabilities().
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
import torch.nn.functional as F

from scripts import benchmark_k32_feature_complexity as original
from features.semantic_attention_topk import target_probabilities
from models import build_model
from models.dgst_capture import (
    apply_decoder_final_norm,
    resolve_decoder_final_norm,
    resolve_decoder_layers,
    resolve_output_embedding_layer,
)
from scripts.extract_endac_811 import _capture, _targets
from utils.config_utils import (
    get_dataset_cfg,
    get_extraction_model_cfg,
    load_config,
)


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_k032_50_semantic_probability_profile"
PHASES = ("projection", "fp32_cast", "logsumexp")


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


@torch.inference_mode()
def profile_probabilities(
    visual_states: torch.Tensor,
    target_ids: list[int],
    output_layer: Any,
    final_norm: Any,
    chunk_size: int,
) -> dict[str, Any]:
    """Use original operation order; record CUDA time per phase and wall time."""
    weight = output_layer.weight
    targets = torch.tensor(target_ids, dtype=torch.long, device=weight.device)
    raw_parts, norm_parts = [], []
    chunk_count = (len(visual_states) + max(1, int(chunk_size)) - 1) // max(1, int(chunk_size))
    phase_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        phase: [_event_pair() for _ in range(chunk_count)] for phase in PHASES
    }
    stream = torch.cuda.current_stream(weight.device)
    torch.cuda.synchronize(weight.device)
    outer_start, outer_end = _event_pair()
    outer_start.record(stream)
    wall_start = time.perf_counter()
    chunks = 0
    for start in range(0, len(visual_states), max(1, int(chunk_size))):
        raw = visual_states[start : start + int(chunk_size)].to(
            device=weight.device, dtype=weight.dtype,
        )
        normed = apply_decoder_final_norm(final_norm, raw)
        states = torch.cat((raw, normed), dim=0)

        first, last = phase_events["projection"][chunks]
        first.record(stream)
        logits = F.linear(states, weight, bias=None)
        last.record(stream)

        first, last = phase_events["fp32_cast"][chunks]
        first.record(stream)
        logits_fp32 = logits.float()
        last.record(stream)

        first, last = phase_events["logsumexp"][chunks]
        first.record(stream)
        log_partition = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
        last.record(stream)

        selected = logits.index_select(-1, targets).float()
        probabilities = torch.exp(selected - log_partition)
        raw_probability, norm_probability = probabilities.split(len(raw), dim=0)
        raw_parts.append(raw_probability.transpose(0, 1))
        norm_parts.append(norm_probability.transpose(0, 1))
        chunks += 1
        del raw, normed, states, logits, logits_fp32, log_partition, selected, probabilities
    raw_all = torch.cat(raw_parts, dim=1)
    norm_all = torch.cat(norm_parts, dim=1)
    outer_end.record(stream)
    torch.cuda.synchronize(weight.device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    total_gpu_ms = outer_start.elapsed_time(outer_end)
    phase_ms = {
        phase: sum(first.elapsed_time(last) for first, last in events)
        for phase, events in phase_events.items()
    }
    return {
        "raw": raw_all,
        "norm": norm_all,
        "wall_ms": wall_ms,
        "gpu_ms": total_gpu_ms,
        "phase_ms": phase_ms,
        "chunks": chunks,
        "visual_tokens": len(visual_states),
        "target_tokens": len(target_ids),
    }


@torch.inference_mode()
def profile_image(
    *, wrapper: Any, model: str, config: dict[str, Any], image_id: int,
    label: dict[str, Any], generation: dict[str, Any], verify: bool,
) -> dict[str, Any]:
    response_ids = [int(value) for value in generation["response_token_ids"]]
    targets = _targets(label, response_ids)
    if not targets:
        raise ValueError(f"Image {image_id} has no object targets")
    path = (
        Path(get_dataset_cfg(config)["coco_root"]) / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )
    with Image.open(path) as opened:
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
    chunk_size = int(config["semantic_attention_topk"]["vocab_chunk_size"])
    target_ids = [int(t["target_token_id"]) for t in targets]
    rows = []
    for layer_index, capture in enumerate(captures):
        visual = capture["h_prev"][0, visual_start:visual_end].float()
        result = profile_probabilities(
            visual, target_ids, output_layer, final_norm, chunk_size,
        )
        if verify and layer_index == 0:
            original_raw, original_norm = target_probabilities(
                visual, target_ids, output_layer, final_norm, chunk_size,
            )
            torch.cuda.synchronize(output_layer.weight.device)
            if not torch.equal(result["raw"], original_raw) or not torch.equal(
                result["norm"], original_norm
            ):
                raw_delta = (result["raw"] - original_raw).abs().max().item()
                norm_delta = (result["norm"] - original_norm).abs().max().item()
                raise ValueError(
                    f"Probability readout mismatch: raw={raw_delta}, norm={norm_delta}"
                )
        rows.append({
            "layer": layer_index + 1,
            "wall_ms": result["wall_ms"],
            "gpu_ms": result["gpu_ms"],
            "phase_ms": result["phase_ms"],
            "chunks": result["chunks"],
        })
        captures[layer_index] = None
        del capture, visual, result
    return {
        "image_id": image_id,
        "object_tokens": len(target_ids),
        "visual_tokens": visual_end - visual_start,
        "layers": rows,
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
    kwargs = {"wrapper": wrapper, "model": args.model, "config": config}
    profile_image(
        image_id=warmup_id, label=labels[warmup_id],
        generation=generations[warmup_id], verify=True, **kwargs,
    )
    rows = []
    for offset, image_id in enumerate(image_ids, 1):
        rows.append(profile_image(
            image_id=image_id, label=labels[image_id],
            generation=generations[image_id], verify=False, **kwargs,
        ))
        print(f"[{args.model}] {offset}/{len(image_ids)}", flush=True)
    keys = ("wall_ms", "gpu_ms", *PHASES)
    per_image = []
    for row in rows:
        totals = {
            key: sum(
                layer[key] if key in ("wall_ms", "gpu_ms") else layer["phase_ms"][key]
                for layer in row["layers"]
            )
            for key in keys
        }
        per_image.append({
            "image_id": row["image_id"],
            "object_tokens": row["object_tokens"],
            "visual_tokens": row["visual_tokens"],
            "chunks": sum(layer["chunks"] for layer in row["layers"]),
            "milliseconds": totals,
        })
    summary = {
        key: {
            "mean_ms_per_image": statistics.mean(row["milliseconds"][key] for row in per_image),
            "median_ms_per_image": statistics.median(row["milliseconds"][key] for row in per_image),
        }
        for key in keys
    }
    payload = {
        "schema": "k32-original-semantic-probability-stage-profile-v1",
        "model": args.model, "device": args.device,
        "image_ids": image_ids, "warmup_image_id": warmup_id,
        "vocabulary_size": int(resolve_output_embedding_layer(wrapper.model).weight.shape[0]),
        "chunk_size": int(config["semantic_attention_topk"]["vocab_chunk_size"]),
        "object_tokens": sum(row["object_tokens"] for row in per_image),
        "per_image": per_image, "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"{args.model}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
