#!/usr/bin/env python3
"""Time optimized SAR/EMS components and caption generation on shared COCO images.

The feature path computes only final-norm target probabilities and projects all
visual tokens in one batch. Model loading and image file reads are excluded from
the component rows; teacher-forced capture is measured separately for context.
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
from scripts import benchmark_k32_final_only_vocab as optimized
from scripts import benchmark_k32_sar_ems_js_ot_stages as stages
from models import build_model
from utils.config_utils import get_dataset_cfg, get_extraction_model_cfg, load_config


OUT = Path(__file__).resolve().parents[1] / "outputs/complexity_sam_components_optimized_50"


def _mean_ms(rows: list[dict[str, Any]], key: str) -> float:
    return 1000.0 * statistics.mean(row[key] for row in rows)


@torch.inference_mode()
def _generate(*, wrapper: Any, image_path: Path, prompt: str, device: torch.device) -> dict[str, Any]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = wrapper.generate(image, prompt=prompt)
    torch.cuda.synchronize(device)
    return {
        "seconds": time.perf_counter() - started,
        "response_tokens": len(output.response_token_ids),
    }


def run(model: str, device_name: str, num_images: int, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    image_ids, warmup_id = original.benchmark_image_ids(num_images)
    source = original._source_dir(model)
    labels = {int(k): v for k, v in original._json(source / "labeling.json").items()}
    generations = {int(k): v for k, v in original._json(source / "generations.json").items()}
    config = load_config(str(original.OURS_CONFIG))
    config["semantic_attention_topk"]["vocab_chunk_size"] = optimized.ALL_CHUNK
    baseline_config = load_config(str(original.BASELINE_CONFIG))
    wrapper = build_model(model, get_extraction_model_cfg(baseline_config, model), device=device_name)
    wrapper.model.eval().requires_grad_(False)
    prompt = str(config.get("run", {}).get("prompt") or "Describe this image.")
    coco_root = Path(get_dataset_cfg(config)["coco_root"]) / "val2014"

    # The stage profiler looks up this module-level probability readout.
    production_function = stages.target_probabilities
    stages.target_probabilities = optimized.target_probabilities_final_only
    stage_kwargs = {
        "wrapper": wrapper, "model": model, "config": config, "device": device,
    }

    def image_path(image_id: int) -> Path:
        return coco_root / f"COCO_val2014_{image_id:012d}.jpg"

    try:
        _generate(wrapper=wrapper, image_path=image_path(warmup_id), prompt=prompt, device=device)
        stages.profile_image(
            image_id=warmup_id, label=labels[warmup_id],
            generation=generations[warmup_id], **stage_kwargs,
        )
        rows: list[dict[str, Any]] = []
        for offset, image_id in enumerate(image_ids, 1):
            # Alternating order reduces systematic GPU/cache warmness effects.
            if offset % 2:
                generation = _generate(
                    wrapper=wrapper, image_path=image_path(image_id), prompt=prompt, device=device,
                )
                feature = stages.profile_image(
                    image_id=image_id, label=labels[image_id],
                    generation=generations[image_id], **stage_kwargs,
                )
            else:
                feature = stages.profile_image(
                    image_id=image_id, label=labels[image_id],
                    generation=generations[image_id], **stage_kwargs,
                )
                generation = _generate(
                    wrapper=wrapper, image_path=image_path(image_id), prompt=prompt, device=device,
                )
            t = feature["seconds"]
            rows.append({
                "image_id": image_id,
                "response_tokens": generation["response_tokens"],
                "object_tokens": feature["objects"],
                "generation": generation["seconds"],
                "teacher_forward": t["teacher_forward"],
                "image_load": t["image_load"],
                "semantic_probability": t["semantic_probability"],
                "sar_write": t["sar_write"],
                "ems_source_target": t["source_target"],
                "ems_cost": t["ot_cost"],
                "ems_problem": t["ot_problem"],
                "ems_solve": t["ot_solve"],
                "sar": t["semantic_probability"] + t["sar_write"],
                "ems_after_shared_probability": (
                    t["source_target"] + t["ot_cost"] + t["ot_problem"] + t["ot_solve"]
                ),
                "sam_feature_postprocessing": (
                    t["semantic_probability"] + t["sar_write"] + t["source_target"]
                    + t["ot_cost"] + t["ot_problem"] + t["ot_solve"]
                ),
            })
            print(f"[{model}] {offset}/{num_images}", flush=True)

        keys = [key for key in rows[0] if key not in {"image_id", "response_tokens", "object_tokens"}]
        summary = {key: {"mean_ms_per_image": _mean_ms(rows, key)} for key in keys}
        payload = {
            "schema": "sam-component-time-optimized-v1",
            "model": model,
            "device": device_name,
            "image_ids": image_ids,
            "warmup_image_id": warmup_id,
            "vocab_chunk_size": optimized.ALL_CHUNK,
            "probability_readout": "final_norm_only",
            "top_k": original.TOP_K,
            "source_tau": original.TAU,
            "generation_max_new_tokens": wrapper.generation_max_new_tokens,
            "timing_note": "SAM=optimized feature postprocessing from captured states; SAR includes shared probability; EMS excludes shared probability so SAR+EMS=SAM. Answer generation uses wrapper.generate and excludes image file read. Teacher-forced capture separately recorded. Model load, serialization, training excluded.",
            "objects_total": sum(row["object_tokens"] for row in rows),
            "generated_tokens_total": sum(row["response_tokens"] for row in rows),
            "summary": summary,
            "per_image": rows,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{model}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {destination}", flush=True)
    finally:
        stages.target_probabilities = production_function


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=original.MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.model, args.device, args.num_images, args.output_dir)


if __name__ == "__main__":
    main()
