#!/usr/bin/env python3
"""Benchmark K=32 object-feature extraction against trainable MLP baselines.

The benchmark starts from saved generations.  It includes image loading,
teacher-forced LVLM forward passes, and feature post-processing, while excluding
model loading, caption generation, serialization, and detector training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))
# ``scripts`` is a namespace package in both repositories.  Appending gfchair
# exposes the ENDAC Log-S extractors while keeping the newer baseline modules
# and model wrappers from token-detector authoritative.
sys.path.append(str(GF_ROOT))

import features  # noqa: E402
import models  # noqa: E402

# ``features`` is a regular package, so explicitly extend its search path for
# the two ENDAC-only feature modules that live in gfchair.
features.__path__.append(str(GF_ROOT / "features"))
models.__path__.append(str(GF_ROOT / "models"))

from features.baseline import BaselineRuntime, baseline_config  # noqa: E402
from features.dgst_t import (  # noqa: E402
    _cosine_distance_matrix,
    _renormalize,
    _solve_exact_emd_problem_series,
)
from features.semantic_attention_topk import target_probabilities  # noqa: E402
from features.visual_ffn_jacobian import reconstruct_visual_directions  # noqa: E402
from models import build_model  # noqa: E402
from models.dgst_capture import (  # noqa: E402
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
from utils.io_utils import load_json  # noqa: E402


MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
METHODS = ("ours", "svar", "dhcp", "token_grounding")
DISPLAY = {
    "ours": "Ours (SAR + EMS)",
    "svar": "SVAR",
    "dhcp": "DHCP",
    "token_grounding": "TokenGrounding-MLP",
}
TOP_K = 32
TAU = 0.03
DEFAULT_NUM_IMAGES = 50
BASELINE_CONFIG = ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
OURS_CONFIG = GF_ROOT / "configs/model_configs_endac_811.yaml"
SOURCE_RUN = "COCO4000-512-ENDAC-BASELINES-V2"
OUTPUT = ROOT / "outputs/complexity_k032_50"


def _json(path: Path) -> dict[str, Any]:
    return load_json(str(path))


def _source_dir(model: str) -> Path:
    return ROOT / "outputs" / model / SOURCE_RUN


def _valid_image_ids(model: str) -> set[int]:
    source = _source_dir(model)
    labels = {int(key): value for key, value in _json(source / "labeling.json").items()}
    generations = {
        int(key): value for key, value in _json(source / "generations.json").items()
    }
    valid: set[int] = set()
    for image_id, label in labels.items():
        generation = generations.get(image_id)
        if not isinstance(generation, Mapping):
            continue
        response_ids = [int(value) for value in generation.get("response_token_ids") or []]
        if not response_ids or str(generation.get("generated_text", "")) != str(
            label.get("generated_text", "")
        ):
            continue
        if _targets(label, response_ids):
            valid.add(image_id)
    return valid


def benchmark_image_ids(num_images: int) -> tuple[list[int], int]:
    common = sorted(set.intersection(*(_valid_image_ids(model) for model in MODELS)))
    if len(common) < num_images + 1:
        raise ValueError(
            f"Need {num_images + 1} shared extractable images including warm-up; "
            f"found {len(common)}"
        )
    return common[:num_images], common[num_images]


def _spans_from_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "word": target["token"],
            "label": int(target["label"]),
            "token_indices": [int(value) for value in target["token_indices"]],
            "occurrence_count": int(target.get("occurrence_count", 1)),
        }
        for target in targets
    ]


@torch.inference_mode()
def extract_ours(
    *,
    wrapper: Any,
    model: str,
    config: dict[str, Any],
    image_id: int,
    label: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Compute exactly the deployed K=32 EMS and SAR/F_log feature blocks."""

    response_ids = [int(value) for value in generation["response_token_ids"]]
    targets = _targets(label, response_ids)
    if not targets:
        raise ValueError(f"Image {image_id} has no extractable object targets")
    image_path = (
        Path(get_dataset_cfg(config)["coco_root"])
        / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )
    with Image.open(image_path) as source:
        captures, positions, visual_start, visual_end, _grid, _stats = _capture(
            wrapper,
            model,
            source.convert("RGB"),
            response_ids,
            targets,
            str(config.get("run", {}).get("prompt") or "Describe this image."),
            collect_response_statistics=False,
        )

    layers = resolve_decoder_layers(wrapper.model)
    expected_layers = int(config["endac_811"]["models"][model]["num_layers"])
    if len(layers) != expected_layers or len(captures) != expected_layers:
        raise ValueError(f"{model}: expected {expected_layers} decoder layers")
    output_layer = resolve_output_embedding_layer(wrapper.model)
    final_norm = resolve_decoder_final_norm(wrapper.model)
    vocabulary_size = int(output_layer.weight.shape[0])
    token_ids = [int(target["target_token_id"]) for target in targets]
    problems = {str(index): [] for index in range(len(targets))}
    shifts = {str(index): [] for index in range(len(targets))}
    f_log = [[] for _ in targets]
    vocab_chunk_size = int(config["semantic_attention_topk"]["vocab_chunk_size"])

    for layer_index, (layer, capture) in enumerate(zip(layers, captures)):
        hpre = capture["h_prev"][0].float()
        visual_hpre = hpre[int(visual_start) : int(visual_end)]
        _raw_probability, norm_probability = target_probabilities(
            visual_hpre,
            token_ids,
            output_layer,
            final_norm,
            vocab_chunk_size,
        )
        full_cost = _cosine_distance_matrix(visual_hpre)

        # SAR uses the norm of each visual key position's attention write.
        writes = reconstruct_visual_directions(
            layer=layer,
            capture=capture,
            prediction_positions=positions,
            visual_start=visual_start,
            visual_end=visual_end,
        )["a_tokens"].float()
        am_norm = writes.norm(dim=-1).transpose(0, 1)
        if am_norm.shape != norm_probability.shape:
            raise ValueError(
                f"{model}/{image_id}/layer{layer_index}: SAR shape mismatch "
                f"{tuple(am_norm.shape)} != {tuple(norm_probability.shape)}"
            )
        log_support = torch.log1p(float(vocabulary_size) * norm_probability.float())
        sar = (am_norm * (1.0 + log_support)).sum(dim=-1)

        for target_index, prediction_position in enumerate(positions):
            attention = _mean_head_visual_attention(
                capture, prediction_position, visual_start, visual_end
            )
            target_dist = _renormalize(attention * log_support[target_index])
            update_scores = F.cosine_similarity(
                capture["o_ffn"][0, prediction_position].float().unsqueeze(0),
                visual_hpre,
                dim=-1,
            )
            source_dist = _source_from_scores(update_scores, TAU)
            penalty = _penalty(hpre[prediction_position], visual_hpre)
            problem, shift = _problem_from_full_cost(
                source=source_dist,
                target=target_dist,
                full_cost=full_cost,
                penalty=penalty,
                top_k=TOP_K,
            )
            problems[str(target_index)].append(problem)
            shifts[str(target_index)].append(shift)
            f_log[target_index].append(float(sar[target_index].item()))

        captures[layer_index] = None
        del capture, hpre, visual_hpre, full_cost, writes, am_norm
        del norm_probability, _raw_probability, log_support, sar

    solved = _solve_exact_emd_problem_series(problems)
    vectors = []
    for target_index in range(len(targets)):
        risk = np.asarray(
            [
                value + shift
                for value, shift in zip(
                    solved[str(target_index)], shifts[str(target_index)]
                )
            ],
            dtype=np.float32,
        )
        sar = np.asarray(f_log[target_index], dtype=np.float32)
        vector = np.concatenate((risk, sar))
        if vector.shape != (2 * expected_layers,) or not np.isfinite(vector).all():
            raise ValueError(f"{model}/{image_id}: invalid Ours vector")
        vectors.append(vector)
    return {
        "object_count": len(targets),
        "feature_dim": int(vectors[0].size),
    }


@torch.inference_mode()
def extract_baseline(
    *,
    wrapper: Any,
    runtime: BaselineRuntime,
    model: str,
    config: dict[str, Any],
    image_id: int,
    label: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    response_ids = [int(value) for value in generation["response_token_ids"]]
    targets = _targets(label, response_ids)
    spans = _spans_from_targets(targets)
    indices = list(dict.fromkeys(int(target["response_index"]) for target in targets))
    target_ids = [response_ids[index] for index in indices]
    image_path = (
        Path(get_dataset_cfg(config)["coco_root"])
        / "val2014"
        / f"COCO_val2014_{image_id:012d}.jpg"
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    outputs = wrapper.extract_token_features_batch(
        image=image,
        response_token_ids=response_ids,
        response_token_indices=indices,
        target_token_ids=target_ids,
        cfg_dgst_t=None,
        prompt=str(config.get("run", {}).get("prompt") or "Describe this image."),
        requirements=runtime.requirements_for(controlled=True, official=False),
    )
    output_by_index = dict(zip(indices, outputs))
    records = runtime.build_image_records(
        image=image,
        image_id=image_id,
        response_token_ids=response_ids,
        spans=spans,
        model_outputs=[
            output_by_index[int(target["response_index"])] for target in targets
        ],
    )
    if len(records) != len(targets):
        raise ValueError(
            f"{model}/{image_id}: expected {len(targets)} records, got {len(records)}"
        )
    payload = records[0]["baselines"][runtime.methods[0]]
    if runtime.methods[0] == "svar":
        raw_shape = list(np.asarray(payload["visual_attention_ratio"]).shape)
        feature_dim = int(np.asarray(payload["vector"]).size)
    elif runtime.methods[0] == "dhcp":
        raw_shape = list(np.asarray(payload["attention"]).shape)
        feature_dim = int(np.asarray(payload["attention"]).size)
    else:
        raw_shape = list(np.asarray(payload["vector"]).shape)
        feature_dim = int(np.asarray(payload["vector"]).size)
    del outputs, records, output_by_index, image
    return {
        "object_count": len(targets),
        "feature_dim": feature_dim,
        "raw_shape": raw_shape,
    }


def _two_output_mlp_parameters(input_dim: int, hidden_dim: int) -> int:
    return (int(input_dim) + 1) * int(hidden_dim) + (int(hidden_dim) + 1) * 2


def _one_output_mlp_parameters(input_dim: int, hidden_dim: int) -> int:
    return (int(input_dim) + 1) * int(hidden_dim) + int(hidden_dim) + 1


def _parameter_metadata(
    *,
    model: str,
    method: str,
    feature_meta: Mapping[str, Any],
    baseline_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if method == "ours":
        input_dim = int(feature_meta["feature_dim"])
        count = _one_output_mlp_parameters(input_dim, 128)
        return {
            "mlp_architecture": f"Linear({input_dim},128)-ReLU-Dropout-Linear(128,1)",
            "mlp_input_dim": input_dim,
            "mlp_parameter_count": count,
            "mlp_parameter_counts_by_seed": {str(seed): count for seed in (43, 44, 45)},
        }
    if method == "svar":
        layers, heads = map(int, feature_meta["raw_shape"])
        cfg = dict(baseline_cfg["svar"])
        selected_layers = int(cfg.get("layer_end", 19)) - int(cfg.get("layer_start", 5))
        if selected_layers <= 0 or selected_layers > layers:
            raise ValueError(f"Invalid SVAR layer slice for {model}")
        input_dim = selected_layers * heads
        hidden = int(cfg.get("hidden_dim", 248))
        count = _two_output_mlp_parameters(input_dim, hidden)
        return {
            "mlp_architecture": f"Linear({input_dim},{hidden})-ReLU-Linear({hidden},2)",
            "mlp_input_dim": input_dim,
            "mlp_parameter_count": count,
            "mlp_parameter_counts_by_seed": {str(seed): count for seed in (43, 44, 45)},
        }
    if method == "dhcp":
        input_dim = int(np.prod(feature_meta["raw_shape"]))
        hidden = int(dict(baseline_cfg["dhcp"]).get("hidden_dim", 128))
        count = _two_output_mlp_parameters(input_dim, hidden)
        return {
            "mlp_architecture": f"Linear({input_dim},{hidden})-ReLU-Linear({hidden},2)",
            "mlp_input_dim": input_dim,
            "mlp_parameter_count": count,
            "mlp_parameter_counts_by_seed": {str(seed): count for seed in (43, 44, 45)},
        }
    if method == "token_grounding":
        input_dim = int(feature_meta["feature_dim"])
        audit = _json(OUTPUT.parent / "COCO4000-512-ENDAC-BASELINES-V2" / "native_training_audit.json")
        widths = {
            str(seed): int(audit[model]["seeds"][str(seed)]["mlp"]["best_params"][
                "hidden_layer_sizes"
            ][0])
            for seed in (43, 44, 45)
        }
        counts = {
            seed: _one_output_mlp_parameters(input_dim, width)
            for seed, width in widths.items()
        }
        unique_counts = sorted(set(counts.values()))
        return {
            "mlp_architecture": "sklearn MLPClassifier; selected hidden width varies by seed",
            "mlp_input_dim": input_dim,
            "mlp_hidden_widths_by_seed": widths,
            "mlp_parameter_count": unique_counts[0] if len(unique_counts) == 1 else None,
            "mlp_parameter_count_min": min(unique_counts),
            "mlp_parameter_count_max": max(unique_counts),
            "mlp_parameter_counts_by_seed": counts,
        }
    raise ValueError(method)


def _cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _benchmark_method(
    *,
    method: str,
    wrapper: Any,
    model: str,
    ours_config: dict[str, Any],
    baseline_config_full: dict[str, Any],
    baseline_cfg: dict[str, Any],
    labels: Mapping[int, dict[str, Any]],
    generations: Mapping[int, dict[str, Any]],
    image_ids: list[int],
    warmup_image_id: int,
    device: torch.device,
    scratch: Path,
) -> dict[str, Any]:
    runtime = None
    if method != "ours":
        runtime = BaselineRuntime(
            wrapper=wrapper,
            methods=(method,),
            baseline_dir=scratch / method,
            config=baseline_cfg,
            device=str(device),
            resume=False,
            inline_dhcp=True,
        )

    def run_one(image_id: int) -> dict[str, Any]:
        common = {
            "wrapper": wrapper,
            "model": model,
            "image_id": image_id,
            "label": labels[image_id],
            "generation": generations[image_id],
        }
        if method == "ours":
            return extract_ours(config=ours_config, **common)
        assert runtime is not None
        return extract_baseline(
            runtime=runtime,
            config=baseline_config_full,
            **common,
        )

    try:
        # Warm-up is deliberately outside both the 50-image timer and peak reset.
        feature_meta = run_one(warmup_image_id)
        _cuda_sync(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        object_counts: list[int] = []
        for offset, image_id in enumerate(image_ids, 1):
            _cuda_sync(device)
            started = time.perf_counter()
            current = run_one(image_id)
            _cuda_sync(device)
            elapsed = time.perf_counter() - started
            if current["feature_dim"] != feature_meta["feature_dim"]:
                raise ValueError(f"{model}/{method}: feature dimension changed")
            times.append(elapsed)
            object_counts.append(int(current["object_count"]))
            del current
            print(
                f"[{model}/{method}] {offset}/{len(image_ids)} "
                f"image={image_id} elapsed={elapsed:.3f}s",
                flush=True,
            )
        total_seconds = float(sum(times))
        total_objects = int(sum(object_counts))
        result = {
            "method": method,
            "display_name": DISPLAY[method],
            "images": len(image_ids),
            "object_tokens": total_objects,
            "objects_per_image": total_objects / len(image_ids),
            "total_seconds": total_seconds,
            "seconds_per_image": total_seconds / len(image_ids),
            "median_seconds_per_image": statistics.median(times),
            "milliseconds_per_object_token": 1000.0 * total_seconds / total_objects,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_gpu_memory_gib": float(
                torch.cuda.max_memory_allocated(device) / (1024**3)
            ),
            "per_image_seconds": times,
            "per_image_object_counts": object_counts,
            "feature_metadata": dict(feature_meta),
        }
        result.update(
            _parameter_metadata(
                model=model,
                method=method,
                feature_meta=feature_meta,
                baseline_cfg=baseline_cfg,
            )
        )
        return result
    finally:
        if runtime is not None:
            runtime.close()
        del runtime
        gc.collect()
        torch.cuda.empty_cache()


def run_model(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device(args.device)
    image_ids, warmup_image_id = benchmark_image_ids(args.num_images)
    source = _source_dir(args.model)
    labels = {int(key): value for key, value in _json(source / "labeling.json").items()}
    generations = {
        int(key): value for key, value in _json(source / "generations.json").items()
    }
    ours_config = load_config(str(OURS_CONFIG))
    baseline_config_full = load_config(str(BASELINE_CONFIG))
    baseline_cfg = baseline_config(baseline_config_full)
    model_cfg = get_extraction_model_cfg(baseline_config_full, args.model)
    wrapper = build_model(args.model, model_cfg, device=str(device))
    wrapper.model.eval().requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / "_runtime_scratch" / args.model
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for method in METHODS:
            results.append(
                _benchmark_method(
                    method=method,
                    wrapper=wrapper,
                    model=args.model,
                    ours_config=ours_config,
                    baseline_config_full=baseline_config_full,
                    baseline_cfg=baseline_cfg,
                    labels=labels,
                    generations=generations,
                    image_ids=image_ids,
                    warmup_image_id=warmup_image_id,
                    device=device,
                    scratch=scratch,
                )
            )
    finally:
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema": "k32-feature-complexity-v1",
        "model": args.model,
        "device": str(device),
        "top_k": TOP_K,
        "tau": TAU,
        "image_ids": image_ids,
        "warmup_image_id": warmup_image_id,
        "timing_scope": (
            "image load + teacher-forced LVLM forward(s) + feature post-processing; "
            "excludes model load, caption generation, serialization, and detector training"
        ),
        "memory_scope": "torch CUDA max_memory_allocated including resident LVLM weights",
        "results": results,
    }
    destination = args.output_dir / f"{args.model}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {destination}", flush=True)


def _parameter_text(row: Mapping[str, Any]) -> str:
    value = row.get("mlp_parameter_count")
    if value is not None:
        return f"{int(value):,}"
    return f"{int(row['mlp_parameter_count_min']):,}--{int(row['mlp_parameter_count_max']):,}"


def summarize(output_dir: Path) -> None:
    payloads = [_json(output_dir / f"{model}.json") for model in MODELS]
    if payloads[0]["image_ids"] != payloads[1]["image_ids"]:
        raise ValueError("The two models were not benchmarked on the same images")
    lines = [
        "# K=32 feature-extraction complexity (50 shared COCO images)",
        "",
        "Timing starts from saved generations and includes image loading, teacher-forced "
        "LVLM forward(s), and all object-feature post-processing. Model loading, caption "
        "generation, serialization, and detector training are excluded. Peak memory is "
        "PyTorch `max_memory_allocated`, including resident LVLM weights. Feature "
        "dimension is the actual input width of each native MLP.",
        "",
        "| Backbone | Method | s / image | ms / object token | Peak GPU (GiB) | Feature dim. | MLP params | Objects |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Feature-extraction complexity on 50 shared COCO images. All methods start from generated captions; $K=32$ and $\tau=0.03$ for ours. Peak memory includes the LVLM weights, and feature dimension denotes the native MLP input width.}",
        r"  \label{tab:feature-complexity-k32}",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \resizebox{\columnwidth}{!}{%",
        r"    \begin{tabular}{@{}llrrrrr@{}}",
        r"      \toprule",
        "      Backbone & Method & s/image & ms/object & Peak GPU & Feature dim. & MLP params \\\\",
        r"      \midrule",
    ]
    for payload_index, payload in enumerate(payloads):
        backbone = "LLaVA-1.5-7B" if payload["model"] == "llava_1_5_7b" else "Qwen3-VL-8B"
        for method_index, row in enumerate(payload["results"]):
            params = _parameter_text(row)
            lines.append(
                f"| {backbone} | {row['display_name']} | {row['seconds_per_image']:.3f} | "
                f"{row['milliseconds_per_object_token']:.1f} | {row['peak_gpu_memory_gib']:.2f} | "
                f"{int(row['mlp_input_dim']):,} | {params} | {row['object_tokens']} |"
            )
            latex.append(
                f"      {backbone if method_index == 0 else ''} & {row['display_name']} & "
                f"{row['seconds_per_image']:.3f} & {row['milliseconds_per_object_token']:.1f} & "
                f"{row['peak_gpu_memory_gib']:.2f} GiB & "
                f"{format(int(row['mlp_input_dim']), ',').replace(',', r'{,}')} & "
                f"{params.replace(',', r'{,}')} \\\\"
            )
        if payload_index + 1 < len(payloads):
            latex.append(r"      \midrule")
    lines.extend(
        [
            "",
            "TokenGrounding-MLP reports a range when its validation-selected hidden width "
            "differs across seeds 43/44/45. Object-token counts are identical across methods "
            "within each backbone.",
            "",
        ]
    )
    latex.extend(
        [
            r"      \bottomrule",
            r"    \end{tabular}%",
            r"  }",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "complexity_table.md").write_text("\n".join(lines))
    (output_dir / "complexity_table.tex").write_text("\n".join(latex))
    print(f"Wrote {output_dir / 'complexity_table.md'}")
    print(f"Wrote {output_dir / 'complexity_table.tex'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.summarize:
        summarize(args.output_dir)
        return
    if args.model is None:
        parser.error("--model is required unless --summarize is used")
    if args.num_images <= 0:
        parser.error("--num-images must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_model(args)


if __name__ == "__main__":
    main()
