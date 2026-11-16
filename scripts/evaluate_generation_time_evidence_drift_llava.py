#!/usr/bin/env python3
"""Evaluate paired LLaVA captions with the repository's COCO CHAIR evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_json, atomic_write_jsonl, load_jsonl


def load_chair(coco_root: Path, output_root: Path, cohort_ids: list[int],
               full_cache_path: Path | None):
    # Register the same module name as the original runner for pickle compatibility.
    module_name = "coco_chair"
    source = ROOT / "coco-labeling/coco_chair.py"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    annotations = coco_root / "annotations"
    if not annotations.is_dir():
        annotations = coco_root
    instances = annotations / "instances_val2014.json"
    captions = annotations / "captions_val2014.json"
    if not instances.is_file() or not captions.is_file():
        raise FileNotFoundError(f"Need COCO val2014 instances and captions under {annotations}")
    chair_class = sys.modules[module_name].CocoChairEvaluator
    if full_cache_path is not None:
        evaluator = chair_class.from_cache(
            instances_file=instances, captions_file=captions, cache_path=full_cache_path
        )
        return evaluator, full_cache_path

    fingerprints = {}
    for annotation in (instances, captions):
        info = annotation.stat()
        fingerprints[str(annotation.resolve())] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    scope = {
        "image_ids": sorted(set(cohort_ids)),
        "annotations": fingerprints,
        "chair_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    cache_id = hashlib.sha256(json.dumps(scope, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = output_root / f"chair_ground_truth.{cache_id[:20]}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else None
    if cache is not None and cache["scope"] != scope:
        raise ValueError(f"Unexpected CHAIR ground-truth cache provenance: {cache_path}")
    selected = set(scope["image_ids"])

    class CohortChair(chair_class):
        def _load_instances_payload(self):
            payload = super()._load_instances_payload()
            return {
                "categories": payload["categories"],
                "images": [row for row in payload["images"] if int(row["id"]) in selected],
                "annotations": [row for row in payload["annotations"] if int(row["image_id"]) in selected],
            }

        def _load_captions_payload(self):
            payload = super()._load_captions_payload()
            return {
                "annotations": [row for row in payload["annotations"] if int(row["image_id"]) in selected],
            }

        def get_annotations(self):
            if cache is None:
                super().get_annotations()
            else:
                self.val_image_ids = scope["image_ids"]
                self.imid_to_objects = {
                    int(image_id): set(objects) for image_id, objects in cache["objects"].items()
                }

    evaluator = CohortChair(instances_file=instances, captions_file=captions)
    if cache is None:
        missing = selected - set(evaluator.val_image_ids)
        if missing:
            raise ValueError(f"{len(missing)} cohort images absent from COCO annotations")
        atomic_write_json(cache_path, {
            "scope": scope,
            "objects": {str(i): sorted(evaluator.get_ground_truth_objects(i)) for i in selected},
        })
    return evaluator, cache_path


def load_variant(folder: Path, variant: str, cohort_ids: list[int], selected_ids: list[int]) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", variant):
        raise ValueError(f"Invalid variant name {variant!r}")
    allowed = set(cohort_ids)
    by_id = {}
    for shard in sorted(folder.glob(f"{variant}.shard*.jsonl")):
        for row in load_jsonl(shard):
            image_id = int(row["image_id"])
            if image_id in by_id or image_id not in allowed:
                raise ValueError(f"{shard}: duplicate or unexpected image {image_id}")
            by_id[image_id] = row
    missing = set(selected_ids) - by_id.keys()
    if missing:
        raise ValueError(f"{variant}: {len(missing)} missing captions, first IDs {sorted(missing)[:8]}")
    return {image_id: by_id[image_id] for image_id in selected_ids}


def _seconds(row: dict, field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"image {row['image_id']}: {field} must be finite and nonnegative")
    return value


def _score_caption(chair, row: dict, cache: dict) -> dict:
    image_id = int(row["image_id"])
    caption = row["generated_text"]
    key = (image_id, caption)
    if key not in cache:
        cache[key] = chair.compute_chair_token(image_id, caption)
    scored = cache[key]
    result = {
        "chair_s": scored["metrics"]["CHAIRs"],
        "chair_i": scored["metrics"]["CHAIRi"],
        "hallucinated_mentions": len(scored["mscoco_hallucinated_words"]),
        "real_mentions": len(scored["mscoco_real_words"]),
        "object_mentions": len(scored["object_mentions"]),
        "hallucinated_objects": scored["mscoco_hallucinated_words"],
        "object_recall": scored["metrics"]["Recall"],
        "object_precision": scored["metrics"]["Precision"],
        "object_f1": scored["metrics"]["F1"],
        "words": len(scored["words"]),
        "tokens": len(row["response_token_ids"]),
        "generation_seconds": _seconds(row, "seconds"),
    }
    if row.get("prefill_seconds") is not None:
        result["shared_prefill_seconds"] = _seconds(row, "prefill_seconds")
    if "changes" in row:
        result["changes"] = row["changes"]
    if row.get("reused_from"):
        result["reused_from"] = row["reused_from"]
    if row.get("ffn_statistics"):
        result["ffn_statistics"] = row["ffn_statistics"]
    if row.get("sar_ems_statistics"):
        result["sar_ems_statistics"] = row["sar_ems_statistics"]
    if row.get("attention_transfer_statistics"):
        result["attention_transfer_statistics"] = row["attention_transfer_statistics"]
    return result


def summarize(rows: list[dict]) -> dict:
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("hallucinated_mentions", "real_mentions", "object_mentions")
    }
    timings = [row["generation_seconds"] for row in rows]
    prefill = [row["shared_prefill_seconds"] for row in rows if "shared_prefill_seconds" in row]
    hallucinated_images = sum(row["chair_s"] for row in rows)
    token_count = sum(row["tokens"] for row in rows)
    ffn = [row["ffn_statistics"] for row in rows if row.get("ffn_statistics")]
    accepted = sum(item["accepted_updates"] for item in ffn)
    sar_ems = [row["sar_ems_statistics"] for row in rows if row.get("sar_ems_statistics")]
    guided = sum(item["guided_steps"] for item in sar_ems)
    matched = sum(item["target_matches_generated"] for item in sar_ems)
    transfers = [row["attention_transfer_statistics"] for row in rows if row.get("attention_transfer_statistics")]
    transfer_steps = sum(item["layer_steps"] for item in transfers)
    return {
        **totals,
        "images": len(rows),
        "reused_base_images": sum(bool(row.get("reused_from")) for row in rows),
        "hallucinated_images": hallucinated_images,
        "images_without_object_mentions": sum(row["object_mentions"] == 0 for row in rows),
        "chair_s": hallucinated_images / len(rows),
        "chair_i": totals["hallucinated_mentions"] / max(1, totals["object_mentions"]),
        "object_recall": float(np.mean([row["object_recall"] for row in rows])),
        "object_precision": float(np.mean([row["object_precision"] for row in rows])),
        "object_f1": float(np.mean([row["object_f1"] for row in rows])),
        "words_per_image": float(np.mean([row["words"] for row in rows])),
        "tokens_per_image": float(np.mean([row["tokens"] for row in rows])),
        "generation_seconds_sum": sum(timings),
        "generation_ms_per_token": 1000 * sum(timings) / max(1, token_count),
        "generation_seconds_per_image": float(np.mean(timings)),
        "generation_seconds_median": float(np.median(timings)),
        "timed_prefill_images": len(prefill),
        "ffn_statistics": {
            "draft_noun_steps": sum(item["draft_noun_steps"] for item in ffn),
            "accepted_updates": accepted,
            "mean_drift_reduction_per_update": (
                sum(item["accepted_updates"] * item["mean_drift_reduction_per_update"]
                    for item in ffn) / accepted if accepted else None
            ),
        } if ffn else None,
        "sar_ems_statistics": {
            "guided_steps": guided,
            "target_matches_generated": matched,
            "target_match_rate": matched / guided if guided else None,
        } if sar_ems else None,
        "attention_transfer_statistics": {
            "layer_steps": transfer_steps,
            "mean_moved_mass": sum(item["layer_steps"] * item["mean_moved_mass"] for item in transfers)
            / transfer_steps,
        } if transfer_steps else None,
        "shared_prefill_seconds_per_image": float(np.mean(prefill)) if prefill else None,
        "seconds_per_image_including_prefill": (
            (sum(timings) + sum(prefill)) / len(rows) if len(prefill) == len(rows) else None
        ),
    }


def _bootstrap_metrics(rows: list[dict], draws: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in (
            "chair_s", "hallucinated_mentions", "object_mentions", "object_recall", "object_f1"
        )
    }
    return {
        "chair_s": arrays["chair_s"][draws].mean(axis=1),
        "chair_i": (
            arrays["hallucinated_mentions"][draws].sum(axis=1)
            / np.maximum(1, arrays["object_mentions"][draws].sum(axis=1))
        ),
        "object_recall": arrays["object_recall"][draws].mean(axis=1),
        "object_f1": arrays["object_f1"][draws].mean(axis=1),
    }


def evaluate(output_root: Path, cohort: str, variants: list[str], coco_root: Path,
             limit: int | None, bootstrap_draws: int, cache_path: Path | None,
             image_ids: Path | None = None) -> Path:
    if len(set(variants)) != len(variants):
        raise ValueError("Each variant must be listed once")
    manifest = json.loads((output_root / "cohort.json").read_text(encoding="utf-8"))
    cohort_ids = [int(value) for value in manifest[cohort]]
    if not cohort_ids or len(set(cohort_ids)) != len(cohort_ids):
        raise ValueError(f"{cohort}: empty cohort or duplicate image IDs")
    if limit is not None and not 1 <= limit <= len(cohort_ids):
        raise ValueError(f"Limit must be within [1, {len(cohort_ids)}]")
    if bootstrap_draws < 0:
        raise ValueError("Bootstrap draws must be nonnegative")
    ids = cohort_ids[:limit]
    subset_tag = ""
    if image_ids is not None:
        selected = set(map(int, json.loads(image_ids.read_text(encoding="utf-8"))))
        if not selected or not selected.issubset(ids):
            raise ValueError("The supplied image IDs must form a nonempty subset of the selected cohort")
        ids = [image_id for image_id in ids if image_id in selected]
        subset_tag = ".subset" + hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest()[:12]
    folder = output_root / cohort
    captions = {
        variant: load_variant(folder, variant, cohort_ids, ids) for variant in variants
    }
    ground_truth_ids = sorted(set(ids) | {
        int(i) for part in ("pilot", "confirmation") for i in manifest.get(part, [])
    })
    chair, chair_path = load_chair(coco_root, output_root, ground_truth_ids, cache_path)
    score_cache = {}
    scores = {
        variant: [_score_caption(chair, captions[variant][i], score_cache) for i in ids]
        for variant in variants
    }
    metrics = {variant: summarize(scores[variant]) for variant in variants}

    bootstrap = {}
    if bootstrap_draws:
        draws = np.random.default_rng(43).integers(0, len(ids), size=(bootstrap_draws, len(ids)))
        bootstrap = {variant: _bootstrap_metrics(scores[variant], draws) for variant in variants}
    for variant in variants:
        paired = {}
        for reference in variants:
            if reference == variant:
                continue
            base = metrics[reference]
            current = metrics[variant]
            pairs = list(zip(scores[reference], scores[variant], strict=True))
            paired[reference] = {
                "changed_token_sequences": sum(
                    captions[reference][i]["response_token_ids"]
                    != captions[variant][i]["response_token_ids"] for i in ids
                ),
                "fixed_hallucinated_images": sum(a["chair_s"] > b["chair_s"] for a, b in pairs),
                "introduced_hallucinated_images": sum(a["chair_s"] < b["chair_s"] for a, b in pairs),
                "delta": {
                    key: current[key] - base[key]
                    for key in ("chair_s", "chair_i", "object_recall", "object_f1")
                },
            }
            if bootstrap_draws:
                paired[reference]["delta_ci95"] = {
                    key: np.quantile(values - bootstrap[reference][key], [0.025, 0.975]).tolist()
                    for key, values in bootstrap[variant].items()
                }
        metrics[variant]["paired_against"] = paired

    variant_tag = "-".join(sorted(variants))
    limit_tag = (f".first{limit}" if limit is not None else "") + subset_tag
    metrics_path = folder / f"metrics{limit_tag}.{variant_tag}.json"
    per_image_path = folder / f"chair_per_image{limit_tag}.{variant_tag}.jsonl"
    atomic_write_jsonl(per_image_path, (
        {"image_id": image_id, "methods": {variant: scores[variant][offset] for variant in variants}}
        for offset, image_id in enumerate(ids)
    ))
    atomic_write_json(metrics_path, {
        "cohort": cohort,
        "images": len(ids),
        "image_ids_sha256": hashlib.sha256(json.dumps(ids).encode("ascii")).hexdigest(),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": 43 if bootstrap_draws else None,
        "object_metric_aggregation": "Mean of per-image scores. CHAIRi uses pooled mention counts.",
        "timing": "Generation time excludes the optional shared image prefill; end-to-end estimates include the prefill once.",
        "methods": metrics,
        "chair_cache": str(chair_path),
        "per_image": str(per_image_path),
    })
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohort", choices=("pilot", "confirmation", "rebuttal_holdout",
                                            "rebuttal_replication", "mitigation_external",
                                            "mitigation_reserve"), required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-ids", type=Path, help="JSON list restricting the evaluated cohort")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--chair-cache", type=Path)
    args = parser.parse_args()
    result = evaluate(args.output_root, args.cohort, args.variants, args.coco_root,
                      args.limit, args.bootstrap_draws, args.chair_cache, args.image_ids)
    print(result, flush=True)


if __name__ == "__main__":
    main()
