#!/usr/bin/env python3
"""Fuse all weighted-cost risks with selected ENDAC attention and I features."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.train_iclr_endac_attention_i_fusion import (  # noqa: E402
    MODELS,
    _key,
    _probe_config,
)
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


OUT = ROOT / "outputs/weighted_cost_attention_i_fusion"
GF_RUN = GF_ROOT / "outputs/coco4000_512_endac_attention_i_fusion_811"
GF_SOURCE = GF_ROOT / "outputs/coco4000_512_endac_811"
RUNS = {
    "wcost_all": "COCO4000-512-WCOST-ALL",
    "wcost_rawattn": "COCO4000-512-WCOST-RAWATTN",
}
SOURCE_MODES = ("softmax", "pam", "cam")
ALL_METHODS = ("hpre_raw_logit_gauss", "hpre_softmax_prob_gauss")
WEIGHTS = (2, 4, 6, 8)
SEEDS = (43, 44, 45)
EXTRAS = {
    "raw_semantic_only": ("all_visual", "raw", 0),
    "norm_semantic_only": ("all_visual", "norm", 0),
    "raw_semantic_attention": ("all_visual", "raw", 3),
    "norm_semantic_attention": ("all_visual", "norm", 3),
    "I": ("I", None, None),
}
EXPECTED_RISKS = 36
EXPECTED_CASES = EXPECTED_RISKS * len(EXTRAS) * len(SEEDS)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2) + "\n")
    os.replace(temporary, path)


def _generation_rows(path: Path) -> dict:
    return json.loads(path.read_text())


def _risk_specs() -> list[dict]:
    specs = []
    for run_key, run_name in RUNS.items():
        methods = ALL_METHODS if run_key == "wcost_all" else ("raw_attention",)
        for source in SOURCE_MODES:
            for method in methods:
                for weight in WEIGHTS:
                    field = (
                        f"dgst_t_{method}_risk_linear_matched_state_target_"
                        f"w0{weight}_hpre_source_{source}_per_layer"
                    )
                    specs.append({
                        "run": run_key,
                        "run_name": run_name,
                        "source": source,
                        "method": method,
                        "weight": f"w0{weight}",
                        "field": field,
                        "name": f"{run_key}__{source}__{method}__w0{weight}",
                    })
    if len(specs) != EXPECTED_RISKS or len({s["name"] for s in specs}) != EXPECTED_RISKS:
        raise AssertionError("Weighted-cost risk specification is incomplete")
    return specs


def _load_model(model: str) -> tuple[dict, dict]:
    run_dirs = {key: ROOT / "outputs" / model / name for key, name in RUNS.items()}
    risk_rows = {key: load_pkl(str(path / "features.pkl")) for key, path in run_dirs.items()}
    if any(not rows for rows in risk_rows.values()):
        raise ValueError(f"{model}: empty weighted-cost feature file")
    by_run = {}
    for run_key, rows in risk_rows.items():
        mapping = {_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"{model}/{run_key}: duplicate target key")
        by_run[run_key] = mapping
    reference_keys = set(by_run["wcost_all"])
    if set(by_run["wcost_rawattn"]) != reference_keys:
        raise ValueError(f"{model}: WCOST-ALL and WCOST-RAWATTN target sets differ")

    gf_path = GF_RUN / model / "features.pt"
    gf = torch.load(gf_path, map_location="cpu", weights_only=False, mmap=True)
    source_path = GF_SOURCE / model / "features.pkl"
    gf_source = load_pkl(str(source_path))
    if len(gf_source) != len(gf["sample_id"]):
        raise ValueError(f"{model}: GF source/cache length differs")
    if [row["sample_id"] for row in gf_source] != list(gf["sample_id"]):
        raise ValueError(f"{model}: GF source/cache order differs")
    gf_keys = [_key(row) for row in gf_source]
    if len(set(gf_keys)) != len(gf_keys):
        raise ValueError(f"{model}: duplicate GF target key")
    if set(gf_keys) != reference_keys:
        missing_risk = len(set(gf_keys) - reference_keys)
        missing_gf = len(reference_keys - set(gf_keys))
        raise ValueError(
            f"{model}: target sets differ (missing risk={missing_risk}, missing GF={missing_gf})"
        )

    generations = {
        "gf": _generation_rows(source_path.parent / "generations.json"),
        **{
            key: _generation_rows(path / "generations.json")
            for key, path in run_dirs.items()
        },
    }
    if not (generations["gf"] == generations["wcost_all"] == generations["wcost_rawattn"]):
        raise ValueError(f"{model}: full generation JSON differs across feature sources")

    order = np.asarray([by_run["wcost_all"][_key(row)] is not None for row in gf_source])
    if not order.all():
        raise AssertionError("Internal alignment failure")
    aligned = {
        key: [mapping[target_key] for target_key in gf_keys]
        for key, mapping in by_run.items()
    }
    labels = np.asarray(gf["labels"], dtype=np.int64)
    image_ids = np.asarray(gf["image_id"], dtype=np.int64)
    for index, source_row in enumerate(gf_source):
        for run_key in RUNS:
            risk_row = aligned[run_key][index]
            if int(risk_row["label"]) != int(source_row["label"]):
                raise ValueError(f"{model}/{run_key}: label mismatch at {_key(source_row)}")
            if str(risk_row["token_str"]) != str(source_row["canonical_object"]):
                raise ValueError(f"{model}/{run_key}: object mismatch at {_key(source_row)}")
        if int(source_row["label"]) != int(labels[index]):
            raise ValueError(f"{model}: GF cached label mismatch at {_key(source_row)}")
        if int(source_row["image_id"]) != int(image_ids[index]):
            raise ValueError(f"{model}: GF cached image mismatch at {_key(source_row)}")

    splits = {
        key: json.loads((path / "image_splits.json").read_text())
        for key, path in run_dirs.items()
    }
    if splits["wcost_all"] != splits["wcost_rawattn"]:
        raise ValueError(f"{model}: weighted-cost image splits differ")
    train_ids = {int(value) for value in splits["wcost_all"]["train"]}
    test_ids = {int(value) for value in splits["wcost_all"]["test"]}
    if train_ids & test_ids or len(train_ids) != 3200 or len(test_ids) != 800:
        raise ValueError(f"{model}: expected strict 3200/800 image split")
    train_mask = np.isin(image_ids, list(train_ids))
    test_mask = np.isin(image_ids, list(test_ids))
    if not np.all(train_mask | test_mask):
        raise ValueError(f"{model}: GF rows outside weighted-cost split")
    if set(np.unique(labels[train_mask])) != {0, 1} or set(np.unique(labels[test_mask])) != {0, 1}:
        raise ValueError(f"{model}: train/test split must contain both classes")

    extras = {}
    for name, (scope, variant, feature_index) in EXTRAS.items():
        if scope == "I":
            matrix = np.asarray(gf["I"], dtype=np.float32)
        else:
            matrix = np.asarray(gf[scope][variant], dtype=np.float32)[:, :, feature_index]
        if not np.isfinite(matrix).all():
            raise ValueError(f"{model}: nonfinite extra feature {name}")
        extras[name] = matrix

    risks = {}
    specs = _risk_specs()
    for spec in specs:
        rows = aligned[spec["run"]]
        if spec["field"] not in rows[0]:
            raise KeyError(f"{model}: missing {spec['field']}")
        matrix = np.stack([np.asarray(row[spec["field"]], dtype=np.float32) for row in rows])
        if not np.isfinite(matrix).all():
            raise ValueError(f"{model}: nonfinite risk {spec['name']}")
        if matrix.shape != extras["I"].shape:
            raise ValueError(
                f"{model}: layer shape differs for {spec['name']}: "
                f"{matrix.shape} vs {extras['I'].shape}"
            )
        risks[spec["name"]] = matrix

    data = {
        "image_id": image_ids,
        "label": labels,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "risks": risks,
        "extras": extras,
    }
    audit = {
        "model": model,
        "schema": "weighted-cost-attention-i-fusion-v1",
        "join_key": "(image_id, response_token_idx, target_token_id)",
        "full_generation_json_identical": True,
        "labels_objects_and_gf_cache_checked": True,
        "matched_rows": len(gf_keys),
        "matched_train_rows": int(train_mask.sum()),
        "matched_test_rows": int(test_mask.sum()),
        "matched_train_images": len(set(image_ids[train_mask].tolist())),
        "matched_test_images": len(set(image_ids[test_mask].tolist())),
        "layers": int(extras["I"].shape[1]),
        "risk_count": len(risks),
        "extra_features": list(EXTRAS),
        "fusion_count": len(risks) * len(extras),
        "seed_count": len(SEEDS),
        "training_case_count": len(risks) * len(extras) * len(SEEDS),
        "split": "weighted-cost strict image-level 3200/800; GF split ignored",
        "gf_source": str(gf_path),
        "weighted_cost_runs": RUNS,
    }
    if audit["risk_count"] != EXPECTED_RISKS or audit["training_case_count"] != EXPECTED_CASES:
        raise AssertionError(f"{model}: unexpected case count")
    return data, audit


def _baseline_path(model: str, spec: dict, seed: int) -> Path:
    return (
        ROOT / "outputs" / model / spec["run_name"] / "weighted_cost_probe"
        / spec["source"] / f"{spec['method']}_{spec['weight']}" / f"seed{seed}"
        / "metrics.json"
    )


def _report(metrics: dict, threshold: str) -> dict:
    return metrics["threshold_reports"][threshold]["test_metrics"]


def _summarize(model: str, model_out: Path, audit: dict) -> None:
    rows = []
    for spec in _risk_specs():
        baselines = [json.loads(_baseline_path(model, spec, seed).read_text()) for seed in SEEDS]
        for extra in EXTRAS:
            cases = [
                json.loads(
                    (model_out / spec["name"] / extra / f"seed{seed}" / "metrics.json").read_text()
                )
                for seed in SEEDS
            ]
            for threshold in ("fixed_0.5", "train_f1"):
                fused_reports = [_report(case, threshold) for case in cases]
                base_reports = [_report(case, threshold) for case in baselines]
                row = {
                    "model": model,
                    "threshold": threshold,
                    "run": spec["run"],
                    "source": spec["source"],
                    "method": spec["method"],
                    "weight": spec["weight"],
                    "risk": spec["name"],
                    "added_feature": extra,
                }
                metrics_to_collect = {
                    "real_f1": ("real_positive", "f1"),
                    "real_auc": ("real_positive", "auc"),
                    "hall_aupr": ("hallucination_positive", "aupr"),
                }
                for metric_name, (side, key) in metrics_to_collect.items():
                    fused = [report[side][key] for report in fused_reports]
                    baseline = [report[side][key] for report in base_reports]
                    row[f"{metric_name}_mean"] = float(np.mean(fused))
                    row[f"{metric_name}_std"] = float(np.std(fused))
                    row[f"baseline_{metric_name}_mean"] = float(np.mean(baseline))
                    row[f"delta_{metric_name}_mean"] = float(np.mean(np.asarray(fused) - baseline))
                rows.append(row)

    csv_path = model_out / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {model}: weighted-cost risk + selected ENDAC attention / I",
        "",
        f"Exact aligned mentions: {audit['matched_rows']} "
        f"(train {audit['matched_train_rows']}, test {audit['matched_test_rows']}).",
        "All generation JSON files are identical. The GF split is ignored; the weighted-cost strict "
        "3200/800 image split is used.",
        "Each fused input is [risk[L], added feature[L]]. Probe: one hidden layer of width 128, "
        "ReLU, dropout 0.3, training-only column z-score, no BatchNorm, seeds 43–45.",
        "Deltas use the already completed risk-only probe for the same run/source/method/weight/seed.",
        "",
        "| Threshold | Run | Source | Method | w | Added feature | Real F1 | Real AUC | Δ AUC | Hall AUPR | Δ Hall AUPR |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {row['run']} | {row['source']} | {row['method']} | "
            f"{row['weight']} | {row['added_feature']} | "
            f"{row['real_f1_mean']:.4f}±{row['real_f1_std']:.4f} | "
            f"{row['real_auc_mean']:.4f}±{row['real_auc_std']:.4f} | "
            f"{row['delta_real_auc_mean']:+.4f} | "
            f"{row['hall_aupr_mean']:.4f}±{row['hall_aupr_std']:.4f} | "
            f"{row['delta_hall_aupr_mean']:+.4f} |"
        )
    (model_out / "summary.md").write_text("\n".join(lines) + "\n")

    best = []
    for threshold in ("fixed_0.5", "train_f1"):
        selected = [row for row in rows if row["threshold"] == threshold]
        for run_key in RUNS:
            subset = [row for row in selected if row["run"] == run_key]
            best.append(max(subset, key=lambda row: row["hall_aupr_mean"]))
    best_lines = [
        f"# {model}: best weighted-cost fusion settings",
        "",
        "| Threshold | Run | Source | Method | w | Added feature | Hall AUPR | Δ Hall AUPR | Real AUC | Δ AUC |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in best:
        best_lines.append(
            f"| {row['threshold']} | {row['run']} | {row['source']} | {row['method']} | "
            f"{row['weight']} | {row['added_feature']} | {row['hall_aupr_mean']:.4f} | "
            f"{row['delta_hall_aupr_mean']:+.4f} | {row['real_auc_mean']:.4f} | "
            f"{row['delta_real_auc_mean']:+.4f} |"
        )
    (model_out / "best.md").write_text("\n".join(best_lines) + "\n")


def train(model: str, device: str, output_root: Path, dry_run: bool = False) -> None:
    torch.set_num_threads(1)
    data, audit = _load_model(model)
    print(json.dumps(audit, indent=2), flush=True)
    if dry_run:
        return
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    audit_path = model_out / "alignment.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError(f"{model}: existing alignment audit differs")
    _atomic_json(audit_path, audit)

    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    probe_example = replace(
        _probe_config(root_config, SEEDS[0]),
        hidden_sizes=(128,),
        batch_norm=False,
        standardize_input=True,
    )
    protocol = {
        **audit,
        "architecture": "one hidden layer, width 128",
        "probe": _json_ready(asdict(probe_example)),
        "seeds": list(SEEDS),
        "input": "concatenate risk[L] and one selected ENDAC feature[L]",
        "attention_features": {
            "raw_semantic_only": "all_visual/raw/semantic_only",
            "norm_semantic_only": "all_visual/norm/semantic_only",
            "raw_semantic_attention": "all_visual/raw/semantic_attention",
            "norm_semantic_attention": "all_visual/norm/semantic_attention",
            "I": "raw visual sum of attention-write norms",
        },
    }
    _atomic_json(model_out / "protocol.json", protocol)

    completed = sum(1 for _ in model_out.glob("*/*/seed*/metrics.json"))
    _atomic_json(model_out / "progress.json", {
        "model": model, "stage": "training", "completed": completed, "total": EXPECTED_CASES,
    })
    labels = data["label"]
    for spec in _risk_specs():
        risk = data["risks"][spec["name"]]
        for extra_name, extra in data["extras"].items():
            features = np.concatenate((risk, extra), axis=1).astype(np.float32, copy=False)
            for seed in SEEDS:
                case_out = model_out / spec["name"] / extra_name / f"seed{seed}"
                metrics_path = case_out / "metrics.json"
                if metrics_path.exists():
                    print("SKIP", model, spec["name"], extra_name, seed, flush=True)
                    continue
                cfg = replace(
                    _probe_config(root_config, seed),
                    hidden_sizes=(128,),
                    batch_norm=False,
                    standardize_input=True,
                )
                metrics = train_and_evaluate_probe(
                    X_train=features[data["train_mask"]],
                    y_train=labels[data["train_mask"]],
                    X_val=np.empty((0, features.shape[1]), dtype=np.float32),
                    y_val=np.empty((0,), dtype=np.int64),
                    X_test=features[data["test_mask"]],
                    y_test=labels[data["test_mask"]],
                    config=cfg,
                    device=torch.device(device),
                    output_dir=str(case_out),
                )
                metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
                completed += 1
                _atomic_json(model_out / "progress.json", {
                    "model": model,
                    "stage": "training",
                    "completed": completed,
                    "total": EXPECTED_CASES,
                    "risk": spec["name"],
                    "added_feature": extra_name,
                    "seed": seed,
                })
                print("DONE", model, spec["name"], extra_name, seed, flush=True)
    _summarize(model, model_out, audit)
    _atomic_json(model_out / "progress.json", {
        "model": model, "stage": "complete", "completed": EXPECTED_CASES, "total": EXPECTED_CASES,
    })
    print("SUMMARY", model_out / "summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.model, args.device, args.output_root.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
