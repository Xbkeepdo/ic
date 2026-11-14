#!/usr/bin/env python3
"""Prepare and train the eight targeted Log-S mechanism ablations on ENDAC 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.run_endac_baselines_811_dual_threshold import (  # noqa: E402
    atomic_json,
    dual_threshold_report,
    predict_real_probability,
    resolve_device,
    set_seed,
)
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


MODELS = (
    "llava_1_5_7b",
    "qwen2_5_vl_7b",
    "internvl_2_5_8b",
    "qwen3_vl_8b",
)
MODEL_NAMES = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}
SEEDS = (43, 44, 45)
TOP_K = 32
BASE_RISK = ""
FULL_EMS = ""
ATTENTION_TARGET_EMS = ""
GEOMETRY_OT = ""
UNION_JS = ""
FFN_INPUT_EMS = ""
BASE_ROOT = ROOT / "outputs/log_s_hpre_k_tau_ablation"
NEW_ROOT = Path()
I_ROOT = GF_ROOT / "outputs/coco4000_512_endac_attention_i_fusion_811"
SPLIT_PATH = GF_ROOT / "outputs/coco4000_512_endac_811/image_splits.json"
OUT = Path()
CASES = (
    "sar",
    "ems",
    "sar_ems",
    "i_ems",
    "sar_ems_attention_target",
    "sar_ot_geometry",
    "sar_union_js",
    "sar_ems_ffn_input_source",
)
DISPLAY = {
    "sar": "SAR only",
    "ems": "EMS only",
    "sar_ems": "Full: SAR + EMS",
    "i_ems": "w/o SAR semantic recalibration",
    "sar_ems_attention_target": "w/o target semantics in EMS",
    "sar_ot_geometry": "w/o state calibration",
    "sar_union_js": "Union-region JS divergence",
    "sar_ems_ffn_input_source": "w/o FFN update",
}
METRICS = (
    "accuracy_real_threshold",
    "accuracy_hall_threshold",
    "auroc",
    "real_f1",
    "real_recall",
    "real_aupr",
    "hall_f1",
    "hall_recall",
    "hall_aupr",
)


def _configure_top_k(top_k: int) -> None:
    global TOP_K, BASE_RISK, FULL_EMS, ATTENTION_TARGET_EMS
    global GEOMETRY_OT, UNION_JS, FFN_INPUT_EMS, NEW_ROOT, OUT
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    TOP_K = int(top_k)
    suffix = f"k{TOP_K:03d}"
    BASE_RISK = f"log_s_hpre_{suffix}_tau003"
    FULL_EMS = f"ems_full_{suffix}_tau003"
    ATTENTION_TARGET_EMS = f"ems_attention_target_{suffix}_tau003"
    GEOMETRY_OT = f"ot_geometry_{suffix}_tau003"
    UNION_JS = f"source_target_union_{suffix}_js"
    FFN_INPUT_EMS = f"ems_ffn_input_source_{suffix}_tau003"
    if TOP_K == 32:
        NEW_ROOT = GF_ROOT / "outputs/coco4000_512_log_s_mechanism_ablation_811"
        OUT = ROOT / "outputs/log_s_mechanism_ablation_811"
    else:
        NEW_ROOT = (
            GF_ROOT
            / f"outputs/coco4000_512_log_s_mechanism_ablation_{suffix}_811"
        )
        OUT = ROOT / f"outputs/log_s_mechanism_ablation_{suffix}_811"


_configure_top_k(TOP_K)


def row_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def key_from_arrays(data: Mapping[str, np.ndarray], index: int) -> tuple[int, int, int]:
    return (
        int(data["image_id"][index]),
        int(data["response_token_idx"][index]),
        int(data["target_token_id"][index]),
    )


def assemble_case_matrices(
    *,
    sar: np.ndarray,
    ems: np.ndarray,
    raw_i: np.ndarray,
    attention_target_ems: np.ndarray,
    geometry_ot: np.ndarray,
    union_js: np.ndarray,
    ffn_input_ems: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "sar": sar,
        "ems": ems,
        "sar_ems": np.concatenate((sar, ems), axis=1),
        "i_ems": np.concatenate((raw_i, ems), axis=1),
        "sar_ems_attention_target": np.concatenate(
            (sar, attention_target_ems), axis=1
        ),
        "sar_ot_geometry": np.concatenate((sar, geometry_ot), axis=1),
        "sar_union_js": np.concatenate((sar, union_js), axis=1),
        "sar_ems_ffn_input_source": np.concatenate((sar, ffn_input_ems), axis=1),
    }
    if tuple(arrays) != CASES:
        raise AssertionError("Mechanism-ablation case order drifted")
    shapes = {name: value.shape for name, value in arrays.items()}
    if any(value.ndim != 2 for value in arrays.values()):
        raise ValueError(f"All case matrices must be two-dimensional: {shapes}")
    if len({value.shape[0] for value in arrays.values()}) != 1:
        raise ValueError(f"Case row counts differ: {shapes}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Mechanism-ablation cases contain non-finite values")
    return {name: value.astype(np.float32, copy=False) for name, value in arrays.items()}


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def prepare(model: str, model_out: Path) -> dict[str, np.ndarray]:
    cache = model_out / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return {name: np.asarray(saved[name]) for name in saved.files}

    base_path = BASE_ROOT / model / "compact_features.npz"
    new_path = NEW_ROOT / model / "features.pkl"
    i_path = I_ROOT / model / "features.pt"
    missing = [str(path) for path in (base_path, new_path, i_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing mechanism-ablation sources: " + ", ".join(missing))
    with np.load(base_path, allow_pickle=False) as saved:
        base = {name: np.asarray(saved[name]) for name in saved.files}
    new_rows = load_pkl(str(new_path))
    i_cache = torch.load(i_path, map_location="cpu", weights_only=False, mmap=True)

    keys = [key_from_arrays(base, index) for index in range(len(base["label"]))]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate keys in base compact cache")
    new_by_key = {row_key(row): row for row in new_rows}
    if len(new_by_key) != len(new_rows) or set(new_by_key) != set(keys):
        raise ValueError(f"{model}: new extraction and base mention keys differ")
    i_sample_ids = [str(value) for value in i_cache["sample_id"]]
    if len(set(i_sample_ids)) != len(i_sample_ids):
        raise ValueError(f"{model}: duplicate sample IDs in I cache")
    i_by_id = {sample_id: index for index, sample_id in enumerate(i_sample_ids)}
    if len(i_by_id) != len(keys):
        raise ValueError(f"{model}: I cache row count differs")

    aligned_rows = [new_by_key[key] for key in keys]
    aligned_sample_ids = [str(row["sample_id"]) for row in aligned_rows]
    if set(aligned_sample_ids) != set(i_by_id):
        raise ValueError(f"{model}: I/new sample ID sets differ")
    for index, row in enumerate(aligned_rows):
        if int(row["label"]) != int(base["label"][index]):
            raise ValueError(f"{model}: label mismatch for {keys[index]}")

    extracted = {
        field: np.stack(
            [np.asarray(row["features"][field], dtype=np.float32) for row in aligned_rows]
        )
        for field in (
            FULL_EMS,
            ATTENTION_TARGET_EMS,
            GEOMETRY_OT,
            UNION_JS,
            FFN_INPUT_EMS,
        )
    }
    ems = np.asarray(base[BASE_RISK], dtype=np.float32)
    sar = np.asarray(base["f_log"], dtype=np.float32)
    raw_i_values = np.asarray(i_cache["I"], dtype=np.float32)
    raw_i = np.stack(
        [raw_i_values[i_by_id[sample_id]] for sample_id in aligned_sample_ids]
    ).astype(np.float32, copy=False)
    expected_shape = ems.shape
    source_shapes = {
        "sar": sar.shape,
        "ems": ems.shape,
        "I": raw_i.shape,
        **{name: value.shape for name, value in extracted.items()},
    }
    if any(shape != expected_shape for shape in source_shapes.values()):
        raise ValueError(f"{model}: source feature shapes differ: {source_shapes}")
    anchor_delta = np.abs(extracted[FULL_EMS].astype(np.float64) - ems.astype(np.float64))
    anchor_max_abs_delta = float(anchor_delta.max(initial=0.0))
    if not np.allclose(extracted[FULL_EMS], ems, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{model}: recomputed full EMS differs from cached EMS; "
            f"max_abs_delta={anchor_max_abs_delta:.8g}"
        )

    cases = assemble_case_matrices(
        sar=sar,
        ems=ems,
        raw_i=raw_i,
        attention_target_ems=extracted[ATTENTION_TARGET_EMS],
        geometry_ot=extracted[GEOMETRY_OT],
        union_js=extracted[UNION_JS],
        ffn_input_ems=extracted[FFN_INPUT_EMS],
    )
    result = {
        **cases,
        "label": np.asarray(base["label"], dtype=np.int8),
        "image_id": np.asarray(base["image_id"], dtype=np.int64),
        "response_token_idx": np.asarray(base["response_token_idx"], dtype=np.int32),
        "target_token_id": np.asarray(base["target_token_id"], dtype=np.int32),
    }
    atomic_npz(cache, **result)
    atomic_json(
        model_out / "alignment.json",
        {
            "schema": "log-s-mechanism-ablation-811-alignment-v1",
            "model": model,
            "samples": len(keys),
            "layers": int(ems.shape[1]),
            "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
            "sample_id_alignment": "exact",
            "label_alignment": "exact",
            "full_ems_anchor_max_abs_delta": anchor_max_abs_delta,
            "full_ems_anchor_tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "input_dimensions": {name: int(value.shape[1]) for name, value in cases.items()},
            "base": str(base_path),
            "new_extraction": str(new_path),
            "i_cache": str(i_path),
        },
    )
    return result


def load_split() -> dict[str, set[int]]:
    payload = json.loads(SPLIT_PATH.read_text())
    split = {
        name: {int(value) for value in payload[name]}
        for name in ("train", "val", "test")
    }
    counts = {name: len(values) for name, values in split.items()}
    if counts != {"train": 3200, "val": 400, "test": 400}:
        raise ValueError(f"Expected shared 3200/400/400 image split, got {counts}")
    return split


def train_case(
    *,
    model: str,
    case: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    split: dict[str, set[int]],
    seed: int,
    device: str,
    output_dir: Path,
    split_protocol: str = "shared_image_level_811",
) -> dict[str, Any]:
    masks = {name: np.isin(image_ids, list(ids)) for name, ids in split.items()}
    arrays = {name: (matrix[mask], labels[mask]) for name, mask in masks.items()}
    for name, (_values, split_labels) in arrays.items():
        if set(np.unique(split_labels).tolist()) != {0, 1}:
            raise ValueError(f"{model}/{case}/{name}: both labels are required")
    mean = arrays["train"][0].astype(np.float64).mean(axis=0)
    scale = arrays["train"][0].astype(np.float64).std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = {
        name: ((values.astype(np.float64) - mean) / scale).astype(np.float32)
        for name, (values, _split_labels) in arrays.items()
    }

    set_seed(seed)
    chosen_device = torch.device(device)
    net = DGSTStyleProbe(
        input_dim=matrix.shape[1],
        hidden_sizes=(128,),
        dropout=0.3,
        batch_norm=False,
    ).to(chosen_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(standardized["train"]),
            torch.from_numpy(arrays["train"][1].astype(np.float32)),
        ),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    for epoch in range(1, 101):
        net.train()
        losses = []
        for values, targets in loader:
            values, targets = values.to(chosen_device), targets.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(values).squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _ = predict_real_probability(
            net,
            standardized["val"],
            arrays["val"][1],
            criterion,
            chosen_device,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = float(val_loss), epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in net.state_dict().items()
            }
        else:
            stale += 1
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "is_best": improved,
            }
        )
        if stale >= 10:
            break
    if best_state is None:
        raise RuntimeError(f"{model}/{case}/seed{seed}: no finite checkpoint")
    net.load_state_dict(best_state)
    _, val_real = predict_real_probability(
        net, standardized["val"], arrays["val"][1], criterion, chosen_device
    )
    _, test_real = predict_real_probability(
        net, standardized["test"], arrays["test"][1], criterion, chosen_device
    )
    report = dual_threshold_report(
        val_labels=arrays["val"][1],
        val_hall_scores=1.0 - val_real,
        test_labels=arrays["test"][1],
        test_hall_scores=1.0 - test_real,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "model.pt")
    np.savez(output_dir / "input_scaler.npz", mean=mean, scale=scale)
    np.savez_compressed(
        output_dir / "scores.npz",
        val_label=arrays["val"][1].astype(np.int8),
        val_hall_score=(1.0 - val_real).astype(np.float64),
        test_label=arrays["test"][1].astype(np.int8),
        test_hall_score=(1.0 - test_real).astype(np.float64),
    )
    atomic_json(output_dir / "history.json", history)
    training = {
        "input_dim": int(matrix.shape[1]),
        "hidden_sizes": [128],
        "batch_norm": False,
        "standardize_input": True,
        "dropout": 0.3,
        "batch_size": 256,
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "lr_factor": 0.5,
        "lr_patience": 5,
        "early_stopping_patience": 10,
        "checkpoint_selection": "minimum_validation_loss",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }
    atomic_json(output_dir / "config.json", {"model": model, "case": case, **training})
    result = {
        "model": model,
        "case": case,
        "display": DISPLAY[case],
        "seed": seed,
        "split": split_protocol,
        "image_counts": {name: len(ids) for name, ids in split.items()},
        "token_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "threshold_selection": {
            "real": "maximize validation Real-F1",
            "hall": "maximize validation Hall-F1",
        },
        "thresholds": report["thresholds"],
        "validation": report["validation"],
        "test": report["test"],
        "training": training,
    }
    atomic_json(output_dir / "result.json", result)
    return result


def flatten(result: Mapping[str, Any]) -> dict[str, float]:
    test = result["test"]
    return {
        "accuracy_real_threshold": float(test["accuracy_real_threshold"]),
        "accuracy_hall_threshold": float(test["accuracy_hall_threshold"]),
        "auroc": float(test["auroc"]),
        "real_f1": float(test["real"]["f1"]),
        "real_recall": float(test["real"]["recall"]),
        "real_aupr": float(test["real"]["aupr"]),
        "hall_f1": float(test["hall"]["f1"]),
        "hall_recall": float(test["hall"]["recall"]),
        "hall_aupr": float(test["hall"]["aupr"]),
    }


def summarize_model(model: str, model_out: Path) -> dict[str, Any]:
    rows, missing = [], []
    for case in CASES:
        values = []
        for seed in SEEDS:
            path = model_out / case / f"seed{seed}" / "result.json"
            if not path.exists():
                missing.append(str(path))
                continue
            values.append(flatten(json.loads(path.read_text())))
        if not values:
            continue
        row: dict[str, Any] = {
            "model": model,
            "case": case,
            "display": DISPLAY[case],
            "num_seeds": len(values),
        }
        for metric in METRICS:
            observed = np.asarray([value[metric] for value in values])
            row[f"{metric}_mean"] = float(observed.mean())
            row[f"{metric}_std"] = float(observed.std())
        rows.append(row)
    payload = {
        "model": model,
        "protocol": "shared 3200/400/400 image split; seeds 43/44/45; dual validation thresholds",
        "top_k": TOP_K,
        "tau": 0.03,
        "rows": rows,
        "missing": missing,
    }
    atomic_json(model_out / "complete_results.json", payload)
    fields = ["model", "case", "display", "num_seeds"] + [
        f"{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("mean", "std")
    ]
    if rows:
        with (model_out / "complete_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"# {MODEL_NAMES[model]} targeted Log-S mechanism ablation",
        "",
        "- Shared image split: 3200 train / 400 validation / 400 test.",
        f"- K={TOP_K}, tau=0.03; mean ± population standard deviation over seeds 43/44/45.",
        "- Real and Hall thresholds independently maximize their validation F1.",
        "",
        "| Variant | AUROC | Real F1 | Real AUPR | Hall F1 | Hall Recall | Hall AUPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            f"{100*row[f'{metric}_mean']:.2f} ± {100*row[f'{metric}_std']:.2f}"
            for metric in (
                "auroc",
                "real_f1",
                "real_aupr",
                "hall_f1",
                "hall_recall",
                "hall_aupr",
            )
        ]
        lines.append(f"| {row['display']} | " + " | ".join(values) + " |")
    if missing:
        lines.extend(["", "## Missing", "", *[f"- `{path}`" for path in missing]])
    (model_out / "complete_results.md").write_text("\n".join(lines) + "\n")
    completed = len(CASES) * len(SEEDS) - len(missing)
    total = len(CASES) * len(SEEDS)
    blocks = int(round(30 * completed / total))
    (model_out / "progress.md").write_text(
        f"# {MODEL_NAMES[model]} mechanism ablation\n\n"
        f"`[{'█'*blocks}{'░'*(30-blocks)}] {completed}/{total} "
        f"({100*completed/total:.1f}%)`\n\nMissing runs: {len(missing)}\n"
    )
    print(f"summarized {model}: rows={len(rows)} missing={len(missing)}", flush=True)
    return payload


def summarize_all(output_root: Path) -> None:
    payloads = []
    for model in MODELS:
        path = output_root / model / "complete_results.json"
        if path.exists():
            payloads.append(json.loads(path.read_text()))
    rows = [row for payload in payloads for row in payload["rows"]]
    payload = {
        "protocol": "shared 3200/400/400 image split; seeds 43/44/45; dual validation thresholds",
        "top_k": TOP_K,
        "tau": 0.03,
        "models": [payload["model"] for payload in payloads],
        "rows": rows,
        "complete": len(payloads) == len(MODELS)
        and all(not payload["missing"] for payload in payloads),
    }
    atomic_json(output_root / "complete_results.json", payload)
    if rows:
        fields = list(rows[0])
        with (output_root / "complete_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Four-model targeted Log-S mechanism ablation",
        "",
        "- Shared image split: 3200 train / 400 validation / 400 test.",
        f"- K={TOP_K}, tau=0.03; mean ± population standard deviation over seeds 43/44/45.",
        "",
        "| Model | Variant | AUROC | Real F1 | Hall F1 | Hall AUPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for row in [value for value in rows if value["model"] == model]:
            values = [
                f"{100*row[f'{metric}_mean']:.2f} ± {100*row[f'{metric}_std']:.2f}"
                for metric in ("auroc", "real_f1", "hall_f1", "hall_aupr")
            ]
            lines.append(
                f"| {MODEL_NAMES[model]} | {row['display']} | "
                + " | ".join(values)
                + " |"
            )
    (output_root / "complete_results.md").write_text("\n".join(lines) + "\n")
    print(
        f"summarized all models={len(payloads)}/{len(MODELS)} complete={payload['complete']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--summarize-all", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    _configure_top_k(args.top_k)
    if args.output_root is None:
        args.output_root = OUT
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_all:
        summarize_all(output_root)
        return
    if args.model is None:
        parser.error("--model is required unless --summarize-all is used")
    model_out = output_root / args.model
    model_out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize_model(args.model, model_out)
        return
    data = prepare(args.model, model_out)
    if args.prepare_only:
        print(f"prepared {args.model}: {model_out / 'compact_features.npz'}", flush=True)
        return
    split = load_split()
    device = resolve_device(args.device)
    selected_cases = tuple(args.case) if args.case else CASES
    for case in selected_cases:
        for seed in SEEDS:
            output_dir = model_out / case / f"seed{seed}"
            result_path = output_dir / "result.json"
            if result_path.exists() and not args.overwrite:
                print(f"skip {args.model} {case} seed={seed}", flush=True)
                continue
            print(f"run {args.model} {case} seed={seed} device={device}", flush=True)
            train_case(
                model=args.model,
                case=case,
                matrix=np.asarray(data[case], dtype=np.float32),
                labels=np.asarray(data["label"], dtype=np.int64),
                image_ids=np.asarray(data["image_id"], dtype=np.int64),
                split=split,
                seed=seed,
                device=device,
                output_dir=output_dir,
            )
            summarize_model(args.model, model_out)
    summarize_model(args.model, model_out)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
