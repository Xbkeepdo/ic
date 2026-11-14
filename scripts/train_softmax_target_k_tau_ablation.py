#!/usr/bin/env python3
"""Train attention- and norm-score softmax target ablations on ENDAC."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import hashlib
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

from scripts.train_fixed_wcost_log_gate_log_am import atomic_json, atomic_npz, report  # noqa: E402
from scripts.train_iclr_endac_attention_i_fusion import _probe_config  # noqa: E402
from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    _json_ready,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


MODELS = (
    "qwen2_5_vl_7b",
    "llava_1_5_7b",
    "qwen3_vl_8b",
    "internvl_2_5_8b",
)
SEEDS = (43, 44, 45)
SPLIT_SEED = 20260921
TOP_K_VALUES = (32, 64)
TAU_VALUES = (0.03, 0.05, 0.07)
TARGET_MODES = ("attn_softmax", "norm_softmax")


def risk_name(mode: str, top_k: int, tau: float) -> str:
    return f"{mode}_k{int(top_k):03d}_tau{round(float(tau) * 100):03d}"


RISK_SPECS = tuple(
    (risk_name(mode, top_k, tau), mode, top_k, tau)
    for mode in TARGET_MODES
    for top_k in TOP_K_VALUES
    for tau in TAU_VALUES
)
RISK_NAMES = tuple(name for name, _mode, _top_k, _tau in RISK_SPECS)
EXTRACTED = GF_ROOT / "outputs/coco4000_512_softmax_target_k_tau_ablation_811"
BASE = ROOT / "outputs/linear_cost_log_s_risk_four_f_fusion"
OUT = ROOT / "outputs/softmax_target_k_tau_ablation"
GROUPS = ("risk", "risk_f_log")


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def _metrics(path: Path, *, split: str = "test") -> dict[str, float]:
    saved = json.loads(path.read_text())
    values = report(saved, "train_f1")
    return {
        "real_f1": float(values["real_positive"]["f1"]),
        "real_auc": float(values["real_positive"]["auc"]),
        "hall_aupr": float(values["hallucination_positive"]["aupr"]),
        "split": split,
    }


def _mean(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in values])),
            "std": float(np.std([row[metric] for row in values])),
        }
        for metric in ("real_f1", "real_auc", "hall_aupr")
    }


def prepare(model: str, model_out: Path) -> dict[str, np.ndarray]:
    cache = model_out / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return {key: np.asarray(saved[key]) for key in saved.files}
    new_path = EXTRACTED / model / "features.pkl"
    summary_path = EXTRACTED / model / "summary.json"
    base_path = BASE / model / "compact_features.npz"
    if not new_path.exists() or not summary_path.exists() or not base_path.exists():
        raise FileNotFoundError(f"{model}: extraction/base compact features are incomplete")
    rows = load_pkl(new_path)
    summary = json.loads(summary_path.read_text())
    with np.load(base_path, allow_pickle=False) as saved:
        base = {key: np.asarray(saved[key]) for key in saved.files}
    if len(rows) != len(base["label"]) or len(rows) != int(summary["samples"]):
        raise ValueError(f"{model}: row count differs between ablation and base")
    base_keys = list(zip(
        base["image_id"].astype(int).tolist(),
        base["response_token_idx"].astype(int).tolist(),
        base["target_token_id"].astype(int).tolist(),
    ))
    row_by_key = {_key(row): row for row in rows}
    if len(row_by_key) != len(rows) or set(row_by_key) != set(base_keys):
        raise ValueError(f"{model}: ablation/base mention keys differ")
    if summary.get("schema") != "softmax-target-k-tau-ablation-811-v1":
        raise ValueError(f"{model}: unexpected extraction schema")
    layers = int(base["log_residual_fusion"].shape[1])
    data = {
        name: np.stack([
            np.asarray(row_by_key[key]["risks"][name], dtype=np.float32)
            for key in base_keys
        ])
        for name in RISK_NAMES
    }
    for name, value in data.items():
        if value.shape != (len(rows), layers) or not np.isfinite(value).all():
            raise ValueError(f"{model}: invalid {name} matrix {value.shape}")
    reconstructed_f_log = np.stack([
        np.asarray(row_by_key[key]["f_log"], dtype=np.float32)
        for key in base_keys
    ])
    baseline_f_log = np.asarray(base["log_residual_fusion"], dtype=np.float32)
    if not np.allclose(reconstructed_f_log, baseline_f_log, rtol=2e-4, atol=1e-4):
        maximum_error = float(np.max(np.abs(reconstructed_f_log - baseline_f_log)))
        raise ValueError(f"{model}: F_log differs from aligned baseline (max={maximum_error})")
    data.update({
        "f_log": np.asarray(base["log_residual_fusion"], dtype=np.float32),
        "label": np.asarray(base["label"], dtype=np.int8),
        "image_id": np.asarray(base["image_id"], dtype=np.int64),
        "response_token_idx": np.asarray(base["response_token_idx"], dtype=np.int32),
        "target_token_id": np.asarray(base["target_token_id"], dtype=np.int32),
        "train_mask": np.asarray(base["train_mask"], dtype=bool),
        "strict_test_mask": np.asarray(base["strict_test_mask"], dtype=bool),
    })
    atomic_npz(cache, **data)
    atomic_json(model_out / "feature_audit.json", {
        "model": model,
        "schema": summary["schema"],
        "source": str(new_path),
        "base": str(base_path),
        "targets": len(rows),
        "layers": layers,
        "risk_fields": list(RISK_NAMES),
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "f_log": "reused log_residual_fusion from the aligned base compact cache",
        "f_log_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed_f_log - baseline_f_log))
        ),
        "target_diagnostics": {
            mode: {
                "mean_peak_probability": float(np.mean([
                    np.asarray(row["target_peak"][mode], dtype=np.float64).mean()
                    for row in rows
                ])),
                "mean_entropy": float(np.mean([
                    np.asarray(row["target_entropy"][mode], dtype=np.float64).mean()
                    for row in rows
                ])),
            }
            for mode in TARGET_MODES
        },
    })
    return data


def make_split(model: str, data: dict[str, np.ndarray], model_out: Path) -> tuple[np.ndarray, np.ndarray]:
    ids = data["image_id"]
    train_images = np.unique(ids[data["train_mask"]])
    test_images = np.unique(ids[data["strict_test_mask"]])
    if np.intersect1d(train_images, test_images).size:
        raise ValueError(f"{model}: train/test images overlap")
    rng = np.random.default_rng(SPLIT_SEED)
    permuted = rng.permutation(np.sort(train_images))
    validation_count = max(1, round(0.2 * len(train_images)))
    validation_images = np.sort(permuted[:validation_count])
    fit_images = np.sort(permuted[validation_count:])
    fit_mask = data["train_mask"] & np.isin(ids, fit_images)
    validation_mask = data["train_mask"] & np.isin(ids, validation_images)
    if np.any(fit_mask & validation_mask) or not np.array_equal(
        fit_mask | validation_mask, data["train_mask"]
    ):
        raise ValueError(f"{model}: internal image split is invalid")
    payload = {
        "seed": SPLIT_SEED,
        "source_train_image_sha256": hashlib.sha256(
            ",".join(map(str, np.sort(train_images))).encode()
        ).hexdigest(),
        "fit_images": fit_images.tolist(),
        "validation_images": validation_images.tolist(),
        "test_images": np.sort(test_images).tolist(),
        "counts": {
            "fit_images": len(fit_images),
            "validation_images": len(validation_images),
            "test_images": len(test_images),
            "fit_targets": int(fit_mask.sum()),
            "validation_targets": int(validation_mask.sum()),
            "test_targets": int(data["strict_test_mask"].sum()),
        },
    }
    path = model_out / "selection_image_split.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError(f"{model}: existing internal split differs")
    atomic_json(path, payload)
    return fit_mask, validation_mask


def _probe(root_config: dict, seed: int):
    return replace(
        _probe_config(root_config, seed),
        hidden_sizes=(128,),
        batch_size=256,
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        early_stopping_patience=10,
        batch_norm=False,
        standardize_input=True,
    )


def _fit(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    eval_mask: np.ndarray,
    seed: int,
    root_config: dict,
    device: torch.device,
    folder: Path,
) -> dict[str, float]:
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        return _metrics(metrics_path)
    probe = _probe(root_config, seed)
    atomic_json(folder / "config.json", _json_ready(asdict(probe)))
    metrics = train_and_evaluate_probe(
        X_train=matrix[fit_mask],
        y_train=labels[fit_mask],
        X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
        y_val=np.empty((0,), dtype=np.int64),
        X_test=matrix[eval_mask],
        y_test=labels[eval_mask],
        config=probe,
        device=device,
        output_dir=str(folder),
    )
    atomic_json(metrics_path, metrics)
    return _metrics(metrics_path)


def validate(model: str, device_name: str, output_root: Path) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    model_out.mkdir(parents=True, exist_ok=True)
    data = prepare(model, model_out)
    fit_mask, validation_mask = make_split(model, data, model_out)
    labels = data["label"]
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    total = len(RISK_NAMES) * len(GROUPS) * len(SEEDS)
    completed = 0
    rows = []
    for name, mode, top_k, tau in RISK_SPECS:
        matrices = {
            "risk": data[name],
            "risk_f_log": np.concatenate((data[name], data["f_log"]), axis=1),
        }
        for group, matrix in matrices.items():
            values = []
            for seed in SEEDS:
                folder = model_out / "validation" / name / group / f"seed{seed}"
                value = _fit(
                    matrix=matrix,
                    labels=labels,
                    fit_mask=fit_mask,
                    eval_mask=validation_mask,
                    seed=seed,
                    root_config=root_config,
                    device=torch.device(device_name),
                    folder=folder,
                )
                values.append(value)
                completed += int((folder / "metrics.json").exists())
                atomic_json(model_out / "progress.json", {
                    "model": model,
                    "stage": "validation",
                    "completed": min(completed, total),
                    "total": total,
                    "risk": name,
                    "group": group,
                    "seed": seed,
                })
            summary = _mean(values)
            rows.append({
                "model": model,
                "risk": name,
                "mode": mode,
                "top_k": top_k,
                "tau": tau,
                "group": group,
                **{
                    f"{metric}_{stat}": summary[metric][stat]
                    for metric in summary
                    for stat in ("mean", "std")
                },
            })
    _write_csv_atomic(model_out / "validation_summary.csv", rows)
    atomic_json(model_out / "progress.json", {
        "model": model,
        "stage": "validation_complete",
        "completed": total,
        "total": total,
    })


def select_shared(output_root: Path) -> dict:
    all_rows = []
    for model in MODELS:
        path = output_root / model / "validation_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"{model}: validation is incomplete")
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["group"] == "risk_f_log"]
        ranked = sorted(
            rows,
            key=lambda row: (float(row["real_auc_mean"]), float(row["hall_aupr_mean"])),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            row["model_rank"] = rank
            all_rows.append(row)
    candidates = []
    for name, mode, top_k, tau in RISK_SPECS:
        rows = [row for row in all_rows if row["risk"] == name]
        candidates.append({
            "risk": name,
            "mode": mode,
            "top_k": top_k,
            "tau": tau,
            "mean_model_rank": float(np.mean([int(row["model_rank"]) for row in rows])),
            "mean_validation_real_auc": float(np.mean([float(row["real_auc_mean"]) for row in rows])),
            "mean_validation_hall_aupr": float(np.mean([float(row["hall_aupr_mean"]) for row in rows])),
            "models_improved_vs_baseline": 0,
        })
    baseline_by_model = {
        model: next(
            row for row in all_rows
            if row["model"] == model and row["mode"] == "attn_softmax"
            and int(row["top_k"]) == 32 and float(row["tau"]) == 0.03
        )
        for model in MODELS
    }
    for candidate in candidates:
        rows = [row for row in all_rows if row["risk"] == candidate["risk"]]
        candidate["models_improved_vs_baseline"] = sum(
            float(row["real_auc_mean"]) > float(baseline_by_model[row["model"]]["real_auc_mean"])
            for row in rows
        )
    candidates.sort(key=lambda row: (
        row["mean_model_rank"],
        -row["mean_validation_real_auc"],
        -row["mean_validation_hall_aupr"],
        row["top_k"],
        -row["tau"],
        row["mode"],
    ))
    selected = candidates[0]
    payload = {
        "selection_protocol": (
            "lowest mean within-model validation AUROC rank; tie mean validation AUROC, "
            "Hall AUPR, smaller K, then larger tau"
        ),
        "models": list(MODELS),
        "selected": selected,
        "ranking": candidates,
    }
    atomic_json(output_root / "shared_selection.json", payload)
    lines = [
        "# Shared K/tau selection", "",
        "只使用原训练集内部的按图片80/20验证划分；固定测试集未参与选择。", "",
        f"选择结果：`{selected['mode']}`, `K={selected['top_k']}`, `tau={selected['tau']:.2f}`。", "",
        "| Rank | T mode | K | tau | Mean model rank | Mean validation AUC | Mean Hall AUPR | Improved models |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(candidates, 1):
        lines.append(
            f"| {rank} | {row['mode']} | {row['top_k']} | {row['tau']:.2f} | {row['mean_model_rank']:.2f} | "
            f"{row['mean_validation_real_auc']:.4f} | {row['mean_validation_hall_aupr']:.4f} | "
            f"{row['models_improved_vs_baseline']}/4 |"
        )
    (output_root / "shared_selection.md").write_text("\n".join(lines) + "\n")
    plot_validation(output_root)
    return payload


def plot_validation(output_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows_by_model = {}
    for model in MODELS:
        with (output_root / model / "validation_summary.csv").open(newline="") as handle:
            rows_by_model[model] = list(csv.DictReader(handle))

    def matrix(rows: list[dict], group: str, mode: str) -> np.ndarray:
        lookup = {
            (int(row["top_k"]), float(row["tau"])): float(row["real_auc_mean"])
            for row in rows if row["group"] == group and row["mode"] == mode
        }
        return np.asarray([
            [lookup[(top_k, tau)] for tau in TAU_VALUES]
            for top_k in TOP_K_VALUES
        ])

    for model in (*MODELS, "four_model_mean"):
        for mode in TARGET_MODES:
          for group in GROUPS:
            values = (
                matrix(rows_by_model[model], group, mode)
                if model in rows_by_model
                else np.mean([matrix(rows_by_model[name], group, mode) for name in MODELS], axis=0)
            )
            fig, axis = plt.subplots(figsize=(9.2, 4.8))
            image = axis.imshow(values, aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(TAU_VALUES)), [f"{tau:.2f}" for tau in TAU_VALUES])
            axis.set_yticks(range(len(TOP_K_VALUES)), [str(value) for value in TOP_K_VALUES])
            axis.set_xlabel("source softmax temperature tau")
            axis.set_ylabel("transport K")
            axis.set_title(f"{model}: {mode} {group} validation AUROC")
            for row_index in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    axis.text(
                        column_index,
                        row_index,
                        f"{values[row_index, column_index]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if values[row_index, column_index] < values.mean() else "black",
                        fontsize=8,
                    )
            fig.colorbar(image, ax=axis, label="AUROC")
            fig.tight_layout()
            fig.savefig(figures / f"{model}_{mode}_{group}_validation_auc.png", dpi=180)
            plt.close(fig)

    # The per-model heatmaps above retain the full two-mode factorial comparison.


def final_test(model: str, device_name: str, output_root: Path) -> None:
    selection_path = output_root / "shared_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run --select-shared after all validation jobs")
    selected = json.loads(selection_path.read_text())["selected"]
    risk = str(selected["risk"])
    model_out = output_root / model
    data = prepare(model, model_out)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    matrices = {
        "risk": data[risk],
        "risk_f_log": np.concatenate((data[risk], data["f_log"]), axis=1),
    }
    results = {}
    for group, matrix in matrices.items():
        values = []
        for seed in SEEDS:
            grid_metrics = model_out / "test_grid" / risk / group / f"seed{seed}" / "metrics.json"
            if grid_metrics.exists():
                values.append(_metrics(grid_metrics))
                continue
            folder = model_out / "final_test" / risk / group / f"seed{seed}"
            values.append(_fit(
                matrix=matrix,
                labels=data["label"],
                fit_mask=data["train_mask"],
                eval_mask=data["strict_test_mask"],
                seed=seed,
                root_config=root_config,
                device=torch.device(device_name),
                folder=folder,
            ))
        results[group] = _mean(values)
    atomic_json(model_out / "final_test_summary.json", {
        "model": model,
        "selected": selected,
        "metrics": results,
    })


def test_grid(model: str, device_name: str, output_root: Path) -> None:
    """Evaluate all six settings for descriptive ablation after validation selection."""
    selection_path = output_root / "shared_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run --select-shared before test-grid evaluation")
    model_out = output_root / model
    data = prepare(model, model_out)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    rows = []
    total = len(RISK_SPECS) * len(GROUPS) * len(SEEDS)
    completed = 0
    for name, mode, top_k, tau in RISK_SPECS:
        matrices = {
            "risk": data[name],
            "risk_f_log": np.concatenate((data[name], data["f_log"]), axis=1),
        }
        for group, matrix in matrices.items():
            values = []
            for seed in SEEDS:
                folder = model_out / "test_grid" / name / group / f"seed{seed}"
                values.append(_fit(
                    matrix=matrix,
                    labels=data["label"],
                    fit_mask=data["train_mask"],
                    eval_mask=data["strict_test_mask"],
                    seed=seed,
                    root_config=root_config,
                    device=torch.device(device_name),
                    folder=folder,
                ))
                completed += 1
                atomic_json(model_out / "test_grid_progress.json", {
                    "model": model,
                    "completed": completed,
                    "total": total,
                    "risk": name,
                    "group": group,
                    "seed": seed,
                })
            summary = _mean(values)
            rows.append({
                "model": model,
                "risk": name,
                "mode": mode,
                "top_k": top_k,
                "tau": tau,
                "group": group,
                **{
                    f"{metric}_{stat}": summary[metric][stat]
                    for metric in summary
                    for stat in ("mean", "std")
                },
            })
    _write_csv_atomic(model_out / "test_grid_summary.csv", rows)


def summarize_grid(output_root: Path) -> None:
    selected = json.loads((output_root / "shared_selection.json").read_text())["selected"]
    old_roots = {
        "attn_sum": ROOT / "outputs/log_s_hpre_k_tau_ablation",
        "norm_sum": ROOT / "outputs/flog_target_k_tau_ablation",
    }
    lines = [
        "# Softmax target K/tau 消融：完整测试网格", "",
        "`s_m=log(1+C*p_m)`；attention 路线 `T=softmax_m(A_m*s_m)`；",
        "norm 路线 `T=softmax_m(||a_m||_2*(1+s_m))`。T softmax 温度固定为 1。", "",
        "source、cost、精确 EMD、标签、训练配置和划分与旧实验一致。",
        f"内部验证选出的共享配置：`{selected['mode']}`, "
        f"`K={selected['top_k']}`, `tau={selected['tau']:.2f}`。",
        "完整测试网格仅作消融描述，未用于选参。", "",
    ]
    for model in MODELS:
        new_path = output_root / model / "test_grid_summary.csv"
        old_paths = {key: root / model / "test_grid_summary.csv" for key, root in old_roots.items()}
        if not new_path.exists() or any(not path.exists() for path in old_paths.values()):
            raise FileNotFoundError(f"{model}: new/old test grid missing")
        with new_path.open(newline="") as handle:
            new_rows = list(csv.DictReader(handle))
        old_by_setting = {}
        for method, path in old_paths.items():
            with path.open(newline="") as handle:
                old_by_setting[method] = {
                    (int(row["top_k"]), float(row["tau"]), row["group"]): row
                    for row in csv.DictReader(handle)
                }
        lines.extend([
            f"## {model}", "",
            "| Target | K | tau | Input | AUROC | Hall AUPR | attn sum AUROC | norm sum AUROC |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ])
        for row in new_rows:
            key = (int(row["top_k"]), float(row["tau"]), row["group"])
            attn_old = float(old_by_setting["attn_sum"][key]["real_auc_mean"])
            norm_old = float(old_by_setting["norm_sum"][key]["real_auc_mean"])
            lines.append(
                f"| {row['mode']} | {key[0]} | {key[1]:.2f} | `{key[2]}` | "
                f"{float(row['real_auc_mean']):.4f}±{float(row['real_auc_std']):.4f} | "
                f"{float(row['hall_aupr_mean']):.4f}±{float(row['hall_aupr_std']):.4f} | "
                f"{attn_old:.4f} | {norm_old:.4f} |"
            )
        lines.append("")
    (output_root / "complete_results.md").write_text("\n".join(lines) + "\n")


def summarize_final(output_root: Path) -> None:
    selection = json.loads((output_root / "shared_selection.json").read_text())["selected"]
    lines = [
        "# Softmax target K/tau ablation: final test", "",
        f"Shared validation-selected setting: `{selection['mode']}`, "
        f"`K={selection['top_k']}`, `tau={selection['tau']:.2f}`.", "",
        "| Model | Feature | Real F1 | Real AUC | Hall AUPR |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        path = output_root / model / "final_test_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"{model}: final test is incomplete")
        metrics = json.loads(path.read_text())["metrics"]
        for group in GROUPS:
            row = metrics[group]
            lines.append(
                f"| {model} | `{group}` | {row['real_f1']['mean']:.4f}±{row['real_f1']['std']:.4f} | "
                f"{row['real_auc']['mean']:.4f}±{row['real_auc']['std']:.4f} | "
                f"{row['hall_aupr']['mean']:.4f}±{row['hall_aupr']['std']:.4f} |"
            )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--stage", choices=("prepare", "validation", "final", "test-grid"), default="validation")
    parser.add_argument("--select-shared", action="store_true")
    parser.add_argument("--summarize-final", action="store_true")
    parser.add_argument("--summarize-grid", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.select_shared:
        select_shared(output_root)
        return
    if args.summarize_final:
        summarize_final(output_root)
        return
    if args.summarize_grid:
        summarize_grid(output_root)
        return
    if args.model is None:
        parser.error("--model is required")
    if args.stage == "prepare":
        prepare(args.model, output_root / args.model)
    elif args.stage == "validation":
        validate(args.model, args.device, output_root)
    elif args.stage == "test-grid":
        test_grid(args.model, args.device, output_root)
    else:
        final_test(args.model, args.device, output_root)


if __name__ == "__main__":
    main()
