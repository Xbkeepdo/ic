#!/usr/bin/env python3
"""Evaluate every extracted K/tau setting on the fixed strict test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_log_s_hpre_k_tau_ablation import (
    GROUPS,
    MODELS,
    OUT,
    RISK_SPECS,
    SEEDS,
    TAU_VALUES,
    TOP_K_VALUES,
    _fit,
    _mean,
    atomic_json,
    load_config,
    prepare,
    ROOT,
)


DISPLAY = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
METRICS = ("real_f1", "real_auc", "hall_aupr")
METRIC_NAMES = {
    "real_f1": "F1",
    "real_auc": "AUROC",
    "hall_aupr": "HALL-AUPR",
}
GROUP_NAMES = {"risk": "Risk-only", "risk_f_log": "Risk + F_log"}


def evaluate_model(
    model: str,
    device_name: str,
    output_root: Path,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    torch.set_num_threads(1)
    model_out = output_root / model
    data = prepare(model, model_out)
    labels = data["label"]
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    specs = RISK_SPECS[shard_index::num_shards]
    total = len(specs) * len(GROUPS) * len(SEEDS)
    completed = 0
    rows: list[dict] = []

    for risk, top_k, tau in specs:
        matrices = {
            "risk": data[risk],
            "risk_f_log": np.concatenate((data[risk], data["f_log"]), axis=1),
        }
        for group, matrix in matrices.items():
            values = []
            for seed in SEEDS:
                folder = model_out / "test_grid" / risk / group / f"seed{seed}"
                values.append(_fit(
                    matrix=matrix,
                    labels=labels,
                    fit_mask=data["train_mask"],
                    eval_mask=data["strict_test_mask"],
                    seed=seed,
                    root_config=root_config,
                    device=torch.device(device_name),
                    folder=folder,
                ))
                completed += 1
                atomic_json(model_out / f"test_grid_progress.shard{shard_index}.json", {
                    "model": model,
                    "stage": "test_grid",
                    "completed": completed,
                    "total": total,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "risk": risk,
                    "group": group,
                    "seed": seed,
                })
            summary = _mean(values)
            rows.append({
                "model": model,
                "risk": risk,
                "top_k": top_k,
                "tau": tau,
                "group": group,
                **{
                    f"{metric}_{stat}": summary[metric][stat]
                    for metric in summary
                    for stat in ("mean", "std")
                },
            })

    path = model_out / f"test_grid_summary.shard{shard_index}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(model_out / f"test_grid_progress.shard{shard_index}.json", {
        "model": model,
        "stage": "test_grid_complete",
        "completed": total,
        "total": total,
        "shard_index": shard_index,
        "num_shards": num_shards,
    })


def merge_model(model: str, output_root: Path, num_shards: int) -> None:
    rows: list[dict] = []
    for shard_index in range(num_shards):
        path = output_root / model / f"test_grid_summary.shard{shard_index}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{model}: missing shard {shard_index}")
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: (
        int(row["top_k"]), float(row["tau"]), GROUPS.index(row["group"])
    ))
    expected = len(RISK_SPECS) * len(GROUPS)
    if len(rows) != expected:
        raise ValueError(f"{model}: expected {expected} rows, got {len(rows)}")
    path = output_root / model / "test_grid_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output_root / model / "test_grid_progress.json", {
        "model": model,
        "stage": "test_grid_complete",
        "completed": len(RISK_SPECS) * len(GROUPS) * len(SEEDS),
        "total": len(RISK_SPECS) * len(GROUPS) * len(SEEDS),
        "num_shards": num_shards,
    })


def _matrix(rows: list[dict], group: str, metric: str) -> np.ndarray:
    lookup = {
        (int(row["top_k"]), float(row["tau"])): float(row[f"{metric}_mean"])
        for row in rows if row["group"] == group
    }
    return np.asarray([
        [lookup[(top_k, tau)] for tau in TAU_VALUES]
        for top_k in TOP_K_VALUES
    ])


def _add_grid(lines: list[str], title: str, values: np.ndarray) -> None:
    lines += [f"### {title}", ""]
    lines.append("| K \\ tau | " + " | ".join(f"{tau:.2f}" for tau in TAU_VALUES) + " |")
    lines.append("|---:|" + "---:|" * len(TAU_VALUES))
    for top_k, row in zip(TOP_K_VALUES, values):
        lines.append(f"| {top_k} | " + " | ".join(f"{value:.4f}" for value in row) + " |")
    lines.append("")


def summarize(output_root: Path) -> None:
    by_model: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    for model in MODELS:
        path = output_root / model / "test_grid_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"{model}: test grid is incomplete")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(RISK_SPECS) * len(GROUPS):
            raise ValueError(f"{model}: expected 56 summary rows, got {len(rows)}")
        by_model[model] = rows
        all_rows.extend(rows)

    combined = output_root / "test_grid_all.csv"
    with combined.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [
        "# Log-s hpre K/tau：完整测试集网格", "",
        "> 这是在查看共享验证选择结果之后补充的测试集描述性分析。测试网格不能用于重新选参或报告无偏的最优测试性能。", "",
        "训练口径：完整训练集拟合、固定严格测试集评估；单隐藏层128、batch 256、dropout 0.3、LR 1e-3、WD 1e-5、patience 10、输入标准化、无BatchNorm；表中为seeds 43/44/45的均值。", "",
        "## 测试网格最高AUROC（仅作描述）", "",
        "| 模型 | 特征 | K | tau | AUROC | HALL-AUPR | F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for group in ("risk_f_log", "risk"):
            candidates = [row for row in by_model[model] if row["group"] == group]
            best = max(candidates, key=lambda row: float(row["real_auc_mean"]))
            lines.append(
                f"| {DISPLAY[model]} | {GROUP_NAMES[group]} | {best['top_k']} | "
                f"{float(best['tau']):.2f} | {float(best['real_auc_mean']):.4f} | "
                f"{float(best['hall_aupr_mean']):.4f} | {float(best['real_f1_mean']):.4f} |"
            )
    lines += ["", "共享参数仍以内部验证选择的 `K=128, tau=0.03` 为正式报告设置。", ""]
    for model in MODELS:
        lines += [f"## {DISPLAY[model]}", ""]
        for group in ("risk_f_log", "risk"):
            for metric in METRICS:
                _add_grid(
                    lines,
                    f"{GROUP_NAMES[group]} {METRIC_NAMES[metric]}",
                    _matrix(by_model[model], group, metric),
                )
    (output_root / "test_grid_results.md").write_text("\n".join(lines) + "\n")
    plot(output_root, by_model)


def plot(output_root: Path, by_model: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for group in GROUPS:
            values = _matrix(by_model[model], group, "real_auc")
            fig, axis = plt.subplots(figsize=(9.2, 4.8))
            image = axis.imshow(values, aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(TAU_VALUES)), [f"{tau:.2f}" for tau in TAU_VALUES])
            axis.set_yticks(range(len(TOP_K_VALUES)), [str(value) for value in TOP_K_VALUES])
            axis.set(xlabel="source softmax temperature tau", ylabel="transport K")
            axis.set_title(f"{DISPLAY[model]}: {GROUP_NAMES[group]} test AUROC")
            for row_index in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    axis.text(
                        column_index, row_index, f"{values[row_index, column_index]:.3f}",
                        ha="center", va="center",
                        color="white" if values[row_index, column_index] < values.mean() else "black",
                        fontsize=8,
                    )
            fig.colorbar(image, ax=axis, label="AUROC")
            fig.tight_layout()
            fig.savefig(figures / f"{model}_{group}_test_auc.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--merge-model", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.summarize:
        summarize(output_root)
    elif args.merge_model and args.model:
        merge_model(args.model, output_root, args.num_shards)
    elif args.model:
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, --num-shards)")
        evaluate_model(
            args.model,
            args.device,
            output_root,
            args.shard_index,
            args.num_shards,
        )
    else:
        parser.error("--model or --summarize is required")


if __name__ == "__main__":
    main()
