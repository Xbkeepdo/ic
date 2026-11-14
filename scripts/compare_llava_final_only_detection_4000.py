#!/usr/bin/env python3
"""Train the fixed LLaVA K32/tau.03 8:2 detector on final-only features."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from scripts.train_log_s_hpre_k_tau_ablation import _fit, SEEDS
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava_1_5_7b"
OLD = ROOT / "outputs/log_s_hpre_k_tau_ablation" / MODEL
OUT = ROOT / "outputs/llava_final_only_all_visual_4000"


def metrics(path: Path) -> dict[str, float]:
    saved = json.loads(path.read_text())["threshold_reports"]["train_f1"]["test_metrics"]
    return {
        "auroc": float(saved["real_positive"]["auc"]),
        "real_aupr": float(saved["real_positive"]["aupr"]),
        "hall_aupr": float(saved["hallucination_positive"]["aupr"]),
        "real_f1": float(saved["real_positive"]["f1"]),
        "hall_f1": float(saved["hallucination_positive"]["f1"]),
        "accuracy": float(saved["accuracy"]),
    }


def stat(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    values = [row[key] for row in rows]
    return {"mean": mean(values), "std": pstdev(values)}


def main() -> None:
    torch.set_num_threads(1)
    with np.load(OLD / "compact_features.npz", allow_pickle=False) as saved:
        old = {key: np.asarray(saved[key]) for key in (
            "image_id", "label", "train_mask", "strict_test_mask",
            "log_s_hpre_k032_tau003", "f_log",
        )}
    n = len(old["label"])
    risk = np.full_like(old["log_s_hpre_k032_tau003"], np.nan)
    sar = np.full_like(old["f_log"], np.nan)
    seen = np.zeros(n, dtype=bool)
    part_count = 0
    for shard in range(2):
        folder = OUT / f"shard{shard}_of_2"
        parts = sorted(folder.glob("part_*.npz"))
        if not parts:
            raise FileNotFoundError(f"Missing extraction parts: {folder}")
        for path in parts:
            with np.load(path, allow_pickle=False) as saved:
                idx = saved["row_indices"].astype(np.int64)
                if seen[idx].any():
                    raise ValueError(f"Duplicate target rows: {path}")
                if not np.array_equal(np.unique(old["image_id"][idx]), saved["image_ids"]):
                    raise ValueError(f"Image/row mismatch: {path}")
                risk[idx] = saved["risk"]
                sar[idx] = saved["sar"]
                seen[idx] = True
            part_count += 1
    if not seen.all() or not np.isfinite(risk).all() or not np.isfinite(sar).all():
        raise ValueError(f"Optimized extraction incomplete: {seen.sum()}/{n} rows")
    if np.intersect1d(
        np.unique(old["image_id"][old["train_mask"]]),
        np.unique(old["image_id"][old["strict_test_mask"]]),
    ).size:
        raise ValueError("Train/test image overlap")

    differences = {}
    for name, before, after in (
        ("risk", old["log_s_hpre_k032_tau003"], risk),
        ("sar", old["f_log"], sar),
    ):
        delta = np.abs(before.astype(np.float64) - after.astype(np.float64)).ravel()
        differences[name] = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "p99_abs": float(np.quantile(delta, 0.99)),
            "fraction_above_1e-3": float(np.mean(delta > 1e-3)),
        }
    matrix = np.concatenate((risk, sar), axis=1)
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for seed in SEEDS:
        folder = OUT / "detector" / f"seed{seed}"
        _fit(
            matrix=matrix,
            labels=old["label"],
            fit_mask=old["train_mask"],
            eval_mask=old["strict_test_mask"],
            seed=seed, root_config=config,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            folder=folder,
        )
        print(f"trained seed {seed}", flush=True)
    before = [metrics(OLD / "test_grid/log_s_hpre_k032_tau003/risk_f_log" / f"seed{seed}" / "metrics.json") for seed in SEEDS]
    after = [metrics(OUT / "detector" / f"seed{seed}" / "metrics.json") for seed in SEEDS]
    names = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")
    payload = {
        "model": MODEL,
        "images_with_targets": int(np.unique(old["image_id"]).size),
        "object_tokens": n,
        "train_object_tokens": int(old["train_mask"].sum()),
        "test_object_tokens": int(old["strict_test_mask"].sum()),
        "parts": part_count,
        "feature_differences": differences,
        "seeds": list(SEEDS),
        "before": before, "after": after,
        "summary": {
            name: {"before": stat(before, name), "after": stat(after, name)}
            for name in names
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# LLaVA COCO4000：final-only 全视觉单批与原方法检测结果",
        "",
        "同一 ENDAC generation、object token 标签及图片级 8:2 划分；原版为 K=32/τ=0.03 的 raw+final norm、概率分块 32；优化版只计算 final norm，全部视觉 token 单批。两者均为完整 SAR+EMS，优化版单独重训同一个 128 宽单隐藏层 MLP；seeds 43/44/45。阈值在训练集上按 Real F1 选定，测试集只评估。",
        "",
        f"覆盖 {payload['images_with_targets']} 张含 object token 的图片、{n} 个 object tokens；训练 {payload['train_object_tokens']}，测试 {payload['test_object_tokens']}。", "",
        "| 指标（%） | 原版均值 ± SD | 优化版均值 ± SD | 差值（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for name in names:
        a = payload["summary"][name]["before"]
        b = payload["summary"][name]["after"]
        lines.append(f"| {name} | {a['mean']*100:.2f} ± {a['std']*100:.2f} | {b['mean']*100:.2f} ± {b['std']*100:.2f} | {(b['mean']-a['mean'])*100:+.2f} |")
    lines += ["", "## 特征数值差异", "", "| 特征 | 最大绝对差 | 平均绝对差 | P99 绝对差 | 差异 > 1e-3 占比 |", "|---|---:|---:|---:|---:|"]
    for name in ("risk", "sar"):
        d = differences[name]
        lines.append(f"| {name} | {d['max_abs']:.6g} | {d['mean_abs']:.6g} | {d['p99_abs']:.6g} | {d['fraction_above_1e-3']*100:.3f}% |")
    lines += ["", "逐种子原始指标及覆盖审计见 `complete_results.json`，优化版检测器结果见 `detector/seed*/metrics.json`。", ""]
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
