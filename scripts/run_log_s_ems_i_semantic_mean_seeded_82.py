#!/usr/bin/env python3
"""Compare semantic probability mean with sum under the same seeded COCO setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from scripts.run_log_s_ems_i_semantic_mass_seeded_82 import (
    MODELS,
    ROOT,
    GF_ROOT,
    OUT_42_44,
    OUT_43_45,
    original_result_path,
    score_metrics,
)


SEEDS = (43, 44, 45)
OUT = ROOT / "outputs/log_s_ems_i_semantic_mean_seeded_82_43_45"
KEYS = ("auroc", "real_aupr", "hall_aupr")


def sum_result_path(model: str, seed: int) -> Path:
    root = OUT_42_44 if seed in (43, 44) else OUT_43_45
    return root / model / f"seed{seed}" / "metrics.json"


def load_features(model: str, audit_dir: Path = OUT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    mechanism_path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    semantic_path = ROOT / "outputs/norm_semantic_transform_curves_by_label" / model / "compact_features.npz"
    visual_path = GF_ROOT / "outputs/coco4000_512_endac_attention_i_fusion_811" / model / "features.pt"
    with np.load(mechanism_path, allow_pickle=False) as source:
        image_id = np.asarray(source["image_id"], dtype=np.int64)
        label = np.asarray(source["label"], dtype=np.int8)
        ems = np.asarray(source["ems"], dtype=np.float32)
        visual_write = np.asarray(source["i_ems"][:, :ems.shape[1]], dtype=np.float32)
    with np.load(semantic_path, allow_pickle=False) as source:
        if not (np.array_equal(image_id, source["image_id"]) and np.array_equal(label, source["label"])):
            raise ValueError(f"{model}: semantic rows differ from EMS")
        semantic_sum = np.asarray(source["semantic_mass"], dtype=np.float32)
        counts = np.asarray(source["visual_tokens"], dtype=np.int32)
    if np.any(counts <= 0) or ems.shape != visual_write.shape or ems.shape != semantic_sum.shape:
        raise ValueError(f"{model}: invalid visual count or feature shapes")
    semantic_mean = (semantic_sum / counts[:, None]).astype(np.float32)
    visual = torch.load(visual_path, map_location="cpu", weights_only=False, mmap=True)
    if not (np.array_equal(image_id, np.asarray(visual["image_id"]))
            and np.array_equal(label, np.asarray(visual["labels"]))
            and np.array_equal(visual_write, np.asarray(visual["I"], dtype=np.float32))):
        raise ValueError(f"{model}: visual I metadata/values differ from EMS")
    visual_semantic_mean = np.asarray(visual["all_visual"]["norm"], dtype=np.float32)[:, :, 0]
    if not np.allclose(semantic_mean, visual_semantic_mean, rtol=1e-5, atol=1e-7):
        raise ValueError(f"{model}: derived semantic mean differs from saved mean")
    matrix = np.concatenate((ems, visual_write, semantic_mean), axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{model}: nonfinite input")
    audit = {
        "model": model,
        "rows": int(len(label)),
        "layers": int(ems.shape[1]),
        "feature_order": ["EMS", "I", "E_mean"],
        "formula": "E_mean[l] = semantic_mass[l] / visual_tokens = mean_m p_lm",
        "visual_tokens_min": int(counts.min()),
        "visual_tokens_max": int(counts.max()),
        "max_abs_error_against_saved_mean": float(np.max(np.abs(semantic_mean - visual_semantic_mean))),
        "mechanism_source": str(mechanism_path),
        "semantic_source": str(semantic_path),
        "visual_validation_source": str(visual_path),
    }
    target = audit_dir / model / "alignment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, indent=2) + "\n")
    return matrix, label, image_id


def run_model(model: str, device_name: str) -> None:
    import torch

    from scripts.train_log_s_hpre_k_tau_ablation import _fit
    from utils.config_utils import load_config

    torch.set_num_threads(1)
    matrix, labels, images = load_features(model)
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for seed in SEEDS:
        reference = OUT_43_45 / "splits" / f"seed{seed}.json"
        split = json.loads(reference.read_text())
        target = OUT / "splits" / f"seed{seed}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and json.loads(target.read_text()) != split:
            raise ValueError(f"seed{seed}: saved split differs")
        target.write_text(json.dumps(split, indent=2) + "\n")
        fit_mask = np.isin(images, split["train"])
        test_mask = np.isin(images, split["test"])
        if not np.all(fit_mask | test_mask) or np.any(fit_mask & test_mask):
            raise ValueError(f"{model}/seed{seed}: invalid image split")
        if not (original_result_path(model, seed).exists() and sum_result_path(model, seed).exists()):
            raise FileNotFoundError(f"{model}/seed{seed}: missing prior comparison metrics")
        folder = OUT / model / f"seed{seed}"
        if (folder / "metrics.json").exists():
            print(f"[{model}] reuse seed{seed}: EMS+I+E_mean", flush=True)
            continue
        _fit(matrix=matrix, labels=labels, fit_mask=fit_mask, eval_mask=test_mask,
             seed=seed, root_config=config, device=torch.device(device_name), folder=folder)
        print(f"[{model}] completed seed{seed}: EMS+I+E_mean", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO 8:2：语义概率 E 的求和与平均对比", "",
        "LLaVA-1.5-7B、Qwen3-VL-8B；图片划分和 MLP 初始化均由种子 43/44/45 控制。复用相同 EMS、I、object-token 对齐和单隐藏层训练器。",
        "求和版 $E_l=\\sum_m p_{l,m}$；平均版 $\\bar E_l=E_l/M_l$，其中 $M_l$ 为该样本的视觉 token 数。各输入维度仅用训练集统计量标准化。",
        "所有指标在对应种子的测试图片上计算；± 为三个种子的总体标准差。", "",
        "| 模型 | 特征 | AUROC (%) | Real AUPR (%) | Hall AUPR (%) |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {"sar": "EMS+SAR", "sum": "EMS+I+E_sum", "mean": "EMS+I+E_mean"}
    for model in MODELS:
        values = {}
        for group in ("sar", "sum", "mean"):
            values[group] = []
            for seed in SEEDS:
                path = (original_result_path(model, seed) if group == "sar" else
                        sum_result_path(model, seed) if group == "sum" else
                        OUT / model / f"seed{seed}" / "metrics.json")
                item = score_metrics(json.loads(path.read_text()))
                values[group].append(item)
                rows.append({"model": model, "group": labels[group], "seed": seed, **item})
            formatted = [f"{100*mean(v[k] for v in values[group]):.2f} ± {100*pstdev(v[k] for v in values[group]):.2f}" for k in KEYS]
            lines.append(f"| {model} | {labels[group]} | {' | '.join(formatted)} |")
        delta = [mean(100*(a[k]-b[k]) for a,b in zip(values["mean"], values["sum"])) for k in KEYS]
        lines.append(f"| {model} | Δ mean − sum | {' | '.join(f'{x:+.2f}' for x in delta)} |")
    lines.extend(["", "## 逐种子：平均版相对求和版（百分点）", "",
                  "| 模型 | 种子 | ΔAUROC | ΔReal AUPR | ΔHall AUPR |", "|---|---:|---:|---:|---:|"])
    for model in MODELS:
        for seed in SEEDS:
            a = next(r for r in rows if r["model"] == model and r["seed"] == seed and r["group"] == labels["mean"])
            b = next(r for r in rows if r["model"] == model and r["seed"] == seed and r["group"] == labels["sum"])
            lines.append(f"| {model} | {seed} | {' | '.join(f'{100*(a[k]-b[k]):+.2f}' for k in KEYS)} |")
    lines.extend(["", "逐种子完整数值见 `per_seed.json`；特征来源与验证见各模型的 `alignment.json`。", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model is required unless --summarize is set")


if __name__ == "__main__":
    main()
