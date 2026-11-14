#!/usr/bin/env python3
"""Compare SAR+EMS with EMS+visual-write-I+semantic-mass on seeded COCO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from scripts.run_log_s_ot_js_seeded_82 import (
    MODELS,
    ROOT,
    score_metrics,
)


SEEDS = (42, 43, 44)
OUT_42_44 = ROOT / "outputs/log_s_ems_i_semantic_mass_seeded_82_42_44"
OUT_43_45 = ROOT / "outputs/log_s_ems_i_semantic_mass_seeded_82_43_45"
OTJS_42_44 = ROOT / "outputs/log_s_ot_js_seeded_82_42_44"
REFERENCE_OUT = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
OUT = OUT_42_44
GF_ROOT = ROOT.parent / "gfchair"


def configure(seed_set: str) -> None:
    global SEEDS, OUT
    if seed_set == "42-44":
        SEEDS, OUT = (42, 43, 44), OUT_42_44
    elif seed_set == "43-45":
        SEEDS, OUT = (43, 44, 45), OUT_43_45
    else:
        raise ValueError(f"unknown seed set: {seed_set}")


def original_result_path(model: str, seed: int) -> Path:
    if seed == 42:
        return OTJS_42_44 / model / "sar_ot" / "seed42" / "metrics.json"
    return REFERENCE_OUT / model / f"seed{seed}" / "ours" / "metrics.json"


def new_result_path(model: str, seed: int) -> Path:
    if OUT == OUT_43_45 and seed in (43, 44):
        return OUT_42_44 / model / f"seed{seed}" / "metrics.json"
    return OUT / model / f"seed{seed}" / "metrics.json"


def load_image_splits() -> dict[int, dict[str, list[int]]]:
    splits = {}
    for seed in SEEDS:
        root = OTJS_42_44 if seed == 42 else REFERENCE_OUT
        split = json.loads((root / "splits" / f"seed{seed}.json").read_text())
        if len(set(split["train"])) != 3200 or len(set(split["test"])) != 800:
            raise ValueError(f"seed{seed}: invalid image split size")
        if set(split["train"]) & set(split["test"]):
            raise ValueError(f"seed{seed}: overlapping image split")
        target = OUT / "splits" / f"seed{seed}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and json.loads(target.read_text()) != split:
            raise ValueError(f"seed{seed}: saved image split differs")
        target.write_text(json.dumps(split, indent=2) + "\n")
        splits[seed] = split
    return splits


def load_features(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    mechanism_path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    semantic_path = ROOT / "outputs/norm_semantic_transform_curves_by_label" / model / "compact_features.npz"
    visual_path = GF_ROOT / "outputs/coco4000_512_endac_attention_i_fusion_811" / model / "features.pt"
    with np.load(mechanism_path, allow_pickle=False) as source:
        image_id = np.asarray(source["image_id"], dtype=np.int64)
        label = np.asarray(source["label"], dtype=np.int8)
        ems = np.asarray(source["ems"], dtype=np.float32)
        i_from_mechanism = np.asarray(source["i_ems"][:, :ems.shape[1]], dtype=np.float32)
    with np.load(semantic_path, allow_pickle=False) as source:
        if not (np.array_equal(image_id, source["image_id"]) and np.array_equal(label, source["label"])):
            raise ValueError(f"{model}: semantic mass row metadata differs from EMS")
        semantic_mass = np.asarray(source["semantic_mass"], dtype=np.float32)
        visual_counts = np.asarray(source["visual_tokens"], dtype=np.int32)
    visual = torch.load(visual_path, map_location="cpu", weights_only=False, mmap=True)
    if not (np.array_equal(image_id, np.asarray(visual["image_id"]))
            and np.array_equal(label, np.asarray(visual["labels"]))
            and np.array_equal(i_from_mechanism, np.asarray(visual["I"], dtype=np.float32))):
        raise ValueError(f"{model}: visual I row order/values differ from EMS")
    semantic_mean = np.asarray(visual["all_visual"]["norm"], dtype=np.float64)[:, :, 0]
    reconstructed_mass = semantic_mean * visual_counts[:, None]
    if not np.allclose(reconstructed_mass, semantic_mass, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{model}: semantic mass differs from all-visual mean × visual-token count")
    if ems.shape != i_from_mechanism.shape or ems.shape != semantic_mass.shape:
        raise ValueError(f"{model}: EMS, I, E shapes differ")
    matrix = np.concatenate((ems, i_from_mechanism, semantic_mass), axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{model}: nonfinite feature")
    audit = {
        "model": model,
        "rows": len(label),
        "layers": int(ems.shape[1]),
        "feature_order": ["EMS", "I", "E"],
        "visual_tokens_min": int(visual_counts.min()),
        "visual_tokens_max": int(visual_counts.max()),
        "i_exact_rowwise_match": True,
        "semantic_mass_reconstruction_max_abs_error": float(np.max(np.abs(reconstructed_mass-semantic_mass))),
        "mechanism_source": str(mechanism_path),
        "semantic_mass_source": str(semantic_path),
        "visual_i_source": str(visual_path),
    }
    path = OUT / model / "alignment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2) + "\n")
    return matrix, label, image_id


def run_model(model: str, device_name: str) -> None:
    import torch

    from scripts.train_log_s_hpre_k_tau_ablation import _fit
    from utils.config_utils import load_config

    torch.set_num_threads(1)
    matrix, labels, images = load_features(model)
    splits = load_image_splits()
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for seed in SEEDS:
        split = splits[seed]
        fit_mask = np.isin(images, split["train"])
        test_mask = np.isin(images, split["test"])
        if not np.all(fit_mask | test_mask) or np.any(fit_mask & test_mask):
            raise ValueError(f"{model}/seed{seed}: invalid image split")
        if not original_result_path(model, seed).exists():
            raise FileNotFoundError(original_result_path(model, seed))
        path = new_result_path(model, seed)
        if path.exists():
            print(f"[{model}] reuse seed{seed}: EMS+I+E", flush=True)
            continue
        _fit(
            matrix=matrix,
            labels=labels,
            fit_mask=fit_mask,
            eval_mask=test_mask,
            seed=seed,
            root_config=config,
            device=torch.device(device_name),
            folder=path.parent,
        )
        print(f"[{model}] completed seed{seed}: EMS+I+E", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO：SAR+EMS 与 EMS+I+E 的同种子对比", "",
        f"LLaVA-1.5-7B、Qwen3-VL-8B；种子 {'/'.join(map(str, SEEDS))} 各自控制 3200/800 图片划分及 MLP 初始化。两组使用相同 object token、原始未优化的 K=32/τ=0.03 EMS、相同单隐藏层 MLP 与训练设置。SAR 是 F_log；I 为逐层视觉写入范数之和；E 为逐层全部视觉 token 的 final-norm 目标词概率之和。先按训练集统计量标准化每个输入维度。AUROC/Real AUPR/Hall AUPR 均在测试集计算。", "",
        "| 模型 | 组合 | AUROC (%) | Real AUPR (%) | Hall AUPR (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        by_group = {}
        for group in ("EMS+SAR", "EMS+I+E"):
            metrics = []
            for seed in SEEDS:
                path = original_result_path(model, seed) if group == "EMS+SAR" else new_result_path(model, seed)
                value = score_metrics(json.loads(path.read_text()))
                metrics.append(value)
                rows.append({"model": model, "group": group, "seed": seed, **value})
            by_group[group] = metrics
            fmt = lambda key: f"{100*mean(v[key] for v in metrics):.2f} ± {100*pstdev(v[key] for v in metrics):.2f}"
            lines.append(f"| {model} | {group} | {fmt('auroc')} | {fmt('real_aupr')} | {fmt('hall_aupr')} |")
        deltas = {key: mean(100*(new[key]-old[key]) for old,new in zip(by_group["EMS+SAR"],by_group["EMS+I+E"])) for key in ("auroc","real_aupr","hall_aupr")}
        lines.append(f"| {model} | Δ (EMS+I+E − EMS+SAR) | {deltas['auroc']:+.2f} | {deltas['real_aupr']:+.2f} | {deltas['hall_aupr']:+.2f} |")
    lines.extend(["", "## 逐种子配对差值（EMS+I+E − EMS+SAR，百分点）", "", "| 模型 | 种子 | ΔAUROC | ΔReal AUPR | ΔHall AUPR |", "|---|---:|---:|---:|---:|"])
    for model in MODELS:
        for seed in SEEDS:
            old = next(r for r in rows if r["model"] == model and r["group"] == "EMS+SAR" and r["seed"] == seed)
            new = next(r for r in rows if r["model"] == model and r["group"] == "EMS+I+E" and r["seed"] == seed)
            lines.append(f"| {model} | {seed} | {100*(new['auroc']-old['auroc']):+.2f} | {100*(new['real_aupr']-old['real_aupr']):+.2f} | {100*(new['hall_aupr']-old['hall_aupr']):+.2f} |")
    lines.extend(["", "完整逐种子数值见 `per_seed.json`；标准差为三个种子的总体标准差。", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    (OUT / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed-set", choices=("42-44", "43-45"), default="42-44")
    args = parser.parse_args()
    configure(args.seed_set)
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model is required unless --summarize is set")


if __name__ == "__main__":
    main()
