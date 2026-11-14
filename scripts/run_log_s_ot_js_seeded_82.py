#!/usr/bin/env python3
"""Compare cached Log-S OT and JS features on seeded COCO image splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (42, 43, 44)
REFERENCE_OUT = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
PRIOR_OUT = ROOT / "outputs/log_s_ot_js_seeded_82"
OUT = ROOT / "outputs/log_s_ot_js_seeded_82_42_44"
VARIANTS = (
    "sar_only",
    "ot_only",
    "js_only",
    "sar_ot",
    "sar_js",
    "sar_geometry_ot",
)
DISPLAY = {
    "sar_only": "SAR only",
    "ot_only": "EMS (OT) only",
    "js_only": "JS only",
    "sar_ot": "SAR + EMS (OT)",
    "sar_js": "SAR + JS",
    "sar_geometry_ot": "SAR + geometry OT",
}


def load_model(model: str) -> dict[str, np.ndarray]:
    path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    local_name = "llava" if model == "llava_1_5_7b" else "qwen3"
    local = Path("/tmp/log_s_ot_js_seeded_82") / f"{local_name}_compact_features.npz"
    if local.exists() and local.stat().st_size == path.stat().st_size:
        path = local
    print(f"[{model}] loading {path}", flush=True)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key]) for key in (
            "image_id", "response_token_idx", "target_token_id", "label",
            "sar", "ems", "sar_ems", "sar_union_js", "sar_ot_geometry",
        )}
    n = arrays["ems"].shape[1]
    if not np.array_equal(arrays["sar_ems"][:, :n], arrays["sar"]):
        raise ValueError(f"{model}: SAR/OT feature order changed")
    if not np.array_equal(arrays["sar_ems"][:, n:], arrays["ems"]):
        raise ValueError(f"{model}: OT feature changed")
    if not np.array_equal(arrays["sar_union_js"][:, :n], arrays["sar"]):
        raise ValueError(f"{model}: SAR/JS feature order changed")
    if not np.array_equal(arrays["sar_ot_geometry"][:, :n], arrays["sar"]):
        raise ValueError(f"{model}: SAR/geometry OT feature order changed")
    if any(not np.isfinite(arrays[key]).all() for key in (
        "sar", "ems", "sar_union_js", "sar_ot_geometry",
    )):
        raise ValueError(f"{model}: nonfinite feature")
    return {
        "image_id": arrays["image_id"],
        "label": arrays["label"],
        "sar_only": arrays["sar"],
        "ot_only": arrays["ems"],
        "js_only": arrays["sar_union_js"][:, n:],
        # Preserve the exact original Ours column order: EMS, then SAR.
        "sar_ot": np.concatenate((arrays["ems"], arrays["sar"]), axis=1),
        "sar_js": np.concatenate((arrays["sar_union_js"][:, n:], arrays["sar"]), axis=1),
        "sar_geometry_ot": np.concatenate((arrays["sar_ot_geometry"][:, n:], arrays["sar"]), axis=1),
    }


def result_path(model: str, variant: str, seed: int) -> Path:
    if variant == "sar_ot" and seed in (43, 44):
        return REFERENCE_OUT / model / f"seed{seed}" / "ours" / "metrics.json"
    if seed in (43, 44):
        return PRIOR_OUT / model / variant / f"seed{seed}" / "metrics.json"
    return OUT / model / variant / f"seed{seed}" / "metrics.json"


def score_metrics(saved: dict) -> dict[str, float]:
    test = saved["threshold_reports"]["train_f1"]["test_metrics"]
    return {
        "auroc": float(test["real_positive"]["auc"]),
        "real_aupr": float(test["real_positive"]["aupr"]),
        "hall_aupr": float(test["hallucination_positive"]["aupr"]),
    }


def load_splits() -> dict[int, dict[str, list[int]]]:
    splits = {}
    reference = json.loads((REFERENCE_OUT / "splits" / "seed43.json").read_text())
    pool = np.asarray(sorted(set(reference["train"] + reference["test"])), dtype=np.int64)
    if len(pool) != 4000:
        raise ValueError("reference image pool must contain 4000 unique IDs")
    for seed in SEEDS:
        if seed == 42:
            shuffled = np.random.default_rng(seed).permutation(pool)
            split = {
                "train": np.sort(shuffled[:3200]).astype(int).tolist(),
                "val": [],
                "test": np.sort(shuffled[3200:]).astype(int).tolist(),
            }
        else:
            split = json.loads((REFERENCE_OUT / "splits" / f"seed{seed}.json").read_text())
        train, test = set(split["train"]), set(split["test"])
        if len(train) != 3200 or len(test) != 800 or train & test or train | test != set(pool):
            raise ValueError(f"seed{seed}: invalid reference image split")
        path = OUT / "splits" / f"seed{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != split:
            raise ValueError(f"seed{seed}: existing new split differs")
        path.write_text(json.dumps(split, indent=2) + "\n")
        splits[seed] = split
    return splits


def run_model(model: str, device_name: str) -> None:
    import torch

    from scripts.train_log_s_hpre_k_tau_ablation import _fit
    from utils.config_utils import load_config

    torch.set_num_threads(1)
    print(f"[{model}] starting", flush=True)
    data = load_model(model)
    print(f"[{model}] loaded {len(data['label'])} object tokens", flush=True)
    splits = load_splits()
    print(f"[{model}] loaded seeded splits", flush=True)
    cfg = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for seed in SEEDS:
        split = splits[seed]
        train_mask = np.isin(data["image_id"], split["train"])
        test_mask = np.isin(data["image_id"], split["test"])
        if not np.all(train_mask | test_mask) or np.any(train_mask & test_mask):
            raise ValueError(f"{model}/seed{seed}: invalid split")
        for variant in VARIANTS:
            if variant == "sar_ot" and seed in (43, 44):
                if not result_path(model, variant, seed).exists():
                    raise FileNotFoundError(result_path(model, variant, seed))
                continue
            path = result_path(model, variant, seed)
            if path.exists():
                print(f"[{model}] skip {variant} seed{seed}", flush=True)
                continue
            _fit(
                matrix=data[variant], labels=data["label"],
                fit_mask=train_mask, eval_mask=test_mask,
                seed=seed, root_config=cfg,
                device=torch.device(device_name), folder=path.parent,
            )
            print(f"[{model}] completed {variant} seed{seed}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO：随机图片划分及初始化下的 OT 与 JS 对比", "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B；三个种子 42/43/44 各自决定 3200/800 图片划分及 MLP 初始化。每个种子内，所有方法使用相同训练/测试 object token、相同单隐藏层 MLP（宽度 128）和训练设置。复用原始未优化的 K=32、τ=0.03 Log-S 特征。OT 为完整 EMS；JS 在相同源/目标分布的 top-K 并集上分别归一化后计算自然对数 JS。43/44 沿用相同划分与设置的既有训练结果，42 重新训练。AUPR 分别以 real、hall 为正类。", "",
        "| 模型 | 特征 | AUROC (%) | Real AUPR (%) | Hall AUPR (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for variant in VARIANTS:
            values = []
            for seed in SEEDS:
                path = result_path(model, variant, seed)
                result = score_metrics(json.loads(path.read_text()))
                values.append(result)
                rows.append({"model": model, "variant": variant, "seed": seed, **result})
            fmt = lambda metric: f"{100*mean(v[metric] for v in values):.2f} ± {100*pstdev(v[metric] for v in values):.2f}"
            lines.append(f"| {model} | {DISPLAY[variant]} | {fmt('auroc')} | {fmt('real_aupr')} | {fmt('hall_aupr')} |")
    lines.extend(["", "## 配对差值：OT − JS（百分点）", "", "| 模型 | 对比 | ΔAUROC | ΔReal AUPR | ΔHall AUPR |", "|---|---|---:|---:|---:|"])
    for model in MODELS:
        for left, right, label in (("sar_ot", "sar_js", "SAR+EMS vs SAR+JS"), ("ot_only", "js_only", "EMS vs JS 单独特征")):
            paired = [
                (next(row for row in rows if row["model"] == model and row["variant"] == left and row["seed"] == seed),
                 next(row for row in rows if row["model"] == model and row["variant"] == right and row["seed"] == seed))
                for seed in SEEDS
            ]
            deltas = [mean(100*(a[key]-b[key]) for a,b in paired) for key in ("auroc", "real_aupr", "hall_aupr")]
            lines.append(f"| {model} | {label} | {deltas[0]:+.2f} | {deltas[1]:+.2f} | {deltas[2]:+.2f} |")
    lines.extend(["", "## 每个种子的拼接对比", "", "| 模型 | 种子 | SAR+OT AUROC | SAR+JS AUROC | OT−JS | OT−JS Real AUPR | OT−JS Hall AUPR |", "|---|---:|---:|---:|---:|---:|---:|"])
    for model in MODELS:
        for seed in SEEDS:
            ot = next(row for row in rows if row["model"] == model and row["variant"] == "sar_ot" and row["seed"] == seed)
            js = next(row for row in rows if row["model"] == model and row["variant"] == "sar_js" and row["seed"] == seed)
            lines.append(
                f"| {model} | {seed} | {100*ot['auroc']:.2f} | {100*js['auroc']:.2f} | "
                f"{100*(ot['auroc']-js['auroc']):+.2f} | "
                f"{100*(ot['real_aupr']-js['real_aupr']):+.2f} | "
                f"{100*(ot['hall_aupr']-js['hall_aupr']):+.2f} |"
            )
    lines.extend(["", "`SAR + geometry OT` 仅去掉 EMS 的目标状态校准项，用于判断完整 OT 与 JS 的差异有多少来自该校准项；它不是单独的求解器消融。表中差值单位均为百分点。三个种子的逐项数值保存在 `per_seed.json`。", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    (OUT / "per_seed.json").write_text(json.dumps(rows, indent=2) + "\n")
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
