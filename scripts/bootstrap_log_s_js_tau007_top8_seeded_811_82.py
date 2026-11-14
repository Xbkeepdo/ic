#!/usr/bin/env python3
"""Paired image bootstrap for tau=0.07 Top8 union JS versus fixed OT, 8:1:1 and 8:2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bootstrap_log_s_ot_js_seeded_811 as base

OUT = ROOT / "outputs/log_s_js_tau007_top8_seeded_811_82_43_45"
REF_811 = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
PRIOR_811 = ROOT / "outputs/log_s_js_tau007_variants_seeded_811_43_45"
PRIOR_82 = ROOT / "outputs/log_s_ot_js_tau007_seeded_82_43_45"
SPLITS_82 = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr/splits"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
CASES = ("ot", "union8", "union8_rq", "union16", "union16_rq")
PAIRS = {
    "union8": ("ot", "union8"),
    "union8_rq": ("ot", "union8_rq"),
    "top8_minus_top16": ("union8", "union16"),
    "top8_rq_minus_top16_rq": ("union8_rq", "union16_rq"),
}


def load_82(model: str):
    loaded = {}
    for seed in SEEDS:
        split = json.loads((SPLITS_82 / f"seed{seed}.json").read_text())
        test_ids = np.asarray(split["test"], dtype=np.int64)
        if len(test_ids) != 800 or len(np.unique(test_ids)) != 800:
            raise ValueError(f"seed{seed}: invalid 8:2 test images")
        folder = OUT / "82" / model / f"seed{seed}"
        with np.load(folder / "paired_scores.npz", allow_pickle=False) as data:
            image_ids = np.asarray(data["image_id"], dtype=np.int64)
            labels = np.asarray(data["label"], dtype=np.int8)
            scores = {case: np.asarray(data[f"{case}_real_score"], dtype=np.float64) for case in ("ot", "union8", "union8_rq")}
            response_idx = np.asarray(data["response_token_idx"])
            target_ids = np.asarray(data["target_token_id"])
        with np.load(PRIOR_82 / model / f"seed{seed}" / "paired_scores.npz", allow_pickle=False) as prior:
            if any(not np.array_equal(prior[name], value) for name, value in (
                ("image_id", image_ids), ("label", labels),
                ("response_token_idx", response_idx), ("target_token_id", target_ids),
            )):
                raise ValueError(f"{model}/seed{seed}: Top16 and Top8 test tokens differ")
            scores.update({case: np.asarray(prior[f"{case}_real_score"], dtype=np.float64) for case in ("union16", "union16_rq")})
        if not np.isin(image_ids, test_ids).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{model}/seed{seed}: invalid test alignment")
        curves = {case: base.prepare_score(scores[case], labels) for case in CASES}
        point_auc = {
            case: float(base.weighted_auc(np.ones((1, len(labels))), labels, *curves[case])[0])
            for case in CASES
        }
        metrics = json.loads((folder / "metrics.json").read_text())
        prior_metrics = json.loads((PRIOR_82 / model / f"seed{seed}" / "metrics.json").read_text())
        if len(labels) != metrics["test_tokens"] or any(
            abs(point_auc[case] - (metrics if case in ("ot", "union8", "union8_rq") else prior_metrics)["methods"][case]["auroc"]) > 1e-8
            for case in CASES
        ):
            raise ValueError(f"{model}/seed{seed}: recovered score AUROC differs")
        loaded[seed] = {
            "test_ids": test_ids, "image_ids": image_ids,
            "labels": labels, "curves": curves, "point_auc": point_auc,
        }
    union = np.unique(np.concatenate([loaded[seed]["test_ids"] for seed in SEEDS]))
    for seed in SEEDS:
        indices = np.searchsorted(union, loaded[seed]["image_ids"])
        if not np.array_equal(union[indices], loaded[seed]["image_ids"]):
            raise ValueError(f"{model}/seed{seed}: union image index mismatch")
        loaded[seed]["image_indices"] = indices
    return union, loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch size must be positive")
    base.SEEDS = SEEDS
    base.PAIRS = PAIRS
    original_loader = base.load_model
    rng = np.random.default_rng(args.seed)
    rows, samples = [], {}
    for split in ("811", "82"):
        if split == "811":
            base.METHODS = {
                "ot": ("sar_ems", REF_811),
                "union8": ("sar_js_union8", OUT / "811"),
                "union8_rq": ("sar_js_union8_rq", OUT / "811"),
                "union16": ("sar_js_union16", PRIOR_811),
                "union16_rq": ("sar_js_union16_rq", PRIOR_811),
            }
            base.load_model = original_loader
        else:
            base.METHODS = {case: (case, OUT / "82") for case in CASES}
            base.load_model = load_82
        for model in MODELS:
            result, draws, _ = base.bootstrap_model(
                model, iterations=args.iterations, batch_size=args.batch_size, rng=rng,
            )
            for row in result:
                row["split"] = split
                if split == "82" and row["scope"] != "three_seed_mean":
                    row["n_test_images"] = 800
            rows.extend(result)
            samples.update({f"{split}_{name}": values for name, values in draws.items()})
            print(f"{split} {model} bootstrap done", flush=True)
    folder = OUT / f"bootstrap_{args.iterations}"
    folder.mkdir(parents=True, exist_ok=True)
    columns = ("split", "model", "comparison", "scope", "n_test_images", "n_test_tokens", "ot_auc", "js_auc", "delta_auc", "ci_low", "ci_high")
    with (folder / "bootstrap_auroc_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(folder / "bootstrap_auroc_samples.npz", **samples)
    lines = [
        "# COCO：固定 OT vs tau=0.07 Top8 并集 JS",
        "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B，种子 43/44/45 同时控制图片划分和 MLP 初始化。8:1:1 每种子测试 400 张图片，8:2 每种子测试 800 张图片。JS 在 Top8(P)∪Top8(T) 上分别重归一化后计算自然对数 JS；测试 JS 与 JS+R_q。OT 与 R_q 固定为原 K=32、tau=0.03；JS 的源分布 P 取 tau=0.07，T 与 SAR 不变。",
        "",
        f"每模型/划分做 {args.iterations:,} 次配对图片级 Bootstrap。差值为 real-positive AUROC 百分点。95% 百分位 CI 未做四方案多重比较校正。跨种子重复出现的测试图片共用抽样权重。区间条件于已训练的固定检测器及三次划分，不包含重新训练的不确定性。",
        "",
        "| 划分 | 模型 | JS 方案 | OT AUROC | JS AUROC | OT−JS (pp) | 95% CI (pp) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for split in ("811", "82"):
        for model in MODELS:
            for case in ("union8", "union8_rq"):
                row = next(r for r in rows if r["split"] == split and r["model"] == model and r["comparison"] == case and r["scope"] == "three_seed_mean")
                lines.append(
                    f"| {split} | {model} | {'Top8 并集 + R_q' if case.endswith('_rq') else 'Top8 并集'} | "
                    f"{100*row['ot_auc']:.2f} | {100*row['js_auc']:.2f} | "
                    f"{100*row['delta_auc']:+.2f} | [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
                )
    lines += ["", "## Top8 与 Top16 的直接配对比较", "",
              "差值为 Top8−Top16，正值表示 Top8 更高；两方案使用相同的 tau=0.07、图片划分与模型训练设置。", "",
              "| 划分 | 模型 | 方案 | Top8 AUROC | Top16 AUROC | Top8−Top16 (pp) | 95% CI (pp) |",
              "|---|---|---|---:|---:|---:|---:|"]
    for split in ("811", "82"):
        for model in MODELS:
            for case in ("top8_minus_top16", "top8_rq_minus_top16_rq"):
                row = next(r for r in rows if r["split"] == split and r["model"] == model and r["comparison"] == case and r["scope"] == "three_seed_mean")
                lines.append(
                    f"| {split} | {model} | {'JS + R_q' if case.endswith('rq') else 'JS'} | "
                    f"{100*row['ot_auc']:.2f} | {100*row['js_auc']:.2f} | "
                    f"{100*row['delta_auc']:+.2f} | [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
                )
    lines += ["", "## 逐种子结果", "",
              "| 划分 | 模型 | 比较（左−右） | 种子 | 测试图片 | 测试 token | 差值 (pp) | 95% CI (pp) |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "three_seed_mean":
            continue
        lines.append(
            f"| {row['split']} | {row['model']} | {row['comparison']} | {row['scope']} | "
            f"{row['n_test_images']} | {row['n_test_tokens']} | {100*row['delta_auc']:+.2f} | "
            f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
        )
    lines += ["", "原始未四舍五入结果见 CSV；每次重采样差值见 NPZ。", ""]
    (folder / "bootstrap_auroc_ci.md").write_text("\n".join(lines))
    print(folder / "bootstrap_auroc_ci.md", flush=True)


if __name__ == "__main__":
    main()
