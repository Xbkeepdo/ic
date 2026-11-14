#!/usr/bin/env python3
"""Paired image bootstrap for tau=0.03 Top8 JS versus OT and tau=0.07 Top8 JS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bootstrap_log_s_ot_js_seeded_811 as base

OUT = ROOT / "outputs/log_s_js_tau003_top8_seeded_811_43_45"
PRIOR07 = ROOT / "outputs/log_s_js_tau007_top8_seeded_811_82_43_45/811"
METHODS = {
    "ot": ("sar_ems", base.REFERENCE),
    "js03": ("sar_js_union8", OUT),
    "js07": ("sar_js_union8", PRIOR07),
}
PAIRS = {
    "ot_minus_js03": ("ot", "js03"),
    "js03_minus_js07": ("js03", "js07"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch size must be positive")
    base.METHODS = METHODS
    base.PAIRS = PAIRS
    rng = np.random.default_rng(args.seed)
    rows, samples = [], {}
    for model in base.MODELS:
        result, draws, _ = base.bootstrap_model(
            model, iterations=args.iterations, batch_size=args.batch_size, rng=rng,
        )
        rows.extend(result)
        samples.update(draws)
        print(f"{model} bootstrap done", flush=True)
    folder = OUT / f"bootstrap_{args.iterations}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "bootstrap_auroc_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(folder / "bootstrap_auroc_samples.npz", **samples)
    lines = [
        "# COCO 8:1:1：Top8 JS tau=0.03 配对显著性",
        "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B，种子 43/44/45，每种子测试 400 张图片。OT 固定为原 K=32、tau=0.03、含 R_q 的 SAR+EMS；Top8 JS 不加 R_q。JS 的 tau=0.03 与 tau=0.07 两版本共用 T、SAR、训练配置、图片划分和 object token。tau=0.03 的 P 从缓存 tau=0.07 的 P 按幂变换重建。",
        "",
        f"{args.iterations:,} 次配对图片级 Bootstrap；差值为 real-positive AUROC 百分点。95% 百分位 CI 未做两项比较的多重校正。同一图片全部 object token 一起抽样；三种子测试集重叠的图片共享抽样权重。区间条件于固定检测器和划分，不包含重新训练的不确定性。",
        "",
        "| 模型 | 对比（左−右） | 左侧 AUROC | 右侧 AUROC | 差值 (pp) | 95% CI (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in base.MODELS:
        for pair, label in (
            ("ot_minus_js03", "OT+R_q − Top8 JS tau=0.03"),
            ("js03_minus_js07", "Top8 JS tau=0.03 − tau=0.07"),
        ):
            row = next(r for r in rows if r["model"] == model and r["comparison"] == pair and r["scope"] == "three_seed_mean")
            lines.append(
                f"| {model} | {label} | {100*row['ot_auc']:.2f} | {100*row['js_auc']:.2f} | "
                f"{100*row['delta_auc']:+.2f} | [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
            )
    lines += ["", "## 逐种子结果", "",
              "| 模型 | 对比 | 种子 | 测试图片 | 测试 token | 差值 (pp) | 95% CI (pp) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "three_seed_mean":
            continue
        lines.append(
            f"| {row['model']} | {row['comparison']} | {row['scope']} | {row['n_test_images']} | "
            f"{row['n_test_tokens']} | {100*row['delta_auc']:+.2f} | "
            f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
        )
    lines += ["", "未四舍五入数值见 CSV；逐轮差值见 NPZ。", ""]
    (folder / "bootstrap_auroc_ci.md").write_text("\n".join(lines))
    print(folder / "bootstrap_auroc_ci.md", flush=True)


if __name__ == "__main__":
    main()
