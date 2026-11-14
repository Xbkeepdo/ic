#!/usr/bin/env python3
"""Paired image bootstrap: fixed OT versus six JS variants with P_tau=0.07."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bootstrap_log_s_ot_js_seeded_811 as base  # noqa: E402


OUT = ROOT / "outputs/log_s_js_tau007_variants_seeded_811_43_45"
METHODS = {
    "ot_with_rq": ("sar_ems", base.REFERENCE),
    "union16": ("sar_js_union16", OUT),
    "union16_rq": ("sar_js_union16_rq", OUT),
    "union32": ("sar_js_union32", OUT),
    "union32_rq": ("sar_js_union32_rq", OUT),
    "rank16": ("sar_js_rank16", OUT),
    "rank16_rq": ("sar_js_rank16_rq", OUT),
}
PAIRS = {name: ("ot_with_rq", name) for name in METHODS if name != "ot_with_rq"}
DISPLAY = {
    "union16": "Top16(P)∪Top16(T), JS",
    "union16_rq": "Top16(P)∪Top16(T), JS+R_q",
    "union32": "Top32(P)∪Top32(T), JS",
    "union32_rq": "Top32(P)∪Top32(T), JS+R_q",
    "rank16": "独立 Top16 排名 JS",
    "rank16_rq": "独立 Top16 排名 JS+R_q",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch size must be positive")
    base.METHODS = METHODS
    base.PAIRS = PAIRS
    report_dir = OUT / f"bootstrap_{args.iterations}"
    rng = np.random.default_rng(args.seed)
    rows, samples, union_sizes = [], {}, {}
    for model in base.MODELS:
        model_rows, model_samples, union_sizes[model] = base.bootstrap_model(
            model, iterations=args.iterations, batch_size=args.batch_size, rng=rng,
        )
        rows.extend(model_rows)
        samples.update(model_samples)
        print(f"[{model}] bootstrap complete", flush=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "bootstrap_auroc_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(report_dir / "bootstrap_auroc_samples.npz", **samples)
    lines = [
        "# COCO 8:1:1：OT 与 tau=0.07 JS 的配对逐图 Bootstrap", "",
        "COCO 8:1:1，种子 43/44/45。OT 与 R_q 固定为 tau=0.03 原设置，JS 源分布取缓存 P_tau=0.07，目标分布 T 不变；比较三种 JS 支持集，各自测试原 JS 与 JS+R_q。差值是 OT−JS 的 real-positive AUROC 百分点。", "",
        f"每模型 {args.iterations:,} 次配对图片重采样，随机种子 {args.seed}；95% 百分位 CI，未做六方案多重比较校正。同一图片所有 object token 同时重采样；三种子合并 CI 使用三个测试集图片并集的一次共同抽样，保留测试图像重叠。CI 条件于已训练的检测器和固定划分，不包含重新训练的不确定性。", "",
        "| 模型 | JS 构造 | OT AUROC | JS AUROC | Δ(OT−JS) | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in base.MODELS:
        for comparison in PAIRS:
            row = next(r for r in rows if r["model"] == model and r["comparison"] == comparison and r["scope"] == "three_seed_mean")
            lines.append(
                f"| {model} | {DISPLAY[comparison]} | {100*row['ot_auc']:.2f} | "
                f"{100*row['js_auc']:.2f} | {100*row['delta_auc']:+.2f} | "
                f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
            )
    lines += ["", "## 逐种子差值与 CI", "",
              "| 模型 | JS 构造 | 种子 | Δ(OT−JS) | 95% CI |",
              "|---|---|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "three_seed_mean":
            continue
        lines.append(
            f"| {row['model']} | {DISPLAY[row['comparison']]} | {row['scope']} | "
            f"{100*row['delta_auc']:+.2f} | "
            f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
        )
    lines += ["", "## JS 构造之间的配对差值", "",
              "差值为左侧 JS−右侧 JS 的 real-positive AUROC 百分点。由同一轮图片抽样的 OT−JS 差值相减得到；CI 保留配对关系。", "",
              "| 模型 | JS 对比 | AUROC 差值 | 95% CI |",
              "|---|---|---:|---:|"]
    direct_pairs = (
        ("union32", "union16", "Top32 并集−Top16 并集"),
        ("union32_rq", "union16_rq", "Top32 并集+R_q−Top16 并集+R_q"),
        ("union32", "rank16", "Top32 并集−Top16 排名"),
        ("union32_rq", "rank16_rq", "Top32 并集+R_q−Top16 排名+R_q"),
    )
    for model in base.MODELS:
        for left, right, label in direct_pairs:
            left_row = next(r for r in rows if r["model"] == model and r["comparison"] == left and r["scope"] == "three_seed_mean")
            right_row = next(r for r in rows if r["model"] == model and r["comparison"] == right and r["scope"] == "three_seed_mean")
            draw = samples[f"{model}_{right}_three_seed_mean"] - samples[f"{model}_{left}_three_seed_mean"]
            low, high = np.quantile(draw, (0.025, 0.975))
            delta = right_row["delta_auc"] - left_row["delta_auc"]
            shown = 0.0 if abs(100 * delta) < 0.005 else 100 * delta
            lines.append(f"| {model} | {label} | {shown:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] |")
    lines += ["", "各模型三种子测试图像并集数量：" + "；".join(f"{m}={n}" for m, n in union_sizes.items()) + "。未四舍五入数值与每次抽样结果见 CSV、NPZ。", ""]
    (report_dir / "bootstrap_auroc_ci.md").write_text("\n".join(lines))
    print(report_dir / "bootstrap_auroc_ci.md", flush=True)


if __name__ == "__main__":
    main()
