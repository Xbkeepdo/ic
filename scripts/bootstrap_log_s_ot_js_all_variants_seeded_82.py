#!/usr/bin/env python3
"""Paired image bootstrap for all six JS variants versus OT on seeded COCO 8:2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bootstrap_log_s_ot_js_seeded_811 as base  # noqa: E402
from scripts.run_log_s_ot_js_all_variants_seeded_82 import DISPLAY, METHODS, MODELS, SEEDS  # noqa: E402

OUT = ROOT / "outputs/log_s_ot_js_all_variants_seeded_82_43_45"

SPLITS = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr/splits"
JS_CASES = METHODS[1:]
PAIRS = {case: ("ot", case) for case in JS_CASES}


def load_model(model: str) -> tuple[np.ndarray, dict[int, dict]]:
    loaded = {}
    for seed in SEEDS:
        split = json.loads((SPLITS / f"seed{seed}.json").read_text())
        test_ids = np.asarray(split["test"], dtype=np.int64)
        if len(test_ids) != 800 or len(np.unique(test_ids)) != 800:
            raise ValueError(f"seed{seed}: invalid test image count")
        with np.load(OUT / model / f"seed{seed}" / "paired_scores.npz", allow_pickle=False) as saved:
            image_ids = np.asarray(saved["image_id"], dtype=np.int64)
            labels = np.asarray(saved["label"], dtype=np.int8)
            scores = {case: np.asarray(saved[f"{case}_real_score"], dtype=np.float64) for case in METHODS}
        if not np.isin(image_ids, test_ids).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{model}/seed{seed}: invalid token-image alignment")
        if any(len(values) != len(labels) or not np.isfinite(values).all() for values in scores.values()):
            raise ValueError(f"{model}/seed{seed}: invalid scores")
        curves = {case: base.prepare_score(values, labels) for case, values in scores.items()}
        point_auc = {
            case: float(base.weighted_auc(np.ones((1, len(labels))), labels, *curves[case])[0])
            for case in METHODS
        }
        metrics = json.loads((OUT / model / f"seed{seed}" / "metrics.json").read_text())
        if len(labels) != metrics["test_tokens"] or any(
            abs(point_auc[case] - metrics["methods"][case]["auroc"]) > 1e-10 for case in METHODS
        ):
            raise ValueError(f"{model}/seed{seed}: score AUROC differs from saved result")
        loaded[seed] = {
            "test_ids": test_ids, "image_ids": image_ids,
            "labels": labels, "curves": curves, "point_auc": point_auc,
        }
    union = np.unique(np.concatenate([loaded[seed]["test_ids"] for seed in SEEDS]))
    for seed in SEEDS:
        indices = np.searchsorted(union, loaded[seed]["image_ids"])
        if not np.array_equal(union[indices], loaded[seed]["image_ids"]):
            raise ValueError("image-index alignment failed")
        loaded[seed]["image_indices"] = indices
    return union, loaded


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--source-dir", type=Path, default=OUT)
    parser.add_argument("--js-tau", type=float, default=0.03)
    args = parser.parse_args()
    OUT = args.source_dir.resolve()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch size must be positive")
    base.SEEDS = SEEDS
    base.METHODS = {case: (case, OUT) for case in METHODS}
    base.PAIRS = PAIRS
    base.load_model = load_model
    rng = np.random.default_rng(args.seed)
    rows, samples = [], {}
    for model in MODELS:
        result, draws, _ = base.bootstrap_model(
            model, iterations=args.iterations, batch_size=args.batch_size, rng=rng,
        )
        for row in result:
            if row["scope"] != "three_seed_mean":
                row["n_test_images"] = 800
        rows.extend(result)
        samples.update(draws)
        print(f"{model}: bootstrap complete", flush=True)
    folder = OUT / f"bootstrap_{args.iterations}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "bootstrap_auroc_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(folder / "bootstrap_auroc_samples.npz", **samples)
    lines = [
        f"# COCO 8:2：OT 与全部六种 JS（JS tau={args.js_tau:.2f}）的配对逐图 Bootstrap", "",
        f"种子 43/44/45，每种子 3200 训练图片、800 测试图片。每模型 {args.iterations:,} 次图片级配对重采样，同一图片全部 object token 一起抽样；跨种子重复出现的图片共享权重。差值为 OT−JS 的 real-positive AUROC 百分点。95% CI 为未经六方案多重比较校正的百分位区间，条件于固定检测器及划分，不包括重新训练的不确定性。OT 和 R_q 均固定在原 tau=0.03 特征，只有 JS 的 P 温度取 tau={args.js_tau:.2f}。", "",
        "| 模型 | JS 构造 | OT AUROC | JS AUROC | OT−JS (pp) | 95% CI (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for case in JS_CASES:
            row = next(r for r in rows if r["model"] == model and r["comparison"] == case and r["scope"] == "three_seed_mean")
            delta = 100 * row["delta_auc"]
            if abs(delta) < 0.005:
                delta = 0.0
            lines.append(
                f"| {model} | {DISPLAY[case]} | {100*row['ot_auc']:.2f} | {100*row['js_auc']:.2f} | "
                f"{delta:+.2f} | [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
            )
    lines += ["", "## 逐种子结果", "",
              "| 模型 | JS 构造 | 种子 | 测试图像 | 测试 token | OT−JS (pp) | 95% CI (pp) |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["scope"] == "three_seed_mean":
            continue
        lines.append(
            f"| {row['model']} | {DISPLAY[row['comparison']]} | {row['scope']} | "
            f"{row['n_test_images']} | {row['n_test_tokens']} | {100*row['delta_auc']:+.2f} | "
            f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
        )
    lines += ["", "每种子原始测试分数见各模型 seed*/paired_scores.npz；所有未四舍五入的 Bootstrap 结果见 CSV 与 NPZ。", ""]
    (folder / "bootstrap_auroc_ci.md").write_text("\n".join(lines))
    print(folder / "bootstrap_auroc_ci.md", flush=True)


if __name__ == "__main__":
    main()
