#!/usr/bin/env python3
"""Paired image bootstrap of OT minus rank-top16 JS+R_q on COCO seeded 8:2."""

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


OUT = ROOT / "outputs/log_s_ot_rank16_rq_seeded_82_43_45"
SPLITS = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr/splits"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)


def load_model(model: str) -> tuple[np.ndarray, dict[int, dict]]:
    loaded = {}
    for seed in SEEDS:
        split = json.loads((SPLITS / f"seed{seed}.json").read_text())
        test_ids = np.asarray(split["test"], dtype=np.int64)
        if len(test_ids) != 800 or len(np.unique(test_ids)) != 800:
            raise ValueError(f"seed{seed}: expected 800 distinct test images")
        with np.load(OUT / model / f"seed{seed}" / "paired_scores.npz", allow_pickle=False) as saved:
            image_ids = np.asarray(saved["image_id"], dtype=np.int64)
            labels = np.asarray(saved["label"], dtype=np.int8)
            scores = {
                "ot": np.asarray(saved["ot_real_score"], dtype=np.float64),
                "js": np.asarray(saved["js_real_score"], dtype=np.float64),
            }
        if not np.isin(image_ids, test_ids).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{model}/seed{seed}: invalid test records")
        if any(len(values) != len(labels) or not np.isfinite(values).all() for values in scores.values()):
            raise ValueError(f"{model}/seed{seed}: invalid test scores")
        curves = {name: base.prepare_score(values, labels) for name, values in scores.items()}
        point_auc = {
            name: float(base.weighted_auc(np.ones((1, len(labels))), labels, *curves[name])[0])
            for name in scores
        }
        audit = json.loads((OUT / model / f"seed{seed}" / "audit.json").read_text())
        if len(labels) != audit["test_tokens"] or abs(point_auc["ot"] - audit["ot_auc_recovered"]) > 1e-10 or abs(point_auc["js"] - audit["js_auc_recovered"]) > 1e-10:
            raise ValueError(f"{model}/seed{seed}: bootstrap score audit differs")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch size must be positive")
    base.SEEDS = SEEDS
    base.METHODS = {"ot": ("ot", OUT), "js": ("js", OUT)}
    base.PAIRS = {"rank16_rq": ("ot", "js")}
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
        "# COCO 8:2：OT 与 Top16 排名 JS+R_q 的配对逐图 Bootstrap", "",
        f"种子 43/44/45；每种子 3200 训练图片、800 测试图片。固定已训练检测器和图片划分，对每模型执行 {args.iterations:,} 次图片级配对重采样。同一图片全部 object token 使用相同权重；跨种子重复出现的测试图片使用同一轮抽样权重。差值为 OT−JS 的 real-positive AUROC 百分点；95% CI 是 2.5%/97.5% 百分位数，不包含重新训练的不确定性。", "",
        "| 模型 | 范围 | 测试图片 | 测试 token | OT AUROC (%) | JS+R_q AUROC (%) | OT−JS (pp) | 95% CI (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = 100 * row["delta_auc"]
        if abs(delta) < 0.005:
            delta = 0.0
        lines.append(
            f"| {row['model']} | {row['scope']} | {row['n_test_images']} | {row['n_test_tokens']} | "
            f"{100*row['ot_auc']:.2f} | {100*row['js_auc']:.2f} | {delta:+.2f} | "
            f"[{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
        )
    lines += ["", "三种子均值的测试图片数量是三个测试集的去重并集；测试 token 数是逐种子出现次数之和。", ""]
    (folder / "bootstrap_auroc_ci.md").write_text("\n".join(lines))
    print(folder / "bootstrap_auroc_ci.md", flush=True)


if __name__ == "__main__":
    main()
