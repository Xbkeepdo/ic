#!/usr/bin/env python3
"""Paired image-level bootstrap for OT versus JS on seeded COCO 8:1:1 tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs/log_s_mechanism_ablation_811"
REFERENCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
OUT = ROOT / "outputs/log_s_ot_js_seeded_811_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METHODS = {
    "js_without_rq": ("sar_union_js", OUT),
    "ot_with_rq": ("sar_ems", REFERENCE),
    "js_with_rq": ("sar_js_rq", OUT),
}
PAIRS = {
    "without_rq": ("ot_with_rq", "js_without_rq"),
    "with_rq": ("ot_with_rq", "js_with_rq"),
}


def prepare_score(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    groups = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    return order, groups


def weighted_auc(
    weights: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Tie-correct weighted Mann-Whitney AUROC for many bootstrap draws."""
    ordered_weights = weights[:, order]
    positives = np.add.reduceat(ordered_weights * labels[order], groups, axis=1)
    negatives = np.add.reduceat(ordered_weights * (1 - labels[order]), groups, axis=1)
    negative_below = np.cumsum(negatives, axis=1) - negatives
    numerator = np.sum(positives * (negative_below + 0.5 * negatives), axis=1)
    denominator = np.sum(positives, axis=1) * np.sum(negatives, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def load_model(model: str) -> tuple[np.ndarray, dict[int, dict]]:
    with np.load(FEATURES / model / "compact_features.npz", allow_pickle=False) as data:
        image_ids = np.asarray(data["image_id"], dtype=np.int64)
        labels = np.asarray(data["label"], dtype=np.int8)
    loaded = {}
    for seed in SEEDS:
        split = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        test_ids = np.asarray(split["test"], dtype=np.int64)
        mask = np.isin(image_ids, test_ids)
        token_image_ids = image_ids[mask]
        token_labels = labels[mask]
        if len(test_ids) != 400 or not set(token_image_ids).issubset(set(test_ids)):
            raise ValueError(f"{model}/seed{seed}: invalid test image IDs")
        scores_by_method = {}
        curves = {}
        point_auc = {}
        for name, (case, root) in METHODS.items():
            folder = root / model / f"seed{seed}" / case
            with np.load(folder / "scores.npz", allow_pickle=False) as saved:
                method_labels = np.asarray(saved["test_label"], dtype=np.int8)
                real_score = 1.0 - np.asarray(saved["test_hall_score"], dtype=np.float64)
            if not np.array_equal(method_labels, token_labels) or not np.isfinite(real_score).all():
                raise ValueError(f"{model}/seed{seed}/{case}: test scores are not aligned")
            scores_by_method[name] = real_score
            curves[name] = prepare_score(real_score, token_labels)
            point_auc[name] = float(weighted_auc(
                np.ones((1, len(token_labels)), dtype=np.float64),
                token_labels, *curves[name],
            )[0])
            result = json.loads((folder / "result.json").read_text())
            if not np.isclose(point_auc[name], result["test"]["auroc"], atol=1e-10):
                raise ValueError(f"{model}/seed{seed}/{case}: score AUROC differs from result")
        loaded[seed] = {
            "test_ids": test_ids,
            "image_ids": token_image_ids,
            "labels": token_labels,
            "curves": curves,
            "point_auc": point_auc,
        }
    union_images = np.unique(np.concatenate([loaded[seed]["test_ids"] for seed in SEEDS]))
    for seed in SEEDS:
        indices = np.searchsorted(union_images, loaded[seed]["image_ids"])
        if not np.array_equal(union_images[indices], loaded[seed]["image_ids"]):
            raise AssertionError("image-index alignment failed")
        loaded[seed]["image_indices"] = indices
    return union_images, loaded


def bootstrap_model(
    model: str, *, iterations: int, batch_size: int, rng: np.random.Generator,
) -> tuple[list[dict], dict[str, np.ndarray], int]:
    union_images, data = load_model(model)
    bootstrap = {
        (pair, seed): np.empty(iterations, dtype=np.float64)
        for pair in PAIRS for seed in SEEDS
    }
    probabilities = np.full(len(union_images), 1.0 / len(union_images))
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        # One image draw is shared across methods and across overlapping test splits.
        image_weights = rng.multinomial(len(union_images), probabilities, size=stop - start)
        for seed in SEEDS:
            entry = data[seed]
            token_weights = image_weights[:, entry["image_indices"]].astype(np.float64)
            aucs = {
                name: weighted_auc(token_weights, entry["labels"], *entry["curves"][name])
                for name in METHODS
            }
            for pair, (ot_name, js_name) in PAIRS.items():
                bootstrap[(pair, seed)][start:stop] = aucs[ot_name] - aucs[js_name]
    summary = []
    samples = {}
    for pair, (ot_name, js_name) in PAIRS.items():
        for seed in SEEDS:
            values = bootstrap[(pair, seed)]
            if not np.isfinite(values).all():
                raise ValueError(f"{model}/{pair}/seed{seed}: degenerate bootstrap sample")
            point = data[seed]["point_auc"][ot_name] - data[seed]["point_auc"][js_name]
            low, high = np.quantile(values, (0.025, 0.975))
            summary.append({
                "model": model, "comparison": pair, "scope": f"seed{seed}",
                "n_test_images": 400, "n_test_tokens": len(data[seed]["labels"]),
                "ot_auc": data[seed]["point_auc"][ot_name],
                "js_auc": data[seed]["point_auc"][js_name],
                "delta_auc": point, "ci_low": float(low), "ci_high": float(high),
            })
            samples[f"{model}_{pair}_seed{seed}"] = values
        pooled = np.mean(np.stack([bootstrap[(pair, seed)] for seed in SEEDS]), axis=0)
        point = float(np.mean([
            data[seed]["point_auc"][ot_name] - data[seed]["point_auc"][js_name]
            for seed in SEEDS
        ]))
        low, high = np.quantile(pooled, (0.025, 0.975))
        summary.append({
            "model": model, "comparison": pair, "scope": "three_seed_mean",
            "n_test_images": len(union_images),
            "n_test_tokens": sum(len(data[seed]["labels"]) for seed in SEEDS),
            "ot_auc": float(np.mean([data[seed]["point_auc"][ot_name] for seed in SEEDS])),
            "js_auc": float(np.mean([data[seed]["point_auc"][js_name] for seed in SEEDS])),
            "delta_auc": point, "ci_low": float(low), "ci_high": float(high),
        })
        samples[f"{model}_{pair}_three_seed_mean"] = pooled
    return summary, samples, len(union_images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--report-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_size <= 0:
        parser.error("iterations and batch size must be positive")
    rows, samples, union_sizes = [], {}, {}
    rng = np.random.default_rng(args.seed)
    for model in MODELS:
        model_rows, model_samples, union_sizes[model] = bootstrap_model(
            model, iterations=args.iterations, batch_size=args.batch_size, rng=rng,
        )
        rows.extend(model_rows)
        samples.update(model_samples)
        print(f"[{model}] image-level bootstrap completed", flush=True)
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "bootstrap_auroc_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(report_dir / "bootstrap_auroc_samples.npz", **samples)
    lines = [
        "# OT 与 JS 加/不加 R_q：配对逐图 Bootstrap", "",
        "使用 COCO 种子联动 8:1:1 实验的 LLaVA-1.5-7B 和 Qwen3-VL-8B 测试分数；种子 43/44/45 各有 400 张测试图片。OT 一直为 SAR+OT_G+R_q；分别比较 SAR+JS（JS 不加 R_q）和 SAR+JS+R_q（JS 加 R_q）。指标为 real-positive AUROC；差值定义为 OT−JS，单位百分点。", "",
        f"Bootstrap 随机种子 {args.seed}；{args.iterations:,} 次百分位重采样，95% CI 为 2.5%/97.5% 分位数。重采样单位是图片，同一图片的所有 object token 获得相同权重；同一轮对 OT/JS 使用相同图片权重。三种子均值的 CI 在三个测试集的图片 ID 并集上抽样一次，并将相同图片的权重同步用于其出现的各个测试集，以保留跨种子的图片重叠；每轮取三个种子的 AUROC 差值平均。固定已训练模型与三个划分，本 CI 不包括重新训练或重新划分带来的额外不确定性。", "",
        "| 模型 | JS 是否加 R_q | 范围 | OT AUROC | JS AUROC | Δ(OT−JS) | 95% CI |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for pair in ("without_rq", "with_rq"):
            for row in [r for r in rows if r["model"] == model and r["comparison"] == pair]:
                scope = "三种子均值" if row["scope"] == "three_seed_mean" else row["scope"]
                lines.append(
                    f"| {model} | {'不加' if pair == 'without_rq' else '加'} | {scope} | "
                    f"{100*row['ot_auc']:.2f} | {100*row['js_auc']:.2f} | "
                    f"{100*row['delta_auc']:+.2f} | [{100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] |"
                )
    lines += ["", "图片并集数量：" + "；".join(f"{m}={n}" for m, n in union_sizes.items()) + "。逐轮差值见 `bootstrap_auroc_samples.npz`，未四舍五入的结果见 `bootstrap_auroc_ci.csv`。", ""]
    (report_dir / "bootstrap_auroc_ci.md").write_text("\n".join(lines))
    print(report_dir / "bootstrap_auroc_ci.md", flush=True)


if __name__ == "__main__":
    main()
