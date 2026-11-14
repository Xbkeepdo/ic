#!/usr/bin/env python3
"""Paired image-level bootstrap for SAR+EMS versus native SVAR on COCO 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from detection.baselines import build_dense_baseline_matrix
from scripts import run_endac_baselines_811_dual_threshold as baseline_protocol
from scripts.bootstrap_log_s_ot_js_seeded_811 import prepare_score, weighted_auc


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
FEATURES = ROOT / "outputs/log_s_mechanism_ablation_811"
OUT = ROOT / "outputs/log_s_ours_svar_significance_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METHODS = {"ours": "sar_ems", "svar": "svar"}
METRICS = ("auroc", "real_aupr", "hall_aupr")


def weighted_average_precision(
    weights: np.ndarray, labels: np.ndarray,
    order: np.ndarray, groups: np.ndarray,
) -> np.ndarray:
    """Weighted sklearn-style average precision, including tied score groups."""
    ordered = weights[:, order]
    positives = np.add.reduceat(ordered * labels[order], groups, axis=1)[:, ::-1]
    negatives = np.add.reduceat(ordered * (1 - labels[order]), groups, axis=1)[:, ::-1]
    tp = np.cumsum(positives, axis=1)
    denominator = tp + np.cumsum(negatives, axis=1)
    precision = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    total_pos = tp[:, -1]
    weighted_sum = np.sum(precision * positives, axis=1)
    return np.divide(weighted_sum, total_pos,
                     out=np.full_like(weighted_sum, np.nan), where=total_pos > 0)


def evaluate(weights: np.ndarray, entry: dict, method: str) -> dict[str, np.ndarray]:
    real_labels = entry["labels"]
    hall_labels = 1 - real_labels
    curves = entry["curves"][method]
    return {
        "auroc": weighted_auc(weights, real_labels, *curves["real"]),
        "real_aupr": weighted_average_precision(weights, real_labels, *curves["real"]),
        "hall_aupr": weighted_average_precision(weights, hall_labels, *curves["hall"]),
    }


def load_model(model: str):
    with np.load(FEATURES / model / "compact_features.npz", allow_pickle=False) as saved:
        image_ids = np.asarray(saved["image_id"], dtype=np.int64)
        labels = np.asarray(saved["label"], dtype=np.int8)
        responses = np.asarray(saved["response_token_idx"], dtype=np.int32)
        tokens = np.asarray(saved["target_token_id"], dtype=np.int32)
    if len(image_ids) != len(labels) or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{model}: invalid canonical labels")
    all_keys = list(zip(image_ids.astype(int), responses.astype(int), tokens.astype(int)))
    if len(set(all_keys)) != len(all_keys):
        raise ValueError(f"{model}: duplicate canonical object-token keys")
    baseline_records = baseline_protocol.load_baseline_records(model)
    baseline_keys = [baseline_protocol.key_of(row) for row in baseline_records]
    if (len(set(baseline_keys)) != len(baseline_keys) or
            set(baseline_keys) != set(all_keys)):
        raise ValueError(f"{model}: baseline and ours target keys differ")
    loaded = {}
    for seed in SEEDS:
        split = json.loads((SOURCE / "splits" / f"seed{seed}.json").read_text())
        test_ids = np.asarray(split["test"], dtype=np.int64)
        if len(test_ids) != 400 or len(np.unique(test_ids)) != 400:
            raise ValueError(f"{model}/seed{seed}: invalid test image split")
        mask = np.isin(image_ids, test_ids)
        token_images, token_labels = image_ids[mask], labels[mask]
        canonical_test_keys = [k for k, keep in zip(all_keys, mask) if keep]
        test_set = set(test_ids.tolist())
        entry = {"test_ids": test_ids, "image_ids": token_images,
                 "labels": token_labels, "curves": {}, "points": {},
                 "svar_reordered_positions": 0}
        for method, case in METHODS.items():
            folder = SOURCE / model / f"seed{seed}" / case
            result = json.loads((folder / "result.json").read_text())
            with np.load(folder / "scores.npz", allow_pickle=False) as saved:
                score_labels = np.asarray(saved["test_label"], dtype=np.int8)
                hall_score = np.asarray(saved["test_hall_score"], dtype=np.float64)
            if method == "svar":
                config = result["native_training"]["paper_config"]
                test_records = [row for row in baseline_records
                                if int(row["image_id"]) in test_set]
                _matrix, _labels, kept = build_dense_baseline_matrix(
                    test_records, "svar",
                    svar_layer_start=int(config["layer_start"]),
                    svar_layer_end=int(config["layer_end_exclusive"]),
                )
                scored_keys = [baseline_protocol.key_of(row) for row in kept]
                if (len(scored_keys) != len(hall_score) or
                        set(scored_keys) != set(canonical_test_keys)):
                    raise ValueError(f"{model}/seed{seed}: SVAR score keys differ")
                by_key = {target_key: position for position, target_key in enumerate(scored_keys)}
                order = np.asarray([by_key[k] for k in canonical_test_keys], dtype=np.int64)
                entry["svar_reordered_positions"] = int(np.sum(order != np.arange(len(order))))
                score_labels = score_labels[order]
                hall_score = hall_score[order]
            if (not np.array_equal(score_labels, token_labels) or
                    hall_score.shape != token_labels.shape or
                    not np.isfinite(hall_score).all()):
                raise ValueError(f"{model}/seed{seed}/{method}: test token alignment differs")
            real_score = 1.0 - hall_score
            entry["curves"][method] = {
                "real": prepare_score(real_score, token_labels),
                "hall": prepare_score(hall_score, 1 - token_labels),
            }
            point = {name: float(value[0]) for name, value in evaluate(
                np.ones((1, len(token_labels)), dtype=np.float64), entry, method
            ).items()}
            reported = {
                "auroc": result["test"]["auroc"],
                "real_aupr": result["test"]["real"]["aupr"],
                "hall_aupr": result["test"]["hall"]["aupr"],
            }
            if result["token_counts"]["test"] != len(token_labels):
                raise ValueError(f"{model}/seed{seed}/{method}: token count differs")
            for metric in METRICS:
                if not np.isclose(point[metric], reported[metric], atol=1e-10):
                    raise ValueError(f"{model}/seed{seed}/{method}/{metric}: score disagrees with saved result")
            entry["points"][method] = point
        loaded[seed] = entry
    union_images = np.unique(np.concatenate([loaded[s]["test_ids"] for s in SEEDS]))
    for seed in SEEDS:
        indices = np.searchsorted(union_images, loaded[seed]["image_ids"])
        if not np.array_equal(union_images[indices], loaded[seed]["image_ids"]):
            raise AssertionError("image alignment failed")
        loaded[seed]["image_indices"] = indices
    return union_images, loaded


def bootstrap_model(model: str, iterations: int, batch_size: int, rng: np.random.Generator):
    union_images, loaded = load_model(model)
    draws = {(seed, metric): np.empty(iterations, dtype=np.float64)
             for seed in SEEDS for metric in METRICS}
    probabilities = np.full(len(union_images), 1.0 / len(union_images))
    for start in range(0, iterations, batch_size):
        end = min(start + batch_size, iterations)
        # Shared image weights preserve pairing between methods and overlap
        # between the three seed-specific test sets.
        image_weights = rng.multinomial(
            len(union_images), probabilities, size=end - start
        )
        for seed in SEEDS:
            entry = loaded[seed]
            token_weights = image_weights[:, entry["image_indices"]].astype(np.float64)
            ours = evaluate(token_weights, entry, "ours")
            svar = evaluate(token_weights, entry, "svar")
            for metric in METRICS:
                draws[(seed, metric)][start:end] = ours[metric] - svar[metric]

    samples = {}
    rows = []
    for metric in METRICS:
        for scope in (*SEEDS, "three_seed_mean"):
            if scope == "three_seed_mean":
                values = np.mean(np.stack([draws[(s, metric)] for s in SEEDS]), axis=0)
                ours = np.mean([loaded[s]["points"]["ours"][metric] for s in SEEDS])
                svar = np.mean([loaded[s]["points"]["svar"][metric] for s in SEEDS])
                n_images = len(union_images)
                n_tokens = sum(len(loaded[s]["labels"]) for s in SEEDS)
            else:
                values = draws[(scope, metric)]
                ours = loaded[scope]["points"]["ours"][metric]
                svar = loaded[scope]["points"]["svar"][metric]
                n_images = len(loaded[scope]["test_ids"])
                n_tokens = len(loaded[scope]["labels"])
            if not np.isfinite(values).all():
                raise ValueError(f"{model}/{metric}/{scope}: degenerate bootstrap draw")
            delta = float(ours - svar)
            low, high = np.quantile(values, (0.025, 0.975))
            # Recenter the bootstrap distribution at the null of zero difference.
            p_two_sided = (1 + np.count_nonzero(
                np.abs(values - delta) >= abs(delta)
            )) / (len(values) + 1)
            name = f"{model}_{metric}_{scope}"
            samples[name] = values
            rows.append({
                "model": model, "metric": metric, "scope": str(scope),
                "n_test_images": n_images, "n_test_tokens": n_tokens,
                "ours": float(ours), "svar": float(svar),
                "delta_ours_minus_svar": delta,
                "ci95_low": float(low), "ci95_high": float(high),
                "bootstrap_p_two_sided": float(p_two_sided),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            })
    alignment = {str(seed): loaded[seed]["svar_reordered_positions"]
                 for seed in SEEDS}
    return rows, samples, len(union_images), alignment


def p_string(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def format_row(row: dict) -> str:
    return (f"| {row['model']} | {row['metric']} | {row['scope']} | "
            f"{100*row['ours']:.2f} | {100*row['svar']:.2f} | "
            f"{100*row['delta_ours_minus_svar']:+.2f} | "
            f"[{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}] | "
            f"{p_string(row['bootstrap_p_two_sided'])} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1:
        parser.error("at least 100 bootstrap iterations and a positive batch size are required")
    rows, samples, union_sizes, alignment = [], {}, {}, {}
    rng = np.random.default_rng(args.seed)
    for model in MODELS:
        model_rows, model_samples, union_sizes[model], alignment[model] = bootstrap_model(
            model, args.iterations, args.batch_size, rng
        )
        rows.extend(model_rows)
        samples.update(model_samples)
        print(f"[{model}] completed {args.iterations} paired image bootstrap draws", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "bootstrap_deltas.npz", **samples)
    (OUT / "alignment_audit.json").write_text(json.dumps(
        {"svar_reordered_positions": alignment}, indent=2
    ) + "\n")
    lines = [
        "# COCO 8:1:1：SAR+EMS 与原生 SVAR 的配对显著性检验", "",
        "LLaVA-1.5-7B、Qwen3-VL-8B；种子 43/44/45 同时控制 3200/400/400 图片划分与检测器随机性。每个种子测试集两方法使用同一 object token、标签与图片。比较测试分数的 AUROC（主指标）、Real AUPR 和 Hall AUPR（补充指标）。差值均为 SAR+EMS − SVAR，单位百分点。", "",
        f"采用 {args.iterations:,} 次配对图片级 bootstrap（随机种子 {args.seed}），95% CI 为 2.5%/97.5% 百分位数；同一图片的所有 object token 共用权重，两方法共用抽样。三种子均值的重采样在三个测试集的图片 ID 并集上进行，同一图片若出现多次，其权重在各种子间同步。表中双侧 p 值由以零差值为原假设、重中心化的 bootstrap 分布估计，表内未作多重比较校正；下方另列两模型 AUROC 主指标的 Holm 校正值。固定已有三个划分和已训练检测器，此 CI 不覆盖重新划分或重新训练的不确定性。", "",
        "## 三种子均值", "",
        "| 模型 | 指标 | 范围 | SAR+EMS | SVAR | Δ (pp) | 95% CI (pp) | bootstrap p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for metric in METRICS:
            lines.append(format_row(next(row for row in rows if row["model"] == model and row["metric"] == metric and row["scope"] == "three_seed_mean")))
    primary = sorted(
        (row for row in rows if row["metric"] == "auroc" and row["scope"] == "three_seed_mean"),
        key=lambda row: row["bootstrap_p_two_sided"],
    )
    holm = {}
    running = 0.0
    for rank, row in enumerate(primary):
        running = max(running, min(1.0, (len(primary) - rank) * row["bootstrap_p_two_sided"]))
        holm[row["model"]] = running
    lines += ["", "两模型 AUROC 主指标的 Holm 校正双侧 bootstrap p：" +
              "；".join(f"{model}={p_string(holm[model])}" for model in MODELS) +
              "。AUPR 属于补充分析，其表中 p 值未校正。"]
    lines += ["", "## 逐种子结果", "",
              "| 模型 | 指标 | 种子 | SAR+EMS | SVAR | Δ (pp) | 95% CI (pp) | bootstrap p |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        for metric in METRICS:
            for seed in SEEDS:
                lines.append(format_row(next(row for row in rows if row["model"] == model and row["metric"] == metric and row["scope"] == str(seed))))
    lines += ["", "图片并集数量：" + "；".join(f"{model}={count}" for model, count in union_sizes.items()) + "。SVAR 分数按 `(image_id,response_token_idx,target_token_id)` 对齐，重排位置数见 `alignment_audit.json`。未四舍五入的数据见 `results.csv`，逐次抽样差值见 `bootstrap_deltas.npz`。", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(OUT / "summary.md", flush=True)


if __name__ == "__main__":
    main()
