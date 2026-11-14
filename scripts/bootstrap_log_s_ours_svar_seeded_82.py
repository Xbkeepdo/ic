#!/usr/bin/env python3
"""Paired image bootstrap for Qwen3 SAR+EMS versus native SVAR, seeded COCO 8:2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from detection.baselines import SVARMLP, build_dense_baseline_matrix, torch_hallucination_scores
from scripts.bootstrap_log_s_ours_svar_seeded_811 import evaluate, prepare_score
from scripts.run_coco_seeded_image_split_auc_aupr import row_key
from utils.io_utils import load_pkl


ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3_vl_8b"
SEEDS = (43, 44, 45)
SOURCE = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
PAIRED = ROOT / "outputs/log_s_ot_rank16_rq_seeded_82_43_45"
OUT = ROOT / "outputs/log_s_ours_svar_significance_82"
METRICS = ("auroc", "real_aupr", "hall_aupr")


def check_metrics(labels: np.ndarray, real_scores: np.ndarray, reported: dict, context: str) -> dict:
    scores = {
        "auroc": float(roc_auc_score(labels, real_scores)),
        "real_aupr": float(average_precision_score(labels, real_scores)),
        "hall_aupr": float(average_precision_score(1 - labels, 1 - real_scores)),
    }
    expected = {
        "auroc": reported["real_positive"]["auc"],
        "real_aupr": reported["real_positive"]["aupr"],
        "hall_aupr": reported["hallucination_positive"]["aupr"],
    }
    for metric in METRICS:
        if not np.isclose(scores[metric], expected[metric], atol=1e-6, rtol=0):
            raise ValueError(f"{context}/{metric}: reconstructed {scores[metric]} != reported {expected[metric]}")
    return scores


def load_data() -> tuple[np.ndarray, dict[int, dict], dict[int, dict]]:
    records = load_pkl(str(ROOT / "outputs" / MODEL /
                           "COCO4000-512-ENDAC-BASELINES-V2/baseline/features.pkl"))
    baseline_keys = [row_key(row) for row in records]
    if len(baseline_keys) != len(set(baseline_keys)):
        raise ValueError("duplicate SVAR object-token keys")
    loaded, audits = {}, {}
    for seed in SEEDS:
        split = json.loads((SOURCE / "splits" / f"seed{seed}.json").read_text())
        test_ids = np.asarray(split["test"], dtype=np.int64)
        if len(test_ids) != 800 or len(set(test_ids.tolist())) != 800:
            raise ValueError(f"seed{seed}: expected 800 test images")
        with np.load(PAIRED / MODEL / f"seed{seed}/paired_scores.npz", allow_pickle=False) as saved:
            images = np.asarray(saved["image_id"], dtype=np.int64)
            labels = np.asarray(saved["label"], dtype=np.int8)
            ours = np.asarray(saved["ot_real_score"], dtype=np.float64)
            keys = list(zip(images.astype(int), saved["response_token_idx"].astype(int),
                            saved["target_token_id"].astype(int)))
        if len(keys) != len(set(keys)) or not np.isin(images, test_ids).all():
            raise ValueError(f"seed{seed}: invalid Ours object-token keys or test images")
        ours_report = json.loads((SOURCE / MODEL / f"seed{seed}/ours/metrics.json").read_text())
        ours_points = check_metrics(
            labels, ours, ours_report["threshold_reports"]["train_f1"]["test_metrics"],
            f"seed{seed}/ours",
        )

        result = json.loads((SOURCE / MODEL / "baseline/results" / f"seed{seed}_svar" /
                             f"{MODEL}_baselines.json").read_text())["methods"]["svar"]
        checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
        test_set = set(test_ids.tolist())
        selected = [row for row in records if int(row["image_id"]) in test_set]
        matrix, baseline_labels, kept = build_dense_baseline_matrix(
            selected, "svar", svar_layer_start=int(checkpoint["layer_start"]),
            svar_layer_end=int(checkpoint["layer_end_exclusive"]),
        )
        scored_keys = [row_key(row) for row in kept]
        if len(scored_keys) != len(keys) or set(scored_keys) != set(keys):
            raise ValueError(f"seed{seed}: Ours and SVAR test object tokens differ")
        index_by_key = {key: i for i, key in enumerate(scored_keys)}
        order = np.asarray([index_by_key[key] for key in keys])
        if not np.array_equal(baseline_labels[order], labels):
            raise ValueError(f"seed{seed}: labels differ after token alignment")
        net = SVARMLP(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]))
        net.load_state_dict(checkpoint["state_dict"])
        with torch.inference_mode():
            svar_hall = torch_hallucination_scores(net, matrix, torch.device("cpu"))
        svar = 1.0 - svar_hall[order]
        svar_points = check_metrics(labels, svar, result["test_metrics"], f"seed{seed}/svar")

        loaded[seed] = {
            "test_ids": test_ids, "image_ids": images, "labels": labels,
            "points": {"ours": ours_points, "svar": svar_points},
            "curves": {
                "ours": {
                    "real": prepare_score(ours, labels),
                    "hall": prepare_score(1.0 - ours, 1 - labels),
                },
                "svar": {
                    "real": prepare_score(svar, labels),
                    "hall": prepare_score(1.0 - svar, 1 - labels),
                },
            },
        }
        audits[seed] = {
            "test_images": len(test_ids), "test_tokens": len(labels),
            "real_tokens": int(labels.sum()), "hall_tokens": int((1 - labels).sum()),
            "svar_reordered_positions": int(np.sum(order != np.arange(len(order)))),
            "svar_layer_start": int(checkpoint["layer_start"]),
            "svar_layer_end_exclusive": int(checkpoint["layer_end_exclusive"]),
            "point_metrics_verified": True,
        }
    union = np.unique(np.concatenate([loaded[seed]["test_ids"] for seed in SEEDS]))
    for seed in SEEDS:
        indices = np.searchsorted(union, loaded[seed]["image_ids"])
        if not np.array_equal(union[indices], loaded[seed]["image_ids"]):
            raise ValueError("image alignment failed")
        loaded[seed]["image_indices"] = indices
    return union, loaded, audits


def format_row(row: dict) -> str:
    return (f"| {row['scope']} | {row['metric']} | {100 * row['ours']:.2f} | "
            f"{100 * row['svar']:.2f} | {100 * row['delta']:+.2f} | "
            f"[{100 * row['ci_low']:+.2f}, {100 * row['ci_high']:+.2f}] | "
            f"{row['p_two_sided']:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.iterations < 100 or args.batch_size < 1:
        parser.error("iterations must be at least 100; batch size must be positive")
    torch.set_num_threads(1)
    union, loaded, audits = load_data()
    draws = {(seed, metric): np.empty(args.iterations) for seed in SEEDS for metric in METRICS}
    rng = np.random.default_rng(args.seed)
    probabilities = np.full(len(union), 1.0 / len(union))
    for start in range(0, args.iterations, args.batch_size):
        end = min(start + args.batch_size, args.iterations)
        image_weights = rng.multinomial(len(union), probabilities, size=end - start)
        for seed in SEEDS:
            entry = loaded[seed]
            token_weights = image_weights[:, entry["image_indices"]].astype(np.float64)
            ours = evaluate(token_weights, entry, "ours")
            svar = evaluate(token_weights, entry, "svar")
            for metric in METRICS:
                draws[(seed, metric)][start:end] = ours[metric] - svar[metric]

    rows, samples = [], {}
    for metric in METRICS:
        for scope in ("three_seed_mean", *SEEDS):
            if scope == "three_seed_mean":
                values = np.mean(np.stack([draws[(s, metric)] for s in SEEDS]), axis=0)
                ours = np.mean([loaded[s]["points"]["ours"][metric] for s in SEEDS])
                svar = np.mean([loaded[s]["points"]["svar"][metric] for s in SEEDS])
                n_images = len(union)
                n_tokens = sum(len(loaded[s]["labels"]) for s in SEEDS)
            else:
                values = draws[(scope, metric)]
                ours = loaded[scope]["points"]["ours"][metric]
                svar = loaded[scope]["points"]["svar"][metric]
                n_images = len(loaded[scope]["test_ids"])
                n_tokens = len(loaded[scope]["labels"])
            if not np.isfinite(values).all():
                raise ValueError(f"{scope}/{metric}: nonfinite bootstrap result")
            delta = float(ours - svar)
            low, high = np.quantile(values, [0.025, 0.975])
            p = (1 + np.count_nonzero(np.abs(values - delta) >= abs(delta))) / (len(values) + 1)
            rows.append({
                "scope": str(scope), "metric": metric, "test_images": n_images,
                "test_tokens": n_tokens, "ours": float(ours), "svar": float(svar),
                "delta": delta, "ci_low": float(low), "ci_high": float(high),
                "p_two_sided": float(p),
            })
            samples[f"{metric}_{scope}"] = values
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "bootstrap_deltas.npz", **samples)
    (OUT / "alignment_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = [
        "# Qwen3-VL-8B：COCO 8:2 SAR+EMS 与原生 SVAR 显著性检验", "",
        "种子 43/44/45 同时控制 3200/800 图片划分与检测器随机性。两方法在相同 object token 上测试；SAR+EMS 使用原始 K=32、τ=0.03 特征。SVAR 从既有原生训练检查点恢复分数；两方法的逐种子 AUROC、Real AUPR、Hall AUPR 已与既有报告逐项核对。", "",
        f"对三个测试集图片 ID 并集（{len(union)} 张不同图片）做 {args.iterations:,} 次配对图片级 bootstrap，重复出现的图片在各种子中共用抽样权重。表中差值为 SAR+EMS−SVAR，单位百分点；95% CI 是百分位区间，双侧 p 通过重中心化 bootstrap 分布估计。固定已有三个划分和已训练检测器，不覆盖重训或重新划分的不确定性。AUROC 为主指标；AUPR 为补充分析，表中 p 值未经多重比较校正。", "",
        "## 三种子均值", "",
        "| 范围 | 指标 | SAR+EMS (%) | SVAR (%) | Δ (pp) | 95% CI (pp) | bootstrap p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines += [format_row(row) for row in rows if row["scope"] == "three_seed_mean"]
    lines += ["", "## 逐种子结果", "",
              "| 种子 | 指标 | SAR+EMS (%) | SVAR (%) | Δ (pp) | 95% CI (pp) | bootstrap p |",
              "|---|---|---:|---:|---:|---:|---:|"]
    lines += [format_row(row) for row in rows if row["scope"] != "three_seed_mean"]
    lines += ["", "恢复分数的 token 与原报告数值审计见 `alignment_audit.json`；完整精度见 `results.csv`，逐次 bootstrap 差值见 `bootstrap_deltas.npz`。", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
