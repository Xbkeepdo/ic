#!/usr/bin/env python3
"""Run all six JS variants against saved OT on seeded COCO 8:2 splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_log_s_ot_rank16_rq_seeded_82 import load_features, test_scores  # noqa: E402

OUT = ROOT / "outputs/log_s_ot_js_all_variants_seeded_82_43_45"
SOURCE_82 = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
PRIOR_RANK = ROOT / "outputs/log_s_ot_rank16_rq_seeded_82_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METHODS = ("ot", "union16", "union16_rq", "union32", "union32_rq", "rank16", "rank16_rq")
NEW = ("union16", "union16_rq", "union32", "union32_rq", "rank16")
DISPLAY = {
    "ot": "SAR+EMS (OT+R_q)",
    "union16": "SAR+JS Top16 union",
    "union16_rq": "SAR+JS Top16 union+R_q",
    "union32": "SAR+JS Top32 union",
    "union32_rq": "SAR+JS Top32 union+R_q",
    "rank16": "SAR+JS Top16 rank",
    "rank16_rq": "SAR+JS Top16 rank+R_q",
}


def folder(model: str, seed: int, case: str) -> Path:
    if case == "ot":
        return SOURCE_82 / model / f"seed{seed}" / "ours"
    if case == "rank16_rq":
        return PRIOR_RANK / model / f"seed{seed}" / "rank16_rq"
    return OUT / model / f"seed{seed}" / case


def score_metrics(labels: np.ndarray, real_scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "auroc": float(roc_auc_score(labels, real_scores)),
        "real_aupr": float(average_precision_score(labels, real_scores)),
        "hall_aupr": float(average_precision_score(1 - labels, 1 - real_scores)),
    }


def check_scores(model: str, seed: int, case: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = score_metrics(labels, scores)
    saved = json.loads((folder(model, seed, case) / "metrics.json").read_text())
    test = saved["threshold_reports"]["train_f1"]["test_metrics"]
    expected = {
        "auroc": float(test["real_positive"]["auc"]),
        "real_aupr": float(test["real_positive"]["aupr"]),
        "hall_aupr": float(test["hallucination_positive"]["aupr"]),
    }
    if any(abs(result[key] - expected[key]) > 1e-6 for key in result):
        raise ValueError(f"{model}/seed{seed}/{case}: recovered score metrics differ: {result} vs {expected}")
    return result


def run_model(model: str, device: str) -> None:
    import torch
    from scripts.train_log_s_hpre_k_tau_ablation import _fit
    from utils.config_utils import load_config

    torch.set_num_threads(1)
    matrices, labels, image_ids, token_indices = load_features(model)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    (OUT / model).mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        split = json.loads((SOURCE_82 / "splits" / f"seed{seed}.json").read_text())
        train_ids, test_ids = set(split["train"]), set(split["test"])
        if len(train_ids) != 3200 or len(test_ids) != 800 or train_ids & test_ids:
            raise ValueError(f"seed{seed}: invalid split")
        fit_mask = np.isin(image_ids, list(train_ids))
        test_mask = np.isin(image_ids, list(test_ids))
        if not np.all(fit_mask ^ test_mask):
            raise ValueError(f"{model}/seed{seed}: invalid object-token split")
        for case in NEW:
            _fit(
                matrix=matrices[case], labels=labels, fit_mask=fit_mask, eval_mask=test_mask,
                seed=seed, root_config=root_config, device=torch.device(device),
                folder=folder(model, seed, case),
            )
            print(f"{model} seed{seed} trained {case}", flush=True)
        ot_config = json.loads((folder(model, seed, "ot") / "config.json").read_text())
        for case in METHODS[1:]:
            if json.loads((folder(model, seed, case) / "config.json").read_text()) != ot_config:
                raise ValueError(f"{model}/seed{seed}/{case}: probe configuration differs from OT")
        with np.load(PRIOR_RANK / model / f"seed{seed}" / "paired_scores.npz", allow_pickle=False) as prior:
            expected_rows = {
                "image_id": image_ids[test_mask],
                "response_token_idx": token_indices[test_mask, 0],
                "target_token_id": token_indices[test_mask, 1],
                "label": labels[test_mask],
            }
            if any(not np.array_equal(prior[name], value) for name, value in expected_rows.items()):
                raise ValueError(f"{model}/seed{seed}: prior OT/rank test rows differ")
            scores = {
                "ot": np.asarray(prior["ot_real_score"], dtype=np.float32),
                "rank16_rq": np.asarray(prior["js_real_score"], dtype=np.float32),
            }
        for case in NEW:
            scores[case] = test_scores(folder(model, seed, case), matrices[case][test_mask], device=device)
        metrics = {
            case: check_scores(model, seed, case, labels[test_mask], scores[case])
            for case in METHODS
        }
        destination = OUT / model / f"seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "paired_scores.npz",
            **expected_rows,
            **{f"{case}_real_score": scores[case] for case in METHODS},
        )
        (destination / "metrics.json").write_text(json.dumps({
            "model": model, "seed": seed, "train_images": 3200, "test_images": 800,
            "train_tokens": int(fit_mask.sum()), "test_tokens": int(test_mask.sum()),
            "feature_order": "[EMS or JS(+R_q), SAR]",
            "same_probe_config_as_saved_OT": True,
            "methods": metrics,
        }, indent=2) + "\n")
        print(f"{model} seed{seed}: 7 aligned methods, {int(test_mask.sum())} test tokens", flush=True)


def summarize() -> None:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            result = json.loads((OUT / model / f"seed{seed}" / "metrics.json").read_text())
            for case in METHODS:
                rows.append({
                    "model": model, "seed": seed, "case": case,
                    "test_tokens": result["test_tokens"], **result["methods"][case],
                })
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# COCO 8:2：OT 与六种 JS 构造", "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B；种子 43/44/45 同时控制 3200/800 图片划分与单隐藏层宽度 128 MLP 初始化。OT 复用原 8:2 检测器；Top16 排名+R_q 复用已完成的同配置检测器；其余五种 JS 重新训练。特征顺序统一为 [EMS 或 JS(+R_q), SAR]。R_q 固定复用原 K=32 OT 的目标状态项。旧 Top16 并集 JS 取原始缓存；Top32 并集和排名 JS 从已保存 P/T 重建。AUPR 分别以 real、hall 为正类。", "",
        "三种子均值 ± 总体标准差，单位百分数。", "",
        "| 模型 | 方法 | AUROC | Real AUPR | Hall AUPR | ΔAUROC (OT−JS, pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        ot_rows = [r for r in rows if r["model"] == model and r["case"] == "ot"]
        ot_mean = mean(r["auroc"] for r in ot_rows)
        for case in METHODS:
            subset = [r for r in rows if r["model"] == model and r["case"] == case]
            fmt = lambda key: f"{100*mean(r[key] for r in subset):.2f} ± {100*pstdev(r[key] for r in subset):.2f}"
            delta = 100 * (ot_mean - mean(r["auroc"] for r in subset))
            shown = "—" if case == "ot" else f"{0.0 if abs(delta) < 0.005 else delta:+.2f}"
            lines.append(f"| {model} | {DISPLAY[case]} | {fmt('auroc')} | {fmt('real_aupr')} | {fmt('hall_aupr')} | {shown} |")
    lines += ["", "逐种子完整指标见 per_seed.csv；配对逐图 Bootstrap 见 bootstrap_3000。", ""]
    (OUT / "complete_results.md").write_text("\n".join(lines))
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
