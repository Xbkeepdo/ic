#!/usr/bin/env python3
"""Compare POPE in-domain training with CLEVR+AMBER transfer, varying only initialization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.qa_probe import classification_metrics
from scripts.evaluate_coco_clevr_log_s_transfer import _predict_qa
from scripts.run_amber_exist_clevr_svar_transfer import predict as predict_svar
from scripts.run_clevr_amber_yesonly_sam_svar_to_pope import (
    QUESTIONS_ROOT, combine, load_amber,
)
from scripts.run_clevr_yesonly_sam_svar_transfer import (
    SEEDS, SOURCE_ROOT, TARGET_ROOT, _fmt, _json, _metric_row,
    _save_json, load_clevr, train,
)
from scripts.run_pope_popular_yesonly_sam_svar_indomain import load_pope


OUT = TARGET_ROOT / "pope_yesonly_domain_vs_clevr_amber_init_only_82"
TRAINING_SEED = 43


def _matrix(rows: list[dict], method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([row[method] for row in rows]).astype(np.float32),
        np.asarray([row["label"] for row in rows], dtype=np.int64),
    )


def _check_replay(folder: Path, method: str, test_rows: list[dict]) -> None:
    predictor = _predict_qa if method == "sam" else predict_svar
    matrix, labels = _matrix(test_rows, method)
    scores, threshold = predictor(folder, matrix)
    actual = classification_metrics(labels, scores, threshold)["auroc"]
    report = _json(folder / "result.json")
    recorded = (report["metrics"]["test"]["auroc"] if method == "sam"
                else report["test_metrics"]["real_positive"]["auc"])
    if abs(actual - recorded) > 2e-5:
        raise ValueError(f"{folder}: source test checkpoint replay failed")
    if (report.get("initialization_seed") != int(folder.name.removeprefix("seed_"))
            or report.get("training_random_seed") != TRAINING_SEED
            or report.get("seed_scope") != "model_initialization_only"):
        raise ValueError(f"{folder}: seed scope metadata mismatch")


def evaluate(model: str, sources: dict, pope_test: list[dict], output: Path) -> list[dict]:
    rows = []
    for source, (groups, _) in sources.items():
        for method in ("sam", "svar"):
            predictor = _predict_qa if method == "sam" else predict_svar
            matrix, labels = _matrix(pope_test, method)
            for seed in SEEDS:
                folder = output / source / model / method / f"seed_{seed}"
                _check_replay(folder, method, groups["test"])
                scores, threshold = predictor(folder, matrix)
                rows.append({
                    "training_source": source,
                    "initialization_seed": seed,
                    "training_random_seed": TRAINING_SEED,
                    **_metric_row(model, method, seed, "pope_popular", "test",
                                  labels, scores, threshold),
                })
                print(f"[evaluate] {model}/{source}/{method}/seed_{seed}", flush=True)
    return rows


def write_outputs(model: str, sources: dict, pope_test: list[dict], rows: list[dict],
                  output: Path) -> None:
    folder = output / model
    digest = hashlib.sha256()
    for row in pope_test:
        digest.update(row["key"].encode())
        digest.update(bytes([row["label"]]))
    _save_json(folder / "cohort_audit.json", {
        "model": model,
        "seed_scope": "model_initialization_only",
        "initialization_seeds": list(SEEDS),
        "fixed_training_random_seed": TRAINING_SEED,
        "source_audits": {source: audit for source, (_, audit) in sources.items()},
        "pope_test_n": len(pope_test),
        "pope_test_real": sum(row["label"] == 1 for row in pope_test),
        "pope_test_hall": sum(row["label"] == 0 for row in pope_test),
        "pope_test_keys_labels_sha256": digest.hexdigest(),
    })
    _save_json(folder / "per_seed_results.json", {"rows": rows})
    with (folder / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: POPE popular Yes-only 域内 vs CLEVR＋AMBER 迁移（仅初始化变化）", "",
        f"固定 POPE 图片级 8:2 划分、训练数据顺序、batch shuffle 和 dropout 随机序列（训练随机种子 {TRAINING_SEED}）。43/44/45 仅控制模型初始权重。两方案评估同一 POPE test；迁移方案没有使用 POPE 训练数据、标准化或调阈值。均值 ± 总体标准差，单位 %。", "",
        "| 方法 | 训练来源 | Test N (Real/Hall) | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy | 预测 Real 比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("SAM", "SVAR"):
        for source in ("pope_indomain", "clevr_amber_transfer"):
            selected = [row for row in rows if (row["method"], row["training_source"]) == (method, source)]
            if len(selected) != len(SEEDS):
                raise ValueError(f"{model}/{method}/{source}: incomplete seeds")
            n, real, hall = selected[0]["n"], selected[0]["real_n"], selected[0]["hall_n"]
            if any((row["n"], row["real_n"], row["hall_n"]) != (n, real, hall) for row in selected):
                raise ValueError("POPE test cohort changed across seeds")
            metrics = " | ".join(_fmt(selected, field) for field in (
                "auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
                "hall_recall", "accuracy", "pred_real_rate"))
            lines.append(f"| {method} | {source} | {n} ({real}/{hall}) | {metrics} |")
    lines.append("")
    (folder / "complete_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[complete] {folder / 'complete_results.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("llava_1_5_7b", "qwen3_vl_8b"), required=True)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--clevr-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--questions-root", type=Path, default=QUESTIONS_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pope, pope_audit = load_pope(args.model, args.questions_root, args.target_root)
    clevr, clevr_audit = load_clevr(args.model, args.clevr_root)
    amber, amber_audit = load_amber(args.model, args.questions_root, args.target_root)
    mixed, mixed_audit = combine(args.model, clevr, clevr_audit, amber, amber_audit)
    sources = {
        "pope_indomain": (pope, pope_audit),
        "clevr_amber_transfer": (mixed, mixed_audit),
    }
    print(f"[cohorts] {args.model}: POPE={pope_audit['counts']}, mixed={mixed_audit['counts']}", flush=True)
    for source, (groups, audit) in sources.items():
        train(args.model, groups, audit, args.output_root / source,
              training_seed=TRAINING_SEED)
    rows = evaluate(args.model, sources, pope["test"], args.output_root)
    write_outputs(args.model, sources, pope["test"], rows, args.output_root)


if __name__ == "__main__":
    main()
