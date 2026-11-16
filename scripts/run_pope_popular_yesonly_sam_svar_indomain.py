#!/usr/bin/env python3
"""Train SAM and object-position SVAR on POPE-popular Yes-only 8:2 splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.qa_benchmark import load_jsonl, sha256_file
from detection.qa_probe import classification_metrics
from scripts.evaluate_coco_clevr_log_s_transfer import _predict_qa
from scripts.run_amber_exist_clevr_svar_transfer import predict as predict_svar
from scripts.run_clevr_yesonly_sam_svar_transfer import (
    OBJECT, PROTOCOL, ROOT, SEEDS, TARGET_ROOT, TARGET_RUN,
    _fmt, _json, _metric_row, _save_json, load_targets, train,
)


QUESTIONS_ROOT = Path("/root/rivermind-data/dataset/qa_sam_svar_yesonly")
OUT = TARGET_ROOT / "pope_popular_yesonly_sam_svar_indomain_82"
WIDTHS = {"llava_1_5_7b": 64, "qwen3_vl_8b": 72}


def load_pope(model: str, questions_root: Path, target_root: Path) -> tuple[dict, dict]:
    widths = {"sam": WIDTHS[model], "svar": 448}
    target = load_targets(model, target_root, widths)["pope"]
    question_path = questions_root / model / "pope/questions.jsonl"
    questions = load_jsonl(question_path)
    run = target_root / model / TARGET_RUN / "pope"
    summary = _json(run / "sam_svar_cross_position_summary.json")
    if sha256_file(question_path) != summary["questions_sha256"]:
        raise ValueError(f"{model}: POPE questions changed since feature export")
    feature_path = run / "sam_object_risk_answer_f_log_vs_svar_object_yesonly.npz"
    if sha256_file(feature_path) != summary["npz_sha256"]:
        raise ValueError(f"{model}: POPE feature matrix changed since export")
    with (run / "baseline" / PROTOCOL / "features.pkl").open("rb") as handle:
        baseline_rows = pickle.load(handle)
    by_key = {str(row["key"]): row for row in baseline_rows}
    if len(by_key) != len(baseline_rows) or len(questions) != len(target["keys"]):
        raise ValueError(f"{model}: duplicate or missing POPE records")
    if [str(row["key"]) for row in questions] != target["keys"].tolist():
        raise ValueError(f"{model}: POPE question and NPZ key order differs")
    if set(by_key) != set(target["keys"]):
        raise ValueError(f"{model}: POPE baseline key set differs")

    groups = {"train": [], "test": []}
    image_owner: dict[int, str] = {}
    digest = hashlib.sha256()
    for index, question in enumerate(questions):
        key = str(question["key"])
        split = str(question["probe_split"])
        label = int(target["labels"][index])
        image_id = int(question["image_id"])
        if split not in groups or split != target["splits"][index]:
            raise ValueError(f"{key}: POPE split mismatch")
        if image_owner.setdefault(image_id, split) != split:
            raise ValueError(f"{key}: POPE image appears in both splits")
        if label != int(question["gt_answer"] == "yes"):
            raise ValueError(f"{key}: POPE Yes-only ground truth mismatch")
        baseline = by_key[key]
        position = baseline["positions"][OBJECT]
        if (int(baseline["label"]) != label
                or int(baseline["image_id"]) != image_id
                or str(baseline["probe_split"]) != split
                or int(position["label"]) != label):
            raise ValueError(f"{key}: POPE baseline label or image mismatch")
        digest.update(key.encode())
        digest.update(bytes([label]))
        digest.update(split.encode())
        digest.update(target["sam"][index].tobytes())
        digest.update(target["svar"][index].tobytes())
        groups[split].append({
            "key": key,
            "label": label,
            "image": str(image_id),
            "image_id": image_id,
            "response_token_idx": int(position["response_token_idx"]),
            "sam": target["sam"][index],
            "svar": target["svar"][index],
        })
    counts = {
        split: {
            "real": sum(row["label"] == 1 for row in rows),
            "hallucination": sum(row["label"] == 0 for row in rows),
            "images": len({row["image"] for row in rows}),
        }
        for split, rows in groups.items()
    }
    if any(min(counts[split]["real"], counts[split]["hallucination"]) == 0 for split in groups):
        raise ValueError(f"{model}: single-class POPE split: {counts}")
    audit = {
        "model": model,
        "source": str(run),
        "label_protocol": PROTOCOL,
        "split_protocol": "original_POPE_popular_image_level_82",
        "sam_definition": "concat(risk@question_object_pre_token,f_log@prompt_last_token)",
        "svar_definition": "SVAR@question_object_pre_token, layers [5,19)",
        "counts": counts,
        "widths": widths,
        "questions_sha256": summary["questions_sha256"],
        "npz_sha256": summary["npz_sha256"],
        "cohort_feature_sha256": digest.hexdigest(),
    }
    return groups, audit


def evaluate(model: str, groups: dict, output: Path) -> list[dict]:
    rows = []
    for method in ("sam", "svar"):
        matrix = np.stack([row[method] for row in groups["test"]]).astype(np.float32)
        labels = np.asarray([row["label"] for row in groups["test"]], dtype=np.int64)
        predict = _predict_qa if method == "sam" else predict_svar
        for seed in SEEDS:
            folder = output / model / method / f"seed_{seed}"
            scores, threshold = predict(folder, matrix)
            report = _json(folder / "result.json")
            native = (report["metrics"]["test"]["auroc"] if method == "sam"
                      else report["test_metrics"]["real_positive"]["auc"])
            actual = classification_metrics(labels, scores, threshold)["auroc"]
            if abs(native - actual) > 2e-5:
                raise ValueError(f"{model}/{method}/seed_{seed}: checkpoint replay failed")
            rows.append(_metric_row(model, method, seed, "pope_popular", "test",
                                    labels, scores, threshold))
            print(f"[evaluate] {model}/{method}/seed_{seed}: AUC={actual:.4f}", flush=True)
    return rows


def write_outputs(model: str, audit: dict, rows: list[dict], output: Path) -> None:
    folder = output / model
    _save_json(folder / "cohort_audit.json", audit)
    _save_json(folder / "per_seed_results.json", {"rows": rows})
    with (folder / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: POPE popular Yes-only 域内检测", "",
        "原始图片级 8:2 划分；仅在 POPE popular train 训练，test 从未参与训练或阈值选择。SAM=物体词前 risk＋答案前 F_log；SVAR=物体词前第 5–18 层。种子 43/44/45。均值 ± 总体标准差，单位 %。", "",
        "| 方法 | 测试 N (Real/Hall) | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy | 预测 Real 比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("SAM", "SVAR"):
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != len(SEEDS):
            raise ValueError(f"{model}/{method}: incomplete seed results")
        n, real, hall = selected[0]["n"], selected[0]["real_n"], selected[0]["hall_n"]
        metrics = " | ".join(_fmt(selected, field) for field in (
            "auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
            "hall_recall", "accuracy", "pred_real_rate"))
        lines.append(f"| {method} | {n} ({real}/{hall}) | {metrics} |")
    lines.append("")
    (folder / "complete_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[complete] {folder / 'complete_results.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(WIDTHS))
    parser.add_argument("--questions-root", type=Path, default=QUESTIONS_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    groups, audit = load_pope(args.model, args.questions_root, args.target_root)
    print(f"[source] {args.model}: {audit['counts']}", flush=True)
    train(args.model, groups, audit, args.output_root)
    rows = evaluate(args.model, groups, args.output_root)
    write_outputs(args.model, audit, rows, args.output_root)


if __name__ == "__main__":
    main()
