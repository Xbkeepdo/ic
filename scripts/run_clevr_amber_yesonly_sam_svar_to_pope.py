#!/usr/bin/env python3
"""Mix CLEVR/AMBER Yes-only training and transfer SAM/SVAR to POPE popular."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    OBJECT, PROTOCOL, ROOT, SEEDS, SOURCE_ROOT, TARGET_ROOT, TARGET_RUN,
    _fmt, _json, _metric_row, _save_json, load_clevr, load_targets, train,
)


QUESTIONS_ROOT = Path("/root/rivermind-data/dataset/qa_sam_svar_yesonly")
OUT = TARGET_ROOT / "clevr_amber_yesonly_sam_svar_to_pope_popular_82"
WIDTHS = {"llava_1_5_7b": 64, "qwen3_vl_8b": 72}
AMBER = "amber_discriminative"
POPE = "pope"


def load_amber(model: str, questions_root: Path, target_root: Path) -> tuple[dict, dict]:
    widths = {"sam": WIDTHS[model], "svar": 448}
    amber = load_targets(model, target_root, widths)[AMBER]
    questions_path = questions_root / model / AMBER / "questions.jsonl"
    questions = load_jsonl(questions_path)
    run = target_root / model / TARGET_RUN / AMBER
    summary = _json(run / "sam_svar_cross_position_summary.json")
    matrix_path = run / "sam_object_risk_answer_f_log_vs_svar_object_yesonly.npz"
    if (sha256_file(questions_path) != summary["questions_sha256"]
            or sha256_file(matrix_path) != summary["npz_sha256"]):
        raise ValueError(f"{model}: AMBER source changed since feature export")
    with (run / "baseline" / PROTOCOL / "features.pkl").open("rb") as handle:
        baseline_rows = pickle.load(handle)
    by_key = {str(row["key"]): row for row in baseline_rows}
    if len(by_key) != len(baseline_rows) or len(questions) != len(amber["keys"]):
        raise ValueError(f"{model}: AMBER question/baseline count mismatch")
    if [str(row["key"]) for row in questions] != amber["keys"].tolist() or set(by_key) != set(amber["keys"]):
        raise ValueError(f"{model}: AMBER question/baseline key mismatch")

    groups = {"train": [], "test": []}
    image_owner: dict[int, str] = {}
    digest = hashlib.sha256()
    for index, question in enumerate(questions):
        key = str(question["key"])
        split = str(question["probe_split"])
        label = int(amber["labels"][index])
        image_id = int(question["image_id"])
        if split not in groups or split != amber["splits"][index]:
            raise ValueError(f"{key}: AMBER split mismatch")
        if image_owner.setdefault(image_id, split) != split:
            raise ValueError(f"{key}: AMBER image appears in both splits")
        if question["gt_answer"] != "no" or label != 0:
            raise ValueError(f"{key}: AMBER existence Yes-only must be hallucination")
        baseline = by_key[key]
        position = baseline["positions"][OBJECT]
        if (int(baseline["label"]) != label
                or int(baseline["image_id"]) != image_id
                or str(baseline["probe_split"]) != split
                or int(position["label"]) != label):
            raise ValueError(f"{key}: AMBER baseline label or image mismatch")
        digest.update(key.encode())
        digest.update(bytes([label]))
        digest.update(split.encode())
        digest.update(amber["sam"][index].tobytes())
        digest.update(amber["svar"][index].tobytes())
        groups[split].append({
            "key": key,
            "label": label,
            "image": f"amber:{image_id}",
            "image_id": image_id,
            "response_token_idx": int(position["response_token_idx"]),
            "sam": amber["sam"][index],
            "svar": amber["svar"][index],
        })
    counts = {
        split: {"real": 0, "hallucination": len(rows),
                "images": len({row["image"] for row in rows})}
        for split, rows in groups.items()
    }
    audit = {
        "model": model,
        "source": str(run),
        "label_protocol": PROTOCOL,
        "counts": counts,
        "widths": widths,
        "questions_sha256": summary["questions_sha256"],
        "npz_sha256": summary["npz_sha256"],
        "cohort_feature_sha256": digest.hexdigest(),
    }
    return groups, audit


def combine(model: str, clevr: dict, clevr_audit: dict,
            amber: dict, amber_audit: dict) -> tuple[dict, dict]:
    if clevr_audit["widths"] != amber_audit["widths"]:
        raise ValueError(f"{model}: CLEVR and AMBER feature widths differ")
    groups = {split: clevr[split] + amber[split] for split in ("train", "test")}
    keys = [row["key"] for rows in groups.values() for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: overlapping question keys across sources")
    images: dict[tuple[str, str], str] = {}
    for dataset, source in (("clevr", clevr), ("amber", amber)):
        for split, rows in source.items():
            for row in rows:
                identity = (dataset, str(row["image"]))
                if images.setdefault(identity, split) != split:
                    raise ValueError(f"{model}: source image leakage")
    counts = {
        split: {"real": sum(row["label"] == 1 for row in rows),
                "hallucination": sum(row["label"] == 0 for row in rows),
                "images": clevr_audit["counts"][split]["images"] + amber_audit["counts"][split]["images"]}
        for split, rows in groups.items()
    }
    if any(min(counts[split]["real"], counts[split]["hallucination"]) == 0 for split in groups):
        raise ValueError(f"{model}: mixed source is single-class")
    digest = hashlib.sha256(
        (clevr_audit["cohort_feature_sha256"] + ":" + amber_audit["cohort_feature_sha256"]).encode()
    ).hexdigest()
    audit = {
        "model": model,
        "source": "CLEVR9K Yes-only + AMBER existence Yes-only",
        "label_protocol": PROTOCOL,
        "split_protocol": "per_dataset_original_image_level_82",
        "sam_definition": clevr_audit["sam_definition"],
        "svar_definition": clevr_audit["svar_definition"],
        "counts": counts,
        "widths": clevr_audit["widths"],
        "cohort_feature_sha256": digest,
        "components": {"clevr": clevr_audit, "amber": amber_audit},
        "selection_warning": "All AMBER Yes-only samples are hallucinations; source dataset correlates with label.",
    }
    return groups, audit


def _predict_rows(model: str, method: str, seed: int, folder: Path,
                  target: str, cohort: str, records: list[dict]) -> dict:
    matrix = np.stack([row[method] for row in records]).astype(np.float32)
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    predict = _predict_qa if method == "sam" else predict_svar
    scores, threshold = predict(folder, matrix)
    return _metric_row(model, method, seed, target, cohort, labels, scores, threshold)


def evaluate(model: str, mixed: dict, clevr: dict, amber: dict,
             pope: dict, output: Path) -> list[dict]:
    rows = []
    for method in ("sam", "svar"):
        for seed in SEEDS:
            folder = output / model / method / f"seed_{seed}"
            mixed_row = _predict_rows(model, method, seed, folder,
                                      "clevr+amber", "test", mixed["test"])
            report = _json(folder / "result.json")
            native = (report["metrics"]["test"]["auroc"] if method == "sam"
                      else report["test_metrics"]["real_positive"]["auc"])
            if abs(native - mixed_row["auroc"]) > 2e-5:
                raise ValueError(f"{model}/{method}/seed_{seed}: mixed checkpoint replay failed")
            rows.append(mixed_row)
            rows.append(_predict_rows(model, method, seed, folder,
                                      "clevr", "test", clevr["test"]))
            rows.append(_predict_rows(model, method, seed, folder,
                                      "amber", "test", amber["test"]))
            predict = _predict_qa if method == "sam" else predict_svar
            scores, threshold = predict(folder, pope[method])
            for cohort, selected in (("all", np.ones(len(scores), dtype=bool)),
                                     ("test", pope["splits"] == "test")):
                rows.append(_metric_row(
                    model, method, seed, POPE, cohort,
                    pope["labels"][selected], scores[selected], threshold,
                ))
            print(f"[evaluate] {model}/{method}/seed_{seed}", flush=True)
    return rows


def write_outputs(model: str, audit: dict, pope: dict,
                  rows: list[dict], output: Path) -> None:
    folder = output / model
    _save_json(folder / "cohort_audit.json", {
        "mixed_source": audit,
        "pope_summary_path": pope["summary_path"],
    })
    _save_json(folder / "per_seed_results.json", {"rows": rows})
    with (folder / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: CLEVR＋AMBER Yes-only 混合训练 → POPE popular", "",
        "CLEVR 和 AMBER 各自沿用原图片级 8:2 划分；合并两者 train 样本训练。种子 43/44/45；SAM=物体词前 risk＋答案前 F_log，SVAR=物体词前第 5–18 层。来源训练集选择阈值，POPE 不训练、标准化或调阈值。均值 ± 总体标准差，单位 %。", "",
        "AMBER existence 的 GT 全为 No，因此 AMBER Yes-only 只含 hallucination。混合来源的 pooled AUC 可能利用数据集差异，须同时看 CLEVR test、AMBER test 和 POPE。AMBER 单独 AUC/AUPR 不定义。", "",
        "| 方法 | 测试集 | N (Real/Hall) | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy | 预测 Real 比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cohorts = (("clevr+amber", "test"), ("clevr", "test"), ("amber", "test"),
               (POPE, "all"), (POPE, "test"))
    for target, cohort in cohorts:
        for method in ("SAM", "SVAR"):
            selected = [row for row in rows if (row["target"], row["cohort"], row["method"]) ==
                        (target, cohort, method)]
            if len(selected) != len(SEEDS):
                raise ValueError(f"{model}/{target}/{cohort}/{method}: missing seeds")
            n, real, hall = selected[0]["n"], selected[0]["real_n"], selected[0]["hall_n"]
            if any((row["n"], row["real_n"], row["hall_n"]) != (n, real, hall) for row in selected):
                raise ValueError("Cohort changed across seeds")
            metrics = " | ".join(_fmt(selected, field) for field in (
                "auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
                "hall_recall", "accuracy", "pred_real_rate"))
            lines.append(f"| {method} | {target}/{cohort} | {n} ({real}/{hall}) | {metrics} |")
    lines.append("")
    (folder / "complete_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[complete] {folder / 'complete_results.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(WIDTHS))
    parser.add_argument("--clevr-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--questions-root", type=Path, default=QUESTIONS_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    clevr, clevr_audit = load_clevr(args.model, args.clevr_root)
    amber, amber_audit = load_amber(args.model, args.questions_root, args.target_root)
    mixed, audit = combine(args.model, clevr, clevr_audit, amber, amber_audit)
    print(f"[source] {args.model}: {audit['counts']}", flush=True)
    targets = load_targets(args.model, args.target_root, audit["widths"])
    pope = targets[POPE]
    pope_path = Path(pope["summary_path"]).parent / "sam_object_risk_answer_f_log_vs_svar_object_yesonly.npz"
    if sha256_file(pope_path) != _json(Path(pope["summary_path"]))["npz_sha256"]:
        raise ValueError(f"{args.model}: POPE target changed since feature export")
    train(args.model, mixed, audit, args.output_root)
    rows = evaluate(args.model, mixed, clevr, amber, pope, args.output_root)
    write_outputs(args.model, audit, pope, rows, args.output_root)


if __name__ == "__main__":
    main()
