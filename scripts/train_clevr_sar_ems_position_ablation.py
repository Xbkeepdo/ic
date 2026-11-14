#!/usr/bin/env python3
"""Train CLEVR Yes-only SAR/EMS position combinations on one shared cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from detection.qa_probe import feature_vector, label_for_protocol
from features.qa_baseline import qa_image_identity
from scripts.train_qa_811_cached import _fit_method
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/qa_benchmarks/clevr_sar_ems_position_ablation_811"
QA = ROOT / "outputs/qa_benchmarks"
DATASET = "clevr_exist_9k"
CACHE_RUN = "LOGS-K32-TAU003-NATIVE6-FULL"
SPLIT_RUN = "LOGS-K32-TAU003-NATIVE6-FULL-811"
PROTOCOL = "object_hallucination_yes_only"
ANSWER = "prompt_last_token"
OBJECT = "question_object_pre_token"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
CASES = ("sar_answer_ems_answer", "sar_answer_ems_object")
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def _split_owner() -> dict[str, str]:
    manifest = json.loads((QA / SPLIT_RUN / "splits" / f"{DATASET}.json").read_text())
    if manifest["protocol"] != "strict_outer_image_level_811_from_existing_outer82":
        raise ValueError("Unexpected CLEVR split protocol")
    return {identity: part for part in ("train", "val", "test") for identity in manifest["images"][part]}


def _shared_rows(model: str) -> tuple[dict[str, list[dict]], dict]:
    source = QA / model / CACHE_RUN / DATASET / "features.pkl"
    with source.open("rb") as handle:
        rows = pickle.load(handle)
    owner = _split_owner()
    prepared = {case: [] for case in CASES}
    counts = {part: {"real": 0, "hall": 0} for part in ("train", "val", "test")}
    keys = []
    excluded = 0
    for row in rows:
        label = label_for_protocol(row, PROTOCOL)
        if label is None:
            continue
        try:
            answer = feature_vector(row, f"logs_risk_f_log@{ANSWER}")
            obj = feature_vector(row, f"logs_risk_f_log@{OBJECT}")
        except (KeyError, TypeError, ValueError):
            excluded += 1
            continue
        if answer.shape != obj.shape or answer.size % 2 or not answer.size:
            excluded += 1
            continue
        if not np.isfinite(answer).all() or not np.isfinite(obj).all():
            excluded += 1
            continue
        half = answer.size // 2
        # Both source vectors are serialized as [EMS risk, SAR F_log].
        vectors = {
            CASES[0]: answer,
            CASES[1]: np.concatenate((obj[:half], answer[half:])),
        }
        part = owner[qa_image_identity(row)]
        counts[part]["real" if label else "hall"] += 1
        key = str(row["key"])
        keys.append(key)
        for case, vector in vectors.items():
            prepared[case].append({
                "key": key,
                "prediction": row["prediction"],
                "object_hallucination_yes_only_label": label,
                "probe_split": part,
                "positions": {ANSWER: {"dgst": {"logs_risk_f_log": vector.astype(np.float32)}}},
            })
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError(f"{model}: duplicate or empty shared question keys")
    for part, classes in counts.items():
        if min(classes.values()) == 0:
            raise ValueError(f"{model}/{part}: missing a class")
    widths = {case: {len(row["positions"][ANSWER]["dgst"]["logs_risk_f_log"]) for row in prepared[case]} for case in CASES}
    if len({tuple(value) for value in widths.values()}) != 1:
        raise ValueError(f"{model}: feature widths differ by case: {widths}")
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    audit = {
        "model": model, "dataset": DATASET, "protocol": PROTOCOL,
        "shared_question_keys_sha256": digest,
        "shared_counts": counts, "valid_questions": len(keys),
        "excluded_yes_questions_missing_either_position": excluded,
        "feature_dimensions": {case: next(iter(widths[case])) for case in CASES},
        "feature_order": "concat(EMS risk, SAR F_log)",
        "cases": {
            CASES[0]: {"EMS": ANSWER, "SAR": ANSWER},
            CASES[1]: {"EMS": OBJECT, "SAR": ANSWER},
        },
    }
    return prepared, audit


def _train(model: str, device: str) -> None:
    prepared, audit = _shared_rows(model)
    model_out = OUT / model
    model_out.mkdir(parents=True, exist_ok=True)
    (model_out / "cohort_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    config = load_config(str(ROOT / "configs/model_configs_qa_logs_k32_tau003.yaml"))
    cfg = config["training"]["torch_probe"]
    for case in CASES:
        for seed in SEEDS:
            output = model_out / case / f"seed_{seed}"
            result_path = output / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved.get("case") != case or saved.get("cohort_sha256") != audit["shared_question_keys_sha256"]:
                    raise ValueError(f"Incompatible previous result: {result_path}")
                saved["feature_set"] = case
                saved["position"] = "answer" if case == CASES[0] else "sar_answer_ems_object"
                saved["trained_feature_selector"] = f"logs_risk_f_log@{ANSWER}"
                result_path.write_text(json.dumps(saved, indent=2) + "\n")
                print(f"Reuse {model}/{case}/seed_{seed}", flush=True)
                continue
            report = _fit_method(
                prepared[case], model_name=model, dataset_name=DATASET,
                feature_set=f"logs_risk_f_log@{ANSWER}", protocol=PROTOCOL,
                seed=seed, cfg=cfg, device=device, output=output,
            )
            report["case"] = case
            report["feature_set"] = case
            report["position"] = "answer" if case == CASES[0] else "sar_answer_ems_object"
            report["trained_feature_selector"] = f"logs_risk_f_log@{ANSWER}"
            report["feature_components"] = audit["cases"][case]
            report["cohort_sha256"] = audit["shared_question_keys_sha256"]
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Done {model}/{case}/seed_{seed}: AUROC={report['metrics']['test']['auroc']:.4f}", flush=True)


def _summarize() -> None:
    rows = []
    lines = [
        "# CLEVR 上 SAR/EMS 位置组合对比", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；CLEVR-Exist-9K 的 `object_hallucination_yes_only` 标签，只保留模型回答 Yes 的问题，Real=1。答案位置指 `prompt_last_token`，物体位置指 `question_object_pre_token`。", "",
        "两方案严格使用同时具有两种位置特征的相同问题、固定图片级 8:1:1 划分；种子 43/44/45 初始化单隐藏层 128 MLP。输入均为 [EMS, SAR]，K=32、τ=0.03；训练集 z-score，验证损失选 checkpoint，验证集 Real-F1 选阈值。数值为测试集三种子均值 ± 总体标准差，单位 %。", "",
        "| 模型 | SAR 位置 | EMS 位置 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        audit = json.loads((OUT / model / "cohort_audit.json").read_text())
        for case in CASES:
            for seed in SEEDS:
                path = OUT / model / case / f"seed_{seed}" / "result.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                report = json.loads(path.read_text())
                if report["cohort_sha256"] != audit["shared_question_keys_sha256"]:
                    raise ValueError(f"Cohort mismatch: {path}")
                test = report["metrics"]["test"]
                rows.append({
                    "model": model, "case": case, "seed": seed,
                    "test_n": report["counts"]["test"],
                    "test_real": report["class_counts"]["test"][1],
                    "test_hall": report["class_counts"]["test"][0],
                    "best_epoch": report["best_epoch"],
                    "threshold": report["threshold"],
                    "auroc": test["auroc"],
                    "real_aupr": test["real_aupr"],
                    "hall_aupr": test["hallucination_aupr"],
                    "real_f1": test["real"]["f1"],
                    "hall_f1": test["hallucination"]["f1"],
                    "accuracy": test["accuracy"],
                })
            selected = [row for row in rows if row["model"] == model and row["case"] == case]
            if len(selected) != 3 or len({(row["test_n"], row["test_real"], row["test_hall"]) for row in selected}) != 1:
                raise ValueError(f"Unequal test cohort: {model}/{case}")
            def fmt(metric: str) -> str:
                vals = np.asarray([row[metric] for row in selected]) * 100
                return f"{vals.mean():.2f} ± {vals.std():.2f}"
            row = selected[0]
            lines.append(f"| {model} | 答案 | {'答案' if case == CASES[0] else '物体'} | {row['test_n']} ({row['test_real']}/{row['test_hall']}) | " + " | ".join(fmt(metric) for metric in METRICS) + " |")
        first = [row for row in rows if row["model"] == model and row["case"] == CASES[0]]
        second = [row for row in rows if row["model"] == model and row["case"] == CASES[1]]
        diffs = {metric: 100 * np.asarray([b[metric] - a[metric] for a, b in zip(first, second)]) for metric in METRICS}
        lines.append(f"| **{model} Δ（物体 EMS−答案 EMS）** | — | — | 同上 | " + " | ".join(f"{diffs[m].mean():+.2f}" for m in METRICS) + " |")
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines.extend(["", "逐种子数据：`per_seed_results.csv`；各模型共享样本审计：`<model>/cohort_audit.json`。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS + ("all",), default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for model in MODELS if args.model == "all" else (args.model,):
        _train(model, args.device)
    if args.model == "all":
        _summarize()


if __name__ == "__main__":
    main()
