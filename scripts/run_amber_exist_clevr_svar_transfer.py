#!/usr/bin/env python3
"""Evaluate native SVAR transfer on the AMBER-existence/CLEVR cohorts."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from detection.baselines import SVARMLP, torch_hallucination_scores
from detection.qa_probe import classification_metrics, label_for_protocol
from features.baseline import baseline_config, svar_training_vector
from features.qa_baseline import qa_image_identity
from scripts.evaluate_coco_clevr_log_s_transfer import _report
from scripts.run_amber_exist_clevr_position_transfer import (
    AMBER, CLEVR, CLEVR_NO, MODELS, PROTOCOL, SEEDS, _owner, _prepare,
)
from scripts.train_baselines import _train_svar
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/qa_benchmarks"
OUT = BASE / "amber_exist_clevr_svar_transfer_811"
CACHE = "LOGS-K32-TAU003-NATIVE6-FULL"
DATASETS = (AMBER, CLEVR, CLEVR_NO)
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def prepare(model: str, dataset: str) -> tuple[dict[str, list[dict]], dict]:
    # Use exactly the same valid question keys and image splits as the Log-S
    # transfer experiment, then take SVAR's native [5,19) layer/head vector.
    log_s_rows, log_s_audit = _prepare(model, dataset)
    method_rows = log_s_rows["sar_answer_ems_answer"]
    by_key = {str(row["key"]): row for row in method_rows}
    del log_s_rows, method_rows
    source_dataset = CLEVR if dataset == CLEVR_NO else dataset
    path = BASE / model / CACHE / source_dataset / "baseline" / PROTOCOL / "features.pkl"
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    groups = {split: [] for split in ("train", "val", "test")}
    counts = {split: {"real": 0, "hall": 0} for split in groups}
    owner = _owner(dataset)
    missed = 0
    for record in raw:
        position = record.get("positions", {}).get("prompt_last_token", record)
        key = str(position["key"])
        method = by_key.get(key)
        if method is None:
            continue
        label = label_for_protocol(position, PROTOCOL)
        if label != method["label"]:
            raise ValueError(f"{model}/{dataset}/{key}: baseline/method labels differ")
        split = owner[qa_image_identity(position)]
        if split != method["probe_split"]:
            raise ValueError(f"{model}/{dataset}/{key}: baseline/method image splits differ")
        try:
            vector = svar_training_vector(position["baselines"]["svar"], layer_start=5, layer_end=19)
        except (KeyError, TypeError, ValueError):
            missed += 1
            continue
        if vector.size == 0 or not np.isfinite(vector).all():
            missed += 1
            continue
        groups[split].append({
            "key": key, "label": label, "probe_split": split,
            "baseline_schema_version": position["baseline_schema_version"],
            "image_id": position["image_id"],
            "response_token_idx": position["response_token_idx"],
            "baselines": {"svar": {"vector": vector}},
        })
        counts[split]["real" if label else "hall"] += 1
    del raw
    kept = {row["key"] for rows in groups.values() for row in rows}
    if len(kept) != sum(len(rows) for rows in groups.values()):
        raise ValueError(f"{model}/{dataset}: duplicate SVAR question keys")
    if len(kept) + missed != len(by_key):
        raise ValueError(f"{model}/{dataset}: missing baseline rows: {len(by_key) - len(kept) - missed}")
    if any(min(values.values()) == 0 for values in counts.values()):
        raise ValueError(f"{model}/{dataset}: single-class SVAR split: {counts}")
    width = {row["baselines"]["svar"]["vector"].size for rows in groups.values() for row in rows}
    if len(width) != 1:
        raise ValueError(f"{model}/{dataset}: inconsistent SVAR vector widths: {width}")
    audit = {
        "model": model, "dataset": dataset, "subset": log_s_audit["subset"],
        "position": "prompt_last_token", "label_protocol": PROTOCOL,
        "valid_questions": len(kept), "missing_svar_features": missed,
        "log_s_cohort_sha256": log_s_audit["question_keys_sha256"],
        "split_class_counts": counts, "input_dim": next(iter(width)),
        "training_layers": "[5,19)",
    }
    return groups, audit


def test_matrix(groups: dict[str, list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    rows = groups["test"]
    matrix = np.stack([row["baselines"]["svar"]["vector"] for row in rows]).astype(np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    return matrix, labels


def predict(folder: Path, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    saved = torch.load(folder / "svar.pt", map_location="cpu", weights_only=False)
    if saved["input_dim"] != matrix.shape[1] or (saved["layer_start"], saved["layer_end_exclusive"]) != (5, 19):
        raise ValueError(f"Incompatible SVAR checkpoint: {folder}")
    model = SVARMLP(saved["input_dim"], saved["hidden_dim"])
    model.load_state_dict(saved["state_dict"])
    score = 1.0 - torch_hallucination_scores(model, matrix, torch.device("cpu"))
    if saved["threshold_score_class"] != "real":
        raise ValueError(f"Unexpected SVAR score convention: {folder}")
    return score, float(saved["threshold"])


def train(model: str, dataset: str, groups: dict[str, list[dict]], audit: dict) -> None:
    cfg_file = "model_configs_qa_logs_k32_tau003_amber.yaml" if dataset == AMBER else "model_configs_qa_logs_k32_tau003.yaml"
    cfg = {**baseline_config(load_config(str(ROOT / "configs" / cfg_file))), "_strict_82_no_validation": False}
    folder = OUT / model / dataset
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cohort_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for seed in SEEDS:
        run = folder / f"seed_{seed}"
        report_path = run / "result.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["cohort_audit"] != audit:
                raise ValueError(f"Incompatible existing SVAR result: {report_path}")
            print(f"Reuse {model}/{dataset}/seed{seed}", flush=True)
            continue
        run.mkdir(parents=True, exist_ok=True)
        report = _train_svar(groups, run, {**cfg, "seed": seed}, "cpu", "real")
        report_path.write_text(json.dumps({"cohort_audit": audit, **report}, indent=2) + "\n")
        print(f"Done {model}/{dataset}/seed{seed}: AUROC={report['test_metrics']['real_positive']['auc']:.4f}", flush=True)


def run() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, audits = [], []
    for model in MODELS:
        cached = {}
        for dataset in DATASETS:
            groups, audit = prepare(model, dataset)
            train(model, dataset, groups, audit)
            cached[dataset] = test_matrix(groups)
            audits.append(audit)
            del groups
        for seed in SEEDS:
            for source in DATASETS:
                folder = OUT / model / source / f"seed_{seed}"
                for target in DATASETS:
                    if source == CLEVR_NO and target == CLEVR:
                        continue
                    matrix, labels = cached[target]
                    score, threshold = predict(folder, matrix)
                    if source == target:
                        native = json.loads((folder / "result.json").read_text())["test_metrics"]
                        actual = classification_metrics(labels, score, threshold)
                        if abs(actual["auroc"] - native["real_positive"]["auc"]) > 2e-5:
                            raise ValueError(f"{model}/{source}/seed{seed}: native AUROC replay failed")
                        if abs(actual["real"]["f1"] - native["real_positive"]["f1"]) > 2e-5:
                            raise ValueError(f"{model}/{source}/seed{seed}: native F1 replay failed")
                    all_rows.append(_report(model, seed, source, target, "prompt_last_token", labels, score, threshold))
        print(f"{model}: SVAR transfer complete", flush=True)
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (OUT / "cohort_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    lines = [
        "# SVAR: AMBER existence ↔ CLEVR 检测器迁移", "",
        "与 Log-S 迁移实验使用相同的有效问题、答案正确性标签和图片级 8:1:1 划分。AMBER 只取 existence；CLEVR 报告全体及 GT=No 诊断子集。LLaVA-1.5-7B 和 Qwen3-VL-8B，各训练三次（种子 43/44/45）。", "",
        "SVAR 使用原生逐层逐头视觉注意力比特征（绝对层 5–18，输入 448 维）和 248 宽单隐藏层 MLP。跨域直接使用来源模型及来源验证集 Real-F1 阈值；目标域没有重训、重新标准化或调阈值。AUROC/AUPR 与阈值无关；F1/Accuracy 使用来源阈值。表中为均值 ± 总体标准差，单位 %。", "",
        "| 模型 | SVAR 训练集 | 测试集 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(rows: list[dict], key: str) -> str:
        values = 100 * np.asarray([row[key] for row in rows], dtype=np.float64)
        return f"{values.mean():.2f} ± {values.std():.2f}"
    display = {AMBER: "AMBER existence", CLEVR: "CLEVR 全部", CLEVR_NO: "CLEVR GT=No"}
    for model in MODELS:
        for source, target in ((AMBER, AMBER), (CLEVR, CLEVR), (AMBER, CLEVR), (CLEVR, AMBER),
                               (CLEVR_NO, CLEVR_NO), (AMBER, CLEVR_NO), (CLEVR_NO, AMBER)):
            selected = [row for row in all_rows if (row["model"], row["source"], row["target"]) == (model, source, target)]
            if len(selected) != len(SEEDS):
                raise ValueError(f"Missing SVAR seeds: {model}/{source}/{target}")
            n, real, hall = selected[0]["n"], selected[0]["real_n"], selected[0]["hall_n"]
            if any((row["n"], row["real_n"], row["hall_n"]) != (n, real, hall) for row in selected):
                raise ValueError(f"Different SVAR test cohort across seeds: {model}/{target}")
            lines.append(f"| {model} | {display[source]} | {display[target]} | {n} ({real}/{hall}) | " + " | ".join(fmt(selected, metric) for metric in METRICS) + " |")
    previous = BASE / "amber_exist_clevr_position_transfer_811" / "per_seed_results.csv"
    with previous.open(newline="") as handle:
        log_s_results = list(csv.DictReader(handle))
    lines.extend([
        "", "## 解释", "",
        "与同一测试问题上的 Log-S（SAR 答案位置 + EMS 答案位置）相比，完整 CLEVR 的双向迁移 AUROC 如下：", "",
        "| 模型 | 方向 | SVAR AUROC | Log-S AUROC |",
        "|---|---|---:|---:|",
    ])
    for model in MODELS:
        for source, target in ((AMBER, CLEVR), (CLEVR, AMBER)):
            svar_rows = [row for row in all_rows if (row["model"], row["source"], row["target"]) == (model, source, target)]
            log_s_source = "AMBER-existence" if source == AMBER else f"{CLEVR}/sar_answer_ems_answer"
            log_s_target = "AMBER-existence" if target == AMBER else CLEVR
            log_s_rows = [row for row in log_s_results if (row["model"], row["source"], row["target"], row["position"]) ==
                          (model, log_s_source, log_s_target, "sar_answer_ems_answer")]
            if len(svar_rows) != len(SEEDS) or len(log_s_rows) != len(SEEDS):
                raise ValueError(f"Missing matched SVAR/Log-S results: {model}/{source}/{target}")
            lines.append(f"| {model} | {display[source]} → {display[target]} | {fmt(svar_rows, 'auroc')} | {fmt(log_s_rows, 'auroc')} |")
    lines.extend([
        "",
        "Qwen3 的完整 CLEVR 测试集只有 23/810 个 Hall，SVAR 从 AMBER 迁入时 Hall AUPR 为 4.56%，因此较高的 AUROC 不代表少数 Hall 已得到可靠识别。LLaVA 反向迁移的 SVAR AUROC 低于随机水平，说明来源域学到的排序在目标域明显不稳定。",
        "",
        "AMBER existence 与 CLEVR GT=No 均存在答案极性标签捷径：Real 当且仅当模型回答 No。GT=No 按真值筛选，只用于诊断，不代表部署时可无标签选择。迁移表现应主要看完整 CLEVR。",
        "",
        "逐种子结果：`per_seed_results.csv`；样本审计：`cohort_audits.json`；各来源模型与阈值保存在 `<model>/<dataset>/seed_<seed>/`。", "",
    ])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    run()
