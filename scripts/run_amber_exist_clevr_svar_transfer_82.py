#!/usr/bin/env python3
"""Native SVAR transfer on the fixed outer image-level 8:2 QA split."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from detection.qa_probe import classification_metrics
from features.baseline import baseline_config
from scripts.evaluate_coco_clevr_log_s_transfer import _report
from scripts.run_amber_exist_clevr_svar_transfer import (
    AMBER, BASE, CACHE, CLEVR, CLEVR_NO, DATASETS, METRICS, MODELS, OUT as OUT_811,
    PROTOCOL, ROOT, SEEDS, predict, prepare as prepare_811, test_matrix,
)
from scripts.train_baselines import _train_svar
from utils.config_utils import load_config


OUT = BASE / "amber_exist_clevr_svar_transfer_82"


def prepare(model: str, dataset: str) -> tuple[dict[str, list[dict]], dict]:
    # The previous experiment defines the exact Log-S-valid, SVAR-valid
    # question cohort. Reassign those questions to the project's original
    # strict outer 8:2 split rather than its derived 8:1:1 partition.
    old_groups, old_audit = prepare_811(model, dataset)
    source_dataset = CLEVR if dataset == CLEVR_NO else dataset
    manifest_path = BASE / model / CACHE / source_dataset / "baseline" / PROTOCOL / "qa_probe_splits.json"
    if not manifest_path.exists():
        manifest_path = manifest_path.with_name("probe_splits.prompt_last_token.json")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["label_protocol"] != PROTOCOL or manifest["image_counts"]["val"] != 0
            or manifest["question_counts"]["val"] != 0):
        raise ValueError(f"Expected a strict 8:2 split: {manifest_path}")
    owner = {str(key): split for split in ("train", "test") for key in manifest[split]}
    if len(owner) != sum(manifest["question_counts"][split] for split in ("train", "test")):
        raise ValueError(f"Duplicate question keys in {manifest_path}")
    groups = {"train": [], "val": [], "test": []}
    counts = {split: {"real": 0, "hall": 0} for split in ("train", "test")}
    for row in (record for rows in old_groups.values() for record in rows):
        split = owner[row["key"]]
        row = {**row, "probe_split": split}
        groups[split].append(row)
        counts[split]["real" if row["label"] else "hall"] += 1
    for split in ("train", "test"):
        if min(counts[split].values()) == 0:
            raise ValueError(f"Single-class 8:2 partition: {model}/{dataset}/{split}: {counts[split]}")
    keys = sorted(row["key"] for split in ("train", "test") for row in groups[split])
    if len(keys) != old_audit["valid_questions"] or len(keys) != len(set(keys)):
        raise ValueError(f"8:2 question cohort changed for {model}/{dataset}")
    audit = {
        "model": model, "dataset": dataset, "subset": old_audit["subset"],
        "position": "prompt_last_token", "label_protocol": PROTOCOL,
        "split_protocol": "strict_outer_image_level_82",
        "split_manifest": str(manifest_path),
        "valid_questions": len(keys),
        "question_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
        "split_class_counts": counts, "input_dim": old_audit["input_dim"],
        "training_layers": "[5,19)",
    }
    if audit["question_keys_sha256"] != old_audit["log_s_cohort_sha256"]:
        raise ValueError(f"8:2 and 8:1:1 valid question keys differ for {model}/{dataset}")
    return groups, audit


def train(model: str, dataset: str, groups: dict[str, list[dict]], audit: dict) -> None:
    cfg_file = "model_configs_qa_logs_k32_tau003_amber.yaml" if dataset == AMBER else "model_configs_qa_logs_k32_tau003.yaml"
    cfg = {**baseline_config(load_config(str(ROOT / "configs" / cfg_file))), "_strict_82_no_validation": True}
    folder = OUT / model / dataset
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cohort_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    strict_groups = {**groups, "val": groups["train"]}
    for seed in SEEDS:
        run = folder / f"seed_{seed}"
        report_path = run / "result.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["cohort_audit"] != audit or report["paper_config"]["selection_protocol"] != "fixed_last_epoch_train_f1_threshold":
                raise ValueError(f"Incompatible existing SVAR 8:2 result: {report_path}")
            print(f"Reuse {model}/{dataset}/seed{seed}", flush=True)
            continue
        run.mkdir(parents=True, exist_ok=True)
        report = _train_svar(strict_groups, run, {**cfg, "seed": seed}, "cpu", "real")
        report_path.write_text(json.dumps({"cohort_audit": audit, **report}, indent=2) + "\n")
        print(f"Done {model}/{dataset}/seed{seed}: AUROC={report['test_metrics']['real_positive']['auc']:.4f}", flush=True)


def run() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits = [], []
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
                    rows.append(_report(model, seed, source, target, "prompt_last_token", labels, score, threshold))
        print(f"{model}: SVAR 8:2 transfer complete", flush=True)
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (OUT / "cohort_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    display = {AMBER: "AMBER existence", CLEVR: "CLEVR 全部", CLEVR_NO: "CLEVR GT=No"}
    def fmt(group: list[dict], metric: str) -> str:
        values = 100 * np.asarray([row[metric] for row in group], dtype=np.float64)
        return f"{values.mean():.2f} ± {values.std():.2f}"
    lines = [
        "# SVAR: AMBER existence ↔ CLEVR 检测器迁移（8:2）", "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B；沿用项目原始图片级严格 8:2 外层划分，没有验证集。AMBER 只用 existence 问题，CLEVR 报告全部及 GT=No 诊断子集。使用与 8:1:1 迁移实验完全相同的有效问题和 `answer_correctness_all` 标签（Real=答案正确）。", "",
        "SVAR 原生逐层逐头视觉注意力比特征：层 5–18、448 维、248 宽单隐藏层 MLP。固定图片划分（seed 42），43/44/45 控制训练初始化与 minibatch 顺序。按严格 8:2 协议训练满 50 epoch、使用最后一轮权重，并只在来源训练集选择 Real-F1 阈值；目标域不重训、不重标准化、不调阈值。AUROC/AUPR 与阈值无关，F1/Accuracy 使用来源训练阈值。表中为均值 ± 总体标准差，单位 %。", "",
        "| 模型 | SVAR 训练集 | 测试集 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ((AMBER, AMBER), (CLEVR, CLEVR), (AMBER, CLEVR), (CLEVR, AMBER),
             (CLEVR_NO, CLEVR_NO), (AMBER, CLEVR_NO), (CLEVR_NO, AMBER))
    for model in MODELS:
        for source, target in order:
            selected = [row for row in rows if (row["model"], row["source"], row["target"]) == (model, source, target)]
            if len(selected) != len(SEEDS):
                raise ValueError(f"Missing SVAR seeds: {model}/{source}/{target}")
            counts = {(row["n"], row["real_n"], row["hall_n"]) for row in selected}
            if len(counts) != 1:
                raise ValueError(f"Changing target cohort across seeds: {model}/{source}/{target}")
            n, real, hall = next(iter(counts))
            lines.append(f"| {model} | {display[source]} | {display[target]} | {n} ({real}/{hall}) | " + " | ".join(fmt(selected, metric) for metric in METRICS) + " |")
    lines.extend([
        "", "## 解释", "",
        "完整 CLEVR 应作为主要跨域检验。CLEVR GT=No 是依据真值挑选的诊断子集，不能用于部署式性能结论；AMBER existence 和该子集均有答案极性标签捷径。Real/Hall AUPR 应结合对应测试集正类占比解读。",
        "", "逐种子：`per_seed_results.csv`；队列与划分审计：`cohort_audits.json`。8:1:1 对照见 `../amber_exist_clevr_svar_transfer_811/complete_results.md`；两种划分的测试集规模不同，不宜直接把数字变化归因于阈值或模型。", "",
    ])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    run()
