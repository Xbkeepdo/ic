#!/usr/bin/env python3
"""Train and transfer Log-S probes between AMBER existence and CLEVR."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from detection.qa_probe import (
    build_matrix, classification_metrics, feature_vector, label_for_protocol,
)
from features.qa_baseline import qa_image_identity
from scripts.evaluate_coco_clevr_log_s_transfer import _predict_qa, _report
from scripts.train_qa_811_cached import _fit_method
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/qa_benchmarks"
OUT = BASE / "amber_exist_clevr_position_transfer_811"
CACHE = "LOGS-K32-TAU003-NATIVE6-FULL"
SPLITS = "LOGS-K32-TAU003-NATIVE6-FULL-811"
PROTOCOL = "answer_correctness_all"
AMBER = "amber_discriminative"
CLEVR = "clevr_exist_9k"
CLEVR_NO = "clevr_exist_9k_gt_no"
ANSWER = "prompt_last_token"
OBJECT = "question_object_pre_token"
AMBER_CASE = "sar_answer_ems_answer"
CLEVR_CASES = (AMBER_CASE, "sar_answer_ems_object")
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def _owner(dataset: str) -> dict[str, str]:
    source_dataset = CLEVR if dataset == CLEVR_NO else dataset
    path = BASE / SPLITS / "splits" / f"{source_dataset}.json"
    manifest = json.loads(path.read_text())
    if manifest["protocol"] != "strict_outer_image_level_811_from_existing_outer82":
        raise ValueError(f"Unexpected split protocol in {path}")
    return {image: split for split in ("train", "val", "test") for image in manifest["images"][split]}


def _prepare(model: str, dataset: str) -> tuple[dict[str, list[dict]], dict]:
    source_dataset = CLEVR if dataset == CLEVR_NO else dataset
    base = BASE / model / CACHE / source_dataset
    source = base / "features.pkl"
    with source.open("rb") as handle:
        records = pickle.load(handle)
    ground_truth = None
    if dataset == CLEVR_NO:
        with (base / "labels.jsonl").open() as handle:
            ground_truth = {
                str(row["key"]): str(row["gt_answer"]).lower()
                for row in (json.loads(line) for line in handle)
            }
    owner = _owner(dataset)
    cases = CLEVR_CASES if dataset != AMBER else (AMBER_CASE,)
    rows = {case: [] for case in cases}
    counts = {part: {"real": 0, "hall": 0} for part in ("train", "val", "test")}
    keys = []
    invalid = 0
    for record in records:
        if dataset == AMBER and record["source_split"] != "existence":
            continue
        if ground_truth is not None and ground_truth[str(record["key"])] != "no":
            continue
        label = label_for_protocol(record, PROTOCOL)
        try:
            answer = feature_vector(record, f"logs_risk_f_log@{ANSWER}")
            obj = feature_vector(record, f"logs_risk_f_log@{OBJECT}") if dataset != AMBER else None
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if answer.size == 0 or answer.size % 2 or not np.isfinite(answer).all():
            invalid += 1
            continue
        if dataset != AMBER and (obj is None or answer.shape != obj.shape or not np.isfinite(obj).all()):
            invalid += 1
            continue
        half = answer.size // 2
        vectors = {AMBER_CASE: answer}
        if dataset != AMBER:
            vectors[CLEVR_CASES[1]] = np.concatenate((obj[:half], answer[half:]))
        split = owner[qa_image_identity(record)]
        counts[split]["real" if label else "hall"] += 1
        key = str(record["key"])
        keys.append(key)
        for case, vector in vectors.items():
            rows[case].append({
                "key": key, "label": label, "probe_split": split,
                "positions": {ANSWER: {"dgst": {"logs_risk_f_log": vector.astype(np.float32)}}},
            })
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError(f"{model}/{dataset}: duplicate or missing question keys")
    if any(min(class_counts.values()) == 0 for class_counts in counts.values()):
        raise ValueError(f"{model}/{dataset}: one of train/val/test is single-class: {counts}")
    widths = {case: {len(row["positions"][ANSWER]["dgst"]["logs_risk_f_log"]) for row in case_rows} for case, case_rows in rows.items()}
    if len({next(iter(case_widths)) for case_widths in widths.values()}) != 1 or any(len(case_widths) != 1 for case_widths in widths.values()):
        raise ValueError(f"{model}/{dataset}: inconsistent feature widths: {widths}")
    audit = {
        "model": model, "dataset": dataset,
        "subset": "existence" if dataset == AMBER else "gt_no" if dataset == CLEVR_NO else "all",
        "label_protocol": PROTOCOL,
        "feature_order": "concat(EMS risk, SAR F_log)",
        "valid_questions": len(keys), "excluded_invalid_features": invalid,
        "question_keys_sha256": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
        "split_class_counts": counts,
        "feature_dimensions": {case: next(iter(case_widths)) for case, case_widths in widths.items()},
        "cases": {case: {"SAR": ANSWER, "EMS": ANSWER if case == AMBER_CASE else OBJECT} for case in cases},
    }
    return rows, audit


def _train(model: str, dataset: str, rows: dict[str, list[dict]], audit: dict) -> None:
    cfg_file = ("model_configs_qa_logs_k32_tau003_amber.yaml" if dataset == AMBER
                else "model_configs_qa_logs_k32_tau003.yaml")
    cfg = load_config(str(ROOT / "configs" / cfg_file))["training"]["torch_probe"]
    folder = OUT / model / dataset
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cohort_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for case, case_rows in rows.items():
        for seed in SEEDS:
            output = folder / case / f"seed_{seed}"
            path = output / "result.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if (saved.get("case") != case or saved.get("cohort_sha256") != audit["question_keys_sha256"]
                        or saved.get("label_protocol") != PROTOCOL):
                    raise ValueError(f"Incompatible existing checkpoint: {path}")
                print(f"Reuse {model}/{dataset}/{case}/seed{seed}", flush=True)
                continue
            report = _fit_method(
                case_rows, model_name=model, dataset_name=dataset,
                feature_set=f"logs_risk_f_log@{ANSWER}", protocol=PROTOCOL,
                seed=seed, cfg=cfg, device="cpu", output=output,
            )
            report["case"] = case
            report["feature_set"] = case
            report["feature_components"] = audit["cases"][case]
            report["dataset_subset"] = audit["subset"]
            report["cohort_sha256"] = audit["question_keys_sha256"]
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Done {model}/{dataset}/{case}/seed{seed}: AUROC={report['metrics']['test']['auroc']:.4f}", flush=True)


def _test_matrices(rows: dict[str, list[dict]]) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    matrices = {}
    labels, keys = None, None
    for case, case_rows in rows.items():
        selected = [row for row in case_rows if row["probe_split"] == "test"]
        matrix, y, kept = build_matrix(selected, f"logs_risk_f_log@{ANSWER}", PROTOCOL)
        current_keys = [str(row["key"]) for row in kept]
        if labels is not None and (not np.array_equal(y, labels) or current_keys != keys):
            raise ValueError("Cases use different test questions")
        matrices[case], labels, keys = matrix, y, current_keys
    if labels is None or keys is None:
        raise ValueError("Empty test cohort")
    return matrices, labels, keys


def _evaluate(cached: dict[tuple[str, str], tuple[dict[str, list[dict]], dict]]) -> None:
    results, replay = [], []
    for model in MODELS:
        amber_rows, amber_audit = cached[(model, AMBER)]
        amber, amber_y, amber_keys = _test_matrices(amber_rows)
        if len(amber_keys) != amber_audit["split_class_counts"]["test"]["real"] + amber_audit["split_class_counts"]["test"]["hall"]:
            raise ValueError("AMBER test count differs from prepared cohort")
        clevr_tests = {}
        for dataset in (CLEVR, CLEVR_NO):
            rows, audit = cached[(model, dataset)]
            matrices, labels, keys = _test_matrices(rows)
            if len(keys) != sum(audit["split_class_counts"]["test"].values()):
                raise ValueError(f"{dataset} test count differs from prepared cohort")
            clevr_tests[dataset] = (matrices, labels)
        for seed in SEEDS:
            amber_folder = OUT / model / AMBER / AMBER_CASE / f"seed_{seed}"
            amber_scores, amber_threshold = _predict_qa(amber_folder, amber[AMBER_CASE])
            amber_native = json.loads((amber_folder / "result.json").read_text())
            amber_replay_delta = abs(classification_metrics(amber_y, amber_scores, amber_threshold)["auroc"] - amber_native["metrics"]["test"]["auroc"])
            if amber_replay_delta > 2e-5:
                raise ValueError(f"{model}/seed{seed}: AMBER checkpoint replay differs by {amber_replay_delta}")
            replay.append({"model": model, "seed": seed, "source": "AMBER existence", "replay_auroc_delta": amber_replay_delta})
            results.append(_report(model, seed, "AMBER-existence", "AMBER-existence", AMBER_CASE, amber_y, amber_scores, amber_threshold))
            for dataset, (clevr, clevr_y) in clevr_tests.items():
                for case in CLEVR_CASES:
                    clevr_folder = OUT / model / dataset / case / f"seed_{seed}"
                    clevr_scores, clevr_threshold = _predict_qa(clevr_folder, clevr[case])
                    native = json.loads((clevr_folder / "result.json").read_text())
                    clevr_replay_delta = abs(classification_metrics(clevr_y, clevr_scores, clevr_threshold)["auroc"] - native["metrics"]["test"]["auroc"])
                    if clevr_replay_delta > 2e-5:
                        raise ValueError(f"{model}/{dataset}/{case}/seed{seed}: CLEVR checkpoint replay differs by {clevr_replay_delta}")
                    source_name = f"{dataset}/{case}"
                    replay.append({"model": model, "seed": seed, "source": source_name, "replay_auroc_delta": clevr_replay_delta})
                    results.append(_report(model, seed, source_name, dataset, case, clevr_y, clevr_scores, clevr_threshold))
                    amber_to_clevr, _ = _predict_qa(amber_folder, clevr[case])
                    results.append(_report(model, seed, "AMBER-existence", dataset, case, clevr_y, amber_to_clevr, amber_threshold))
                    clevr_to_amber, _ = _predict_qa(clevr_folder, amber[AMBER_CASE])
                    results.append(_report(model, seed, source_name, "AMBER-existence", case, amber_y, clevr_to_amber, clevr_threshold))
        print(f"{model}: bidirectional AMBER-existence/CLEVR transfer complete", flush=True)
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (OUT / "replay_audit.json").write_text(json.dumps(replay, indent=2) + "\n")
    _summarize(results, cached)


def _summarize(results: list[dict], cached: dict) -> None:
    lines = [
        "# AMBER existence ↔ CLEVR 检测器迁移", "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B；AMBER 只用 `source_split=existence` 问题，CLEVR 分别使用 Exist-9K 全部有效问题和标准答案为 No 的子集。两数据集均使用 `answer_correctness_all`，Real=答案正确，Hall=答案错误。AMBER existence 的 Yes-only 子集全部为 Hall，无法作为二分类训练集，所以这里不能沿用上一轮 CLEVR 的 Yes-only 检测器；本轮两端均按答案正确性重新训练。", "",
        "两端特征均为 Log-S [EMS,SAR]、K=32、τ=0.03。CLEVR 分为 SAR答案+EMS答案 和 SAR答案+EMS物体；AMBER 仅有答案位置。各域保持官方图片级 8:1:1，种子 43/44/45 控制 MLP 初始化。直接迁移来源检查点、训练 scaler、来源验证集 Real-F1 阈值；不在目标数据集重新训练、重新标准化或调阈值。", "",
        "AUROC/AUPR 与阈值无关，F1/Accuracy 使用来源阈值。结果为三个种子的测试均值 ± 总体标准差，单位 %。Real/Hall AUPR 分别以对应类别为正类。AMBER existence 的正确答案均为 No，因此另报告 CLEVR `GT=No` 子集的对齐诊断；此子集按真值筛选，仅用于分析，不能视为无需真值的部署协议。已核对两模型标签文件：在 AMBER existence 和 CLEVR `GT=No` 上，Real 当且仅当模型回答 No，单用答案极性即可得到完美分类；接近 100% 的分数不能证明一般幻觉检测的跨域泛化。只有两端都采用 SAR答案+EMS答案时特征位置一致；混合位置是额外的位置变化实验。", "",
        "| 模型 | 检测器训练集 | 测试集 | CLEVR 方案 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(rows: list[dict], metric: str) -> str:
        values = 100 * np.asarray([row[metric] for row in rows], dtype=np.float64)
        return f"{values.mean():.2f} ± {values.std():.2f}"
    order = [("AMBER-existence", "AMBER-existence", AMBER_CASE)]
    for dataset in (CLEVR, CLEVR_NO):
        for case in CLEVR_CASES:
            order.extend(((f"{dataset}/{case}", dataset, case), ("AMBER-existence", dataset, case), (f"{dataset}/{case}", "AMBER-existence", case)))
    for model in MODELS:
        for source, target, case in order:
            selected = [row for row in results if row["model"] == model and row["source"] == source and row["target"] == target and row["position"] == case]
            if len(selected) != 3:
                raise ValueError(f"Missing three seeds: {model}/{source}/{target}/{case}")
            counts = {(row["n"], row["real_n"], row["hall_n"]) for row in selected}
            if len(counts) != 1:
                raise ValueError(f"Different target test cohort: {model}/{source}/{target}/{case}")
            n, real, hall = next(iter(counts))
            display_case = "SAR答案+EMS答案" if case == AMBER_CASE else "SAR答案+EMS物体"
            display_source = "AMBER existence" if source == "AMBER-existence" else f"{'CLEVR GT=No' if source.startswith(CLEVR_NO) else 'CLEVR 全部'} / {display_case}"
            display_target = "AMBER existence" if target == "AMBER-existence" else "CLEVR GT=No" if target == CLEVR_NO else "CLEVR 全部"
            lines.append(f"| {model} | {display_source} | {display_target} | {display_case} | {n} ({real}/{hall}) | " + " | ".join(fmt(selected, metric) for metric in METRICS) + " |")
    lines.extend(["", "## Cohort", ""])
    for model in MODELS:
        for dataset in (AMBER, CLEVR, CLEVR_NO):
            audit = cached[(model, dataset)][1]
            lines.append(f"- {model} / {dataset}: train/val/test={','.join(str(sum(audit['split_class_counts'][s].values())) for s in ('train','val','test'))}; test Real/Hall={audit['split_class_counts']['test']['real']}/{audit['split_class_counts']['test']['hall']}; invalid features excluded={audit['excluded_invalid_features']}.")
    lines.extend(["", "逐种子结果：`per_seed_results.csv`；回放审计：`replay_audit.json`；各模型与数据集的样本审计：`<model>/<dataset>/cohort_audit.json`。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    cached = {}
    for model in MODELS:
        for dataset in (AMBER, CLEVR, CLEVR_NO):
            rows, audit = _prepare(model, dataset)
            cached[(model, dataset)] = (rows, audit)
            _train(model, dataset, rows, audit)
    _evaluate(cached)


if __name__ == "__main__":
    main()
