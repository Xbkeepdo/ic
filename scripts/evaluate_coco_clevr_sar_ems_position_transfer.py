#!/usr/bin/env python3
"""Bidirectional COCO/CLEVR transfer for the two CLEVR SAR/EMS position probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from detection.qa_probe import build_matrix, classification_metrics
from scripts.evaluate_coco_clevr_log_s_transfer import (
    COCO, MODELS, PROTOCOL, QA, SEEDS, _coco_data, _predict_coco,
    _predict_qa, _report,
)
from scripts.train_clevr_sar_ems_position_ablation import (
    ANSWER, CASES, OUT as CLEVR_DETECTORS, _shared_rows,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/coco_clevr_sar_ems_position_transfer"
FIELDS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def _qa_test_matrices(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    prepared, audit = _shared_rows(model)
    saved_audit = json.loads((CLEVR_DETECTORS / model / "cohort_audit.json").read_text())
    if audit["shared_question_keys_sha256"] != saved_audit["shared_question_keys_sha256"]:
        raise ValueError(f"{model}: CLEVR cohort changed after training")
    matrices = {}
    labels = None
    keys = None
    for case in CASES:
        test_rows = [row for row in prepared[case] if row["probe_split"] == "test"]
        matrix, y, kept = build_matrix(test_rows, f"logs_risk_f_log@{ANSWER}", PROTOCOL)
        current_keys = [str(row["key"]) for row in kept]
        if labels is not None and (not np.array_equal(y, labels) or current_keys != keys):
            raise ValueError(f"{model}: CLEVR conditions do not share the same ordered test questions")
        matrices[case] = matrix
        labels, keys = y, current_keys
    if labels is None or keys is None:
        raise ValueError(f"{model}: empty CLEVR test cohort")
    return matrices, labels, keys


def _display(case: str) -> str:
    return "SAR答案+EMS答案" if case == CASES[0] else "SAR答案+EMS物体"


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    audits = []
    for model in MODELS:
        qa_matrices, qa_y, qa_keys = _qa_test_matrices(model)
        for seed in SEEDS:
            coco_native, coco_qa_order, coco_y = _coco_data(model, seed)
            coco_folder = COCO / model / f"seed{seed}" / "sar_ems"
            coco_scores, coco_threshold = _predict_coco(coco_folder, coco_native)
            with np.load(coco_folder / "scores.npz") as saved:
                if not np.array_equal(coco_y, saved["test_label"].astype(np.int64)):
                    raise ValueError(f"{model}/seed{seed}: COCO labels differ from saved test scores")
                coco_delta = float(np.max(np.abs((1.0 - coco_scores) - saved["test_hall_score"])))
                if coco_delta > 1e-5:
                    raise ValueError(f"{model}/seed{seed}: COCO checkpoint replay differs by {coco_delta}")
            all_rows.append(_report(model, seed, "COCO", "COCO", "object_token", coco_y, coco_scores, coco_threshold))
            audits.append({"model": model, "seed": seed, "source": "COCO", "checkpoint_replay_max_abs_delta": coco_delta})
            for case in CASES:
                qa_native = qa_matrices[case]
                half = qa_native.shape[1] // 2
                qa_coco_order = np.concatenate((qa_native[:, half:], qa_native[:, :half]), axis=1)
                qa_folder = CLEVR_DETECTORS / model / case / f"seed_{seed}"
                qa_scores, qa_threshold = _predict_qa(qa_folder, qa_native)
                native = json.loads((qa_folder / "result.json").read_text())
                if native["case"] != case or native["counts"]["test"] != len(qa_y):
                    raise ValueError(f"{model}/{case}/seed{seed}: checkpoint metadata differs")
                replay_auc = classification_metrics(qa_y, qa_scores, qa_threshold)["auroc"]
                qa_auc_delta = abs(replay_auc - native["metrics"]["test"]["auroc"])
                if qa_auc_delta > 2e-5:
                    raise ValueError(f"{model}/{case}/seed{seed}: CLEVR checkpoint replay AUC differs by {qa_auc_delta}")
                audits.append({"model": model, "seed": seed, "source": case, "checkpoint_replay_auc_delta": qa_auc_delta, "test_key_count": len(qa_keys)})
                all_rows.append(_report(model, seed, case, "CLEVR", case, qa_y, qa_scores, qa_threshold))
                coco_to_qa, _ = _predict_coco(coco_folder, qa_coco_order)
                all_rows.append(_report(model, seed, "COCO", "CLEVR", case, qa_y, coco_to_qa, coco_threshold))
                qa_to_coco, _ = _predict_qa(qa_folder, coco_qa_order)
                all_rows.append(_report(model, seed, case, "COCO", case, coco_y, qa_to_coco, qa_threshold))
        print(f"{model}: both directions and both CLEVR configurations completed", flush=True)

    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "replay_audit.json").write_text(json.dumps(audits, indent=2) + "\n")

    lines = [
        "# COCO ↔ CLEVR：SAR/EMS 位置组合的检测器迁移", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B。COCO 使用原始 object-token SAR+EMS 检测器；CLEVR 使用同一共享样本上重新训练的 `SAR答案+EMS答案` 与 `SAR答案+EMS物体` 两个检测器。CLEVR 标签为仅预测 Yes 时的目标物体是否真实存在；两数据集均以 Real=1。", "",
        "三种子 43/44/45。COCO 为按种子划分的图片级 8:1:1，CLEVR 为固定图片级 8:1:1；K=32、τ=0.03。直接加载来源权重、训练集 scaler 和来源验证集 Real-F1 阈值，不在目标集重训练或重新选阈值。QA 特征按来源模型的训练列顺序重排；两种 CLEVR 配置使用完全相同的测试问题。", "",
        "AUROC/AUPR 不依赖阈值，F1/Accuracy 使用来源阈值。下表为测试集三个种子的均值 ± 总体标准差，单位 %。Real/Hall AUPR 分别以对应类别为正类。COCO 是生成文本的物体 token，CLEVR 是 Yes 回答及问题中的物体位置，因此迁移同时改变数据、目标词和预测位置，不能单独解释为图像域迁移。", "",
        "| 模型 | 检测器来源 | 测试集 | CLEVR 特征方案 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(rows: list[dict], key: str) -> str:
        values = np.asarray([row[key] for row in rows], dtype=np.float64) * 100
        return f"{values.mean():.2f} ± {values.std():.2f}"
    delta_lines = []
    for model in MODELS:
        order = [("COCO", "COCO", "object_token")]
        for case in CASES:
            order.extend(((case, "CLEVR", case), ("COCO", "CLEVR", case), (case, "COCO", case)))
        for source, target, case in order:
            selected = [row for row in all_rows if row["model"] == model and row["source"] == source and row["target"] == target and row["position"] == case]
            if len(selected) != 3:
                raise ValueError(f"Missing seed result: {model}/{source}/{target}/{case}")
            counts = [(row["n"], row["real_n"], row["hall_n"]) for row in selected]
            n_min, n_max = min(row[0] for row in counts), max(row[0] for row in counts)
            count_str = f"{n_min}" if n_min == n_max else f"{n_min}–{n_max}"
            real_min, real_max = min(row[1] for row in counts), max(row[1] for row in counts)
            hall_min, hall_max = min(row[2] for row in counts), max(row[2] for row in counts)
            real_str = f"{real_min}" if real_min == real_max else f"{real_min}–{real_max}"
            hall_str = f"{hall_min}" if hall_min == hall_max else f"{hall_min}–{hall_max}"
            name = "物体 token" if case == "object_token" else _display(case)
            source_label = "COCO" if source == "COCO" else _display(source)
            lines.append(f"| {model} | {source_label} | {target} | {name} | {count_str} ({real_str}/{hall_str}) | " + " | ".join(fmt(selected, metric) for metric in FIELDS) + " |")
        for case in CASES:
            native_qa = [row for row in all_rows if row["model"] == model and row["source"] == case and row["target"] == "CLEVR"]
            cross_qa = [row for row in all_rows if row["model"] == model and row["source"] == "COCO" and row["target"] == "CLEVR" and row["position"] == case]
            native_coco = [row for row in all_rows if row["model"] == model and row["source"] == "COCO" and row["target"] == "COCO"]
            cross_coco = [row for row in all_rows if row["model"] == model and row["source"] == case and row["target"] == "COCO"]
            drop_to_qa = 100 * (np.mean([r["auroc"] for r in cross_qa]) - np.mean([r["auroc"] for r in native_qa]))
            drop_to_coco = 100 * (np.mean([r["auroc"] for r in cross_coco]) - np.mean([r["auroc"] for r in native_coco]))
            delta_lines.append(f"- {model} / {_display(case)}：COCO→CLEVR AUROC 相对 CLEVR 域内 {drop_to_qa:+.2f} pp；CLEVR→COCO 相对 COCO 域内 {drop_to_coco:+.2f} pp。")
    lines.extend(["", "## 相对域内 AUROC 变化", "", *delta_lines, "", "逐种子结果：`per_seed_results.csv`；来源检查点回放核验：`replay_audit.json`。Qwen3 CLEVR 测试集仅有 13 个 Hall，迁移性能的抽样不确定性较大。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
