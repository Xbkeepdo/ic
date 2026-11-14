#!/usr/bin/env python3
"""Evaluate saved COCO and CLEVR Log-S detectors across datasets.

No detector is refitted. Each source checkpoint keeps its own training-split
scaler and validation-selected Real-positive decision threshold.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from detection.qa_probe import QAProbe, build_matrix, classification_metrics
from scripts.train_torch_probe_feature_sets import DGSTStyleProbe


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/coco_clevr_log_s_detector_transfer"
COCO = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
COCO_FEATURES = ROOT / "outputs/log_s_mechanism_ablation_811"
QA = ROOT / "outputs/qa_benchmarks"
QA_RUN = "LOGS-K32-TAU003-NATIVE6-FULL-811"
QA_CACHE_RUN = "LOGS-K32-TAU003-NATIVE6-FULL"
DATASET = "clevr_exist_9k"
PROTOCOL = "object_hallucination_yes_only"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
POSITIONS = ("prompt_last_token", "question_object_pre_token")
SEEDS = (43, 44, 45)
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def _coco_data(model: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(COCO_FEATURES / model / "compact_features.npz", allow_pickle=False) as data:
        matrix = np.asarray(data["sar_ems"], dtype=np.float32)
        labels = np.asarray(data["label"], dtype=np.int64)
        image_ids = np.asarray(data["image_id"], dtype=np.int64)
    split = json.loads((COCO / "splits" / f"seed{seed}.json").read_text())
    selected = np.isin(image_ids, split["test"])
    if matrix.shape[1] not in (64, 72) or not selected.any():
        raise ValueError(f"{model} COCO features or split invalid")
    # Native COCO order: [SAR, EMS]. Native QA order: [EMS, SAR].
    native = matrix[selected]
    half = native.shape[1] // 2
    qa_order = np.concatenate((native[:, half:], native[:, :half]), axis=1)
    return native, qa_order, labels[selected]


def _qa_data(model: str, position: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = QA / model / QA_CACHE_RUN / DATASET / "features.pkl"
    with path.open("rb") as handle:
        records = pickle.load(handle)
    split_path = QA / QA_RUN / "splits" / f"{DATASET}.json"
    test_keys = set(json.loads(split_path.read_text())["question_keys"]["test"])
    rows = [row for row in records if str(row["key"]) in test_keys]
    matrix, labels, kept = build_matrix(rows, f"logs_risk_f_log@{position}", PROTOCOL)
    if matrix.shape[1] not in (64, 72) or len(kept) == 0:
        raise ValueError(f"{model}/{position}: QA features or split invalid")
    half = matrix.shape[1] // 2
    coco_order = np.concatenate((matrix[:, half:], matrix[:, :half]), axis=1)
    return matrix, coco_order, labels


def _predict_coco(folder: Path, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    config = json.loads((folder / "config.json").read_text())
    with np.load(folder / "input_scaler.npz") as scaler:
        standardized = ((matrix.astype(np.float64) - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError(f"Nonfinite COCO-standardized features: {folder}")
    net = DGSTStyleProbe(matrix.shape[1], (128,), 0.3, batch_norm=False)
    net.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    net.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(net(torch.from_numpy(standardized)).squeeze(-1)).numpy()
    threshold = float(json.loads((folder / "result.json").read_text())["thresholds"]["real_score"])
    if config["input_dim"] != matrix.shape[1]:
        raise ValueError("COCO checkpoint dimension mismatch")
    return scores, threshold


def _predict_qa(folder: Path, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    report = json.loads((folder / "result.json").read_text())
    saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    standardized = ((matrix - saved["mean"]) / saved["std"]).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError(f"Nonfinite QA-standardized features: {folder}")
    cfg = report["training_config"]
    net = QAProbe(matrix.shape[1], tuple(cfg["hidden_sizes"]), float(cfg["dropout"]), batch_norm=bool(cfg["batch_norm"]))
    net.load_state_dict(saved["state_dict"])
    net.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(net(torch.from_numpy(standardized))).numpy()
    if abs(float(saved["threshold"]) - float(report["threshold"])) > 1e-12:
        raise ValueError("QA threshold mismatch")
    return scores, float(saved["threshold"])


def _report(model: str, seed: int, source: str, target: str, position: str,
            labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    result = classification_metrics(labels, scores, threshold)
    return {
        "model": model, "seed": seed, "source": source, "target": target,
        "position": position, "n": len(labels), "real_n": int(labels.sum()),
        "hall_n": int(len(labels) - labels.sum()), "source_threshold": threshold,
        "auroc": result["auroc"], "real_aupr": result["real_aupr"],
        "hall_aupr": result["hallucination_aupr"],
        "real_f1": result["real"]["f1"], "hall_f1": result["hallucination"]["f1"],
        "real_recall": result["real"]["recall"],
        "hall_recall": result["hallucination"]["recall"],
        "accuracy": result["accuracy"],
    }


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    audit = []
    for model in MODELS:
        qa = {position: _qa_data(model, position) for position in POSITIONS}
        for seed in SEEDS:
            coco_native, coco_qa_order, coco_y = _coco_data(model, seed)
            coco_folder = COCO / model / f"seed{seed}" / "sar_ems"
            coco_scores, coco_threshold = _predict_coco(coco_folder, coco_native)
            with np.load(coco_folder / "scores.npz") as saved:
                if not np.array_equal(coco_y, saved["test_label"].astype(np.int64)):
                    raise ValueError(f"{model}/seed{seed}: COCO labels differ from saved evaluation")
                delta = float(np.max(np.abs((1 - coco_scores) - saved["test_hall_score"])))
                if delta > 1e-5:
                    raise ValueError(f"{model}/seed{seed}: COCO prediction replay delta {delta}")
                audit.append({"model": model, "seed": seed, "source": "COCO", "max_prediction_delta": delta})
            all_rows.append(_report(model, seed, "COCO", "COCO", "object_token", coco_y, coco_scores, coco_threshold))
            for position, (qa_native, qa_coco_order, qa_y) in qa.items():
                qa_folder = QA / model / QA_RUN / DATASET / "method" / PROTOCOL / position / f"seed_{seed}"
                qa_scores, qa_threshold = _predict_qa(qa_folder, qa_native)
                native = json.loads((qa_folder / "result.json").read_text())
                if len(qa_y) != native["counts"]["test"]:
                    raise ValueError(f"{model}/seed{seed}/{position}: QA test count mismatch")
                native_auc_delta = abs(classification_metrics(qa_y, qa_scores, qa_threshold)["auroc"] - native["metrics"]["test"]["auroc"])
                if native_auc_delta > 2e-5:
                    raise ValueError(f"{model}/seed{seed}/{position}: QA prediction replay AUC delta {native_auc_delta}")
                audit.append({"model": model, "seed": seed, "source": f"CLEVR/{position}", "auroc_delta": native_auc_delta})
                all_rows.append(_report(model, seed, f"CLEVR/{position}", "CLEVR", position, qa_y, qa_scores, qa_threshold))
                transferred_to_qa, _ = _predict_coco(coco_folder, qa_coco_order)
                all_rows.append(_report(model, seed, "COCO", "CLEVR", position, qa_y, transferred_to_qa, coco_threshold))
                transferred_to_coco, _ = _predict_qa(qa_folder, coco_qa_order)
                all_rows.append(_report(model, seed, f"CLEVR/{position}", "COCO", position, coco_y, transferred_to_coco, qa_threshold))
        print(f"{model}: completed 3 seeds, 2 CLEVR positions", flush=True)
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "replay_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# COCO ↔ CLEVR detector transfer (Log-S SAR+EMS)", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；保存的单隐层检测器直接迁移，不在目标数据集重新训练、标准化或选阈值。来源检查点在验证集选模型和 Real-F1 阈值。COCO 为生成文本 object token 的 ENDAC 标签；CLEVR 为预测 Yes 的问题中目标物体是否存在。两者均以 Real=1。", "",
        "COCO 使用按种子 43/44/45 重新划分的图片级 8:1:1 测试集；CLEVR 使用固定图片级 8:1:1 测试集，种子 43/44/45 只影响检测器初始化。CLEVR 分别在 prompt-last 和 question-object-pre 位置计算特征；K=32、τ=0.03。特征严格重排为来源模型训练时的顺序。", "",
        "AUROC/AUPR 不依赖阈值；F1/Accuracy 用来源验证集选出的 Real 阈值。数值为三个种子的均值 ± 总体标准差，单位 %。Real/Hall AUPR 分别以对应类别为正类。COCO 检测生成的物体 token，CLEVR 检测 Yes 回答所指的物体；因此下表同时检验图像域和预测位置/目标词变化，不能单独归因为图像域迁移。两数据集的阳性率不同，跨域 AUPR 应结合 Real 比例解读。", "",
        "| 模型 | 检测器训练集 | 评估集 | CLEVR 位置 | 样本数 | Real 比例 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordering = [("COCO", "COCO", "object_token")]
    for position in POSITIONS:
        ordering.extend(((f"CLEVR/{position}", "CLEVR", position), ("COCO", "CLEVR", position), (f"CLEVR/{position}", "COCO", position)))
    for model in MODELS:
        for source, target, position in ordering:
            selected = [row for row in all_rows if row["model"] == model and row["source"] == source and row["target"] == target and row["position"] == position]
            if len(selected) != len(SEEDS):
                raise ValueError(f"Incomplete results {model}/{source}/{target}/{position}")
            def cell(key: str) -> str:
                values = np.asarray([row[key] for row in selected], dtype=np.float64) * 100
                return f"{values.mean():.2f} ± {values.std():.2f}"
            counts = np.asarray([row["n"] for row in selected])
            count_text = str(counts[0]) if np.all(counts == counts[0]) else f"{counts.min()}–{counts.max()}"
            real_pct = np.mean([100 * row["real_n"] / row["n"] for row in selected])
            lines.append(f"| {model} | {source} | {target} | {position} | {count_text} | {real_pct:.1f}% | " + " | ".join(cell(key) for key in ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")) + " |")
    lines.extend(["", "逐种子数据：`per_seed_results.csv`；检查点回放核验：`replay_audit.json`。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
