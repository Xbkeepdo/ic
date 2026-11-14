#!/usr/bin/env python3
"""Compare QA 8:1:1 Log-S AUROC with the published local 8:2 baseline tables."""

from __future__ import annotations

import json
import pickle
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from detection.qa_probe import QAProbe, build_matrix
from features.qa_baseline import qa_image_identity
from scripts.train_qa_811_cached import _assignment


BASE = ROOT / "outputs" / "qa_benchmarks"
OLD = "LOGS-K32-TAU003-NATIVE6-FULL"
NEW = f"{OLD}-811"
MODELS = (
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
)
METHODS = ("MetaToken", "DHCP", "SVAR", "ProjectAway", "TokenGrounding", "InsLens")
CATEGORIES = ("existence", "attribute", "relation")
SEEDS = (43, 44, 45)


def old_clevr() -> dict[str, dict[str, float]]:
    path = BASE / OLD / "clevr_answer_correctness_prompt_last_comparison.md"
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] not in {display for _, display in MODELS}:
            continue
        # Published columns: Ours, MetaToken, DHCP, SVAR, ProjectAway,
        # TokenGrounding, InsLens. Each cell is AUROC / Correct-AUPR.
        values = [float(re.search(r"\d+\.\d+", cell).group()) * 100 for cell in cells[1:]]
        if len(values) != len(METHODS) + 1:
            raise ValueError(f"Unexpected CLEVR baseline columns: {line}")
        own_std = re.search(r"±(\d+\.\d+)", cells[1])
        if own_std is None:
            raise ValueError(f"Missing CLEVR Log-S std: {line}")
        result[cells[0]] = {
            "Log-S": values[0], "Log-S std": float(own_std.group(1)) * 100,
            **dict(zip(METHODS, values[1:])),
        }
    if set(result) != {display for _, display in MODELS}:
        raise ValueError("Incomplete 8:2 CLEVR source table")
    return result


def old_amber() -> dict[str, dict[str, dict[str, float]]]:
    path = BASE / OLD / "clevr_amber3_auc_tables.tex"
    section = path.read_text().split(
        "Comparison of hallucination detection on AMBER-Discriminative", 1
    )[1].split(r"\midrule", 1)[1].split(r"\rowcolor", 1)[0]
    order = (
        "LLaVA-1.5-7B", "Qwen2.5-VL-7B", "InternVL2.5-8B", "Qwen3-VL-8B"
    )
    result = {name: {category: {} for category in CATEGORIES} for name in order}
    for source_method, method in (("MetaToken", "MetaToken"), ("SVAR", "SVAR"),
                                  ("ProjectAway", "ProjectAway"),
                                  ("TokenGrounding", "TokenGrounding"),
                                  ("InsLen", "InsLens")):
        match = re.search(rf"(?ms)^{source_method}\n(.*?)\\\\", section)
        if match is None:
            raise ValueError(f"Missing {source_method} in 8:2 AMBER source table")
        values = [float(value) for value in re.findall(r"\d+\.\d{2}", match.group(1))]
        if len(values) != len(order) * len(CATEGORIES):
            raise ValueError(f"Unexpected AMBER values for {method}: {values}")
        for index, value in enumerate(values):
            model = order[index // len(CATEGORIES)]
            category = CATEGORIES[index % len(CATEGORIES)]
            result[model][category][method] = value
    own_section = path.read_text().split(
        "Comparison of hallucination detection on AMBER-Discriminative", 1
    )[1].split(r"\rowcolor", 1)[1].split(r"\bottomrule", 1)[0]
    own_cells = own_section.split("&")[2:]
    if len(own_cells) != len(order) * len(CATEGORIES):
        raise ValueError("Unexpected 8:2 AMBER Log-S columns")
    for index, cell in enumerate(own_cells):
        values = [float(value) for value in re.findall(r"\d+\.\d{2}", cell)]
        if len(values) != 2:
            raise ValueError(f"Unexpected 8:2 AMBER Log-S mean/std: {cell}")
        model = order[index // len(CATEGORIES)]
        category = CATEGORIES[index % len(CATEGORIES)]
        result[model][category]["Log-S"] = values[0]
        result[model][category]["Log-S std"] = values[1]
    return result


def own_amber() -> dict[str, dict[str, tuple[float, float, int]]]:
    result = {}
    assignment = _assignment("amber_discriminative")
    summary = json.loads((BASE / NEW / "complete_results.json").read_text())
    for model, display in MODELS:
        with (BASE / model / OLD / "amber_discriminative" / "features.pkl").open("rb") as handle:
            rows = pickle.load(handle)
        test = [row for row in rows if assignment[qa_image_identity(row)] == "test"]
        X, y, kept = build_matrix(
            test, "logs_risk_f_log@prompt_last_token", "answer_correctness_all"
        )
        expected = summary[
            f"{model}/amber_discriminative/prompt_last_token/answer_correctness_all"
        ]["test_counts"]["method"]
        if len(kept) != expected:
            raise ValueError(f"AMBER test cohort mismatch for {model}: {len(kept)} != {expected}")
        masks = {category: np.asarray([row["source_split"] == category for row in kept])
                 for category in CATEGORIES}
        scores = {category: [] for category in CATEGORIES}
        for seed in SEEDS:
            root = (BASE / model / NEW / "amber_discriminative" / "method"
                    / "answer_correctness_all" / "prompt_last_token" / f"seed_{seed}")
            report = json.loads((root / "result.json").read_text())
            saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
            config = report["training_config"]
            probe = QAProbe(X.shape[1], tuple(config["hidden_sizes"]),
                            float(config["dropout"]), batch_norm=bool(config["batch_norm"]))
            probe.load_state_dict(saved["state_dict"])
            probe.eval()
            features = ((X - saved["mean"]) / saved["std"]).astype(np.float32)
            with torch.no_grad():
                real_score = torch.sigmoid(probe(torch.from_numpy(features))).numpy()
            overall = roc_auc_score(y, real_score)
            if abs(overall - report["metrics"]["test"]["auroc"]) > 1e-6:
                raise ValueError(f"AMBER overall AUROC mismatch: {model}/seed_{seed}")
            for category, mask in masks.items():
                if set(y[mask]) != {0, 1}:
                    raise ValueError(f"Single-class AMBER test cohort: {model}/{category}")
                scores[category].append(roc_auc_score(1 - y[mask], 1 - real_score[mask]))
        result[display] = {
            category: (statistics.mean(values) * 100,
                       statistics.pstdev(values) * 100, int(masks[category].sum()))
            for category, values in scores.items()
        }
    return result


def main() -> None:
    old_clevr_scores, old_amber_scores = old_clevr(), old_amber()
    new_amber_scores = own_amber()
    new_summary = json.loads((BASE / NEW / "complete_results.json").read_text())
    header = ["模型", "任务", "测试题数 8:1:1 / 8:2", "Log-S 8:1:1",
              *(f"{method} 8:2" for method in METHODS)]
    lines = [
        "# QA：Log-S 8:1:1 与 baseline 8:2 的 AUROC 对照",
        "",
        "所有数值均为 **AUROC（%）**。Log-S 显示 seeds 43/44/45 的 mean ± population std；"
        "8:2 baseline 使用原报告数值。CLEVR 使用 `prompt_last_token` + "
        "`answer_correctness_all`；AMBER 使用 `prompt_last_token` + `answer_correctness_all`，"
        "再按 existence / attribute / relation 分别计算幻觉 AUROC。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for model, display in MODELS:
        key = f"{model}/clevr_exist_9k/prompt_last_token/answer_correctness_all"
        own = new_summary[key]["detectors"]["log_s"]["auc"]
        rows = [("CLEVR-Exist-9K", 900, 1800,
                 own["mean"] * 100, own["std"] * 100, old_clevr_scores[display])]
        for category, old_count in zip(CATEGORIES, (987, 1500, 344)):
            mean, std, count = new_amber_scores[display][category]
            rows.append((f"AMBER {category}", count, old_count, mean, std,
                         old_amber_scores[display][category]))
        for task, new_count, old_count, mean, std, old_scores in rows:
            if (new_count, old_count) not in ((900, 1800), (498, 987),
                                              (766, 1500), (173, 344)):
                raise ValueError(f"Unexpected test counts: {display}/{task}")
            cells = [display, task, f"{new_count} / {old_count}",
                     f"{mean:.2f}±{std:.2f}"]
            cells += [f"{old_scores[method]:.2f}" if method in old_scores else "—"
                      for method in METHODS]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "8:1:1 的测试集是原 8:2 测试图片的一部分，表中两侧使用的测试题不同；"
        "因此这些数值是跨划分并列展示，不能当作同题配对提升。8:1:1 在 validation 上选 checkpoint；"
        "8:2 不使用 validation 选点。有训练权重的 baseline 按原方案取最后 epoch。"
        "AUROC 与分类阈值无关。",
        "",
        "CLEVR 的 MetaToken 使用原对比报告的最后层 LR、无 StandardScaler 配置；"
        "AMBER 的 MetaToken 使用原报告的标准化配置。AMBER 原 8:2 三类表未报告 DHCP，故标为“—”。",
        "",
        "来源：[CLEVR 8:2 对比](../LOGS-K32-TAU003-NATIVE6-FULL/clevr_answer_correctness_prompt_last_comparison.md)、"
        "[AMBER 8:2 三类原表](../LOGS-K32-TAU003-NATIVE6-FULL/clevr_amber3_auc_tables.tex)、"
        "[Log-S 8:1:1 总表](complete_results.md)。AMBER 8:1:1 三类指标由已保存的 Log-S "
        "checkpoint 在新测试图片上重新推理，整体 AUROC 逐 seed 与总表核对一致。",
    ]
    target = BASE / NEW / "mixed_811_ours_82_baselines_clevr_amber3.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(MODELS) * 4} rows: {target}")


if __name__ == "__main__":
    main()
