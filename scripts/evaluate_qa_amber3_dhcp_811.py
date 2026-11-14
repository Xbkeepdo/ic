#!/usr/bin/env python3
"""Evaluate saved 8:1:1 QA DHCP checkpoints on AMBER's three question types."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from detection.baselines import DHCPMLP, evaluate_detection_scores  # noqa: E402
from features.baseline import DHCPShardReader, get_baseline_payload  # noqa: E402
from features.qa_baseline import baseline_record_for_position, qa_image_identity  # noqa: E402
from scripts.summarize_qa_811_ours_vs_82_baselines import own_amber  # noqa: E402
from scripts.train_qa_811_cached import _assignment  # noqa: E402


BASE = ROOT / "outputs" / "qa_benchmarks"
OLD = "LOGS-K32-TAU003-NATIVE6-FULL"
NEW = f"{OLD}-811"
MODELS = (
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
)
CATEGORIES = ("existence", "attribute", "relation")
SEEDS = (43, 44, 45)


def evaluate(model_name: str, device: str, assignment: dict[str, str]) -> dict:
    source = (BASE / model_name / OLD / "amber_discriminative" / "baseline"
              / "answer_correctness_all")
    with (source / "features.pkl").open("rb") as handle:
        records = pickle.load(handle)
    selected = []
    for record in records:
        if assignment[qa_image_identity(record)] != "test":
            continue
        positioned = baseline_record_for_position(record, "prompt_last_token")
        if positioned is None:
            raise ValueError(f"Missing prompt-last record: {model_name}/{record['key']}")
        selected.append(positioned)
    if len(selected) != 1437 or len({row["key"] for row in selected}) != len(selected):
        raise ValueError(f"Unexpected 8:1:1 AMBER test cohort: {model_name}/{len(selected)}")
    categories = np.asarray([row["source_split"] for row in selected])
    labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
    masks = {category: categories == category for category in CATEGORIES}
    expected_counts = {"existence": 498, "attribute": 766, "relation": 173}
    for category, mask in masks.items():
        if int(mask.sum()) != expected_counts[category] or set(labels[mask]) != {0, 1}:
            raise ValueError(f"Unexpected AMBER category cohort: {model_name}/{category}")
    output = (BASE / model_name / NEW / "amber_discriminative" / "baseline"
              / "answer_correctness_all" / "prompt_last_token")
    first = torch.load(output / "seed_43" / "dhcp" / "dhcp.pt",
                       map_location="cpu", weights_only=False)
    shape = tuple(int(value) for value in first["input_shape"])
    reader = DHCPShardReader(source / "dhcp" / "shards")
    attention = np.empty((len(selected), *shape), dtype=np.float16)
    for index, row in enumerate(selected):
        payload = get_baseline_payload(row, "dhcp")
        values = (payload["attention"] if "attention" in payload else
                  reader.load(payload.get("shard_reference", payload)))
        values = np.asarray(values, dtype=np.float16)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid DHCP attention: {model_name}/{row['key']}")
        attention[index] = values
    input_tensor = torch.from_numpy(attention).to(device)
    del attention
    runs = {}
    for seed in SEEDS:
        root = output / f"seed_{seed}"
        saved = first if seed == 43 else torch.load(root / "dhcp" / "dhcp.pt",
                                                     map_location="cpu", weights_only=False)
        if tuple(saved["input_shape"]) != shape:
            raise ValueError(f"DHCP input shape changed: {model_name}/seed_{seed}")
        report = json.loads((root / "result.json").read_text())
        baseline = report["methods"]["dhcp"]
        if (report["split_protocol"] != "strict_outer_image_level_811"
                or report["counts"]["test"] != len(selected)
                or baseline["selection_protocol"] != "validation_selected"):
            raise ValueError(f"Unexpected DHCP protocol: {model_name}/seed_{seed}")
        model = DHCPMLP(math.prod(shape), hidden_dim=int(saved["hidden_dim"]))
        model.load_state_dict(saved["state_dict"])
        model.to(device).eval()
        batch_size = int(baseline["paper_config"]["batch_size"])
        chunks = []
        with torch.no_grad():
            for values in input_tensor.split(batch_size):
                logits = model(values.float())
                chunks.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        scores = np.concatenate(chunks)
        threshold = float(saved["threshold"])
        if abs(threshold - baseline["threshold"]) > 1e-12:
            raise ValueError(f"DHCP threshold mismatch: {model_name}/seed_{seed}")
        overall = evaluate_detection_scores(labels, scores, threshold, positive_class="real")
        for metric in ("auc", "aupr", "f1"):
            reference = baseline["test_metrics"]["hallucination_positive"][metric]
            if abs(overall["hallucination_positive"][metric] - reference) > 1e-5:
                raise ValueError(f"DHCP overall {metric} mismatch: {model_name}/seed_{seed}")
        runs[str(seed)] = {
            "overall_auroc": overall["hallucination_positive"]["auc"],
            "threshold_real": threshold,
            "categories": {
                category: evaluate_detection_scores(
                    labels[mask], scores[mask], threshold, positive_class="real"
                )
                for category, mask in masks.items()
            },
        }
        print(f"{model_name}/seed_{seed}: overall AUROC matched", flush=True)
        del model
    return {
        "test_rows": len(selected),
        "category_rows": {category: int(mask.sum()) for category, mask in masks.items()},
        "class_counts": {
            category: {"hallucination": int((labels[mask] == 0).sum()),
                       "real": int((labels[mask] == 1).sum())}
            for category, mask in masks.items()
        },
        "seeds": runs,
    }


def summarize(results: dict) -> None:
    own = own_amber()
    lines = [
        "# AMBER 三类幻觉：DHCP 8:1:1",
        "",
        "四模型均复用正式 8:1:1 DHCP 验证集选出的 checkpoint 和阈值，"
        "仅在该划分的 test 图片上按 existence / attribute / relation 分项评估。"
        "每个 seed 的整体 AUROC、Hall AUPR、Hall F1 已与原 `result.json` 核对一致。"
        "数值为 seeds 43/44/45 的 mean±population std。",
        "",
        "| 模型 | AMBER 类型 | Test questions | Log-S AUROC (%) | DHCP AUROC (%) | DHCP Hall AUPR (%) | DHCP Hall F1 (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, display in MODELS:
        result = results[model_name]
        for category in CATEGORIES:
            values = [result["seeds"][str(seed)]["categories"][category]
                      for seed in SEEDS]
            formatted = []
            for metric in ("auc", "aupr", "f1"):
                scores = [value["hallucination_positive"][metric] * 100
                          for value in values]
                formatted.append(f"{statistics.mean(scores):.2f}±{statistics.pstdev(scores):.2f}")
            own_mean, own_std, own_count = own[display][category]
            count = result["category_rows"][category]
            if own_count != count:
                raise ValueError(f"Log-S/DHCP category count differs: {model_name}/{category}")
            lines.append("| " + " | ".join((
                display, category, str(count), f"{own_mean:.2f}±{own_std:.2f}",
                *formatted,
            )) + " |")
    lines += [
        "",
        "Log-S 与 DHCP 使用同一 8:1:1 测试图片和 `answer_correctness_all` 标签。"
        "Hall F1 使用在整个 AMBER validation 上选定的 Real-F1 阈值，未针对子类重调。"
        "DHCP checkpoint 来自 8:1:1 [完整结果](complete_results.md)；"
        "Log-S 的类别 AUROC 由对应 checkpoint 重新推理得到。",
    ]
    root = BASE / NEW
    (root / "amber3_dhcp_811.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "amber3_dhcp_811.json").write_text(json.dumps(results, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"Wrote {root / 'amber3_dhcp_811.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    assignment = _assignment("amber_discriminative")
    results = {model: evaluate(model, args.device, assignment) for model, _ in MODELS}
    summarize(results)


if __name__ == "__main__":
    main()
