#!/usr/bin/env python3
"""Evaluate the sum of all visual FFN-to-hpre cosine similarities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts.run_ffn_cosine_is_811 import OUT as MEAN_OUT
from scripts.run_ffn_cosine_top32_i_811 import OUT as TOP32_OUT
from scripts.run_ffn_cosine_top32_i_811 import iter_source_rows, key
from scripts.run_log_s_source_correspondence_ablation_811 import METRICS, REFERENCE, SEEDS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ffn_cosine_sum_i_s_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
CASE = "i_s_ffn_cos_sum"
NAMES = {
    "i_s": "I + S",
    CASE: "I + S + F_sum",
    "i_s_ffn_cos_mean": "I + S + F_mean",
    "i_s_ffn_cos_top32_i": "I + S + F_top32(I)",
    "sar_ems": "SAR + EMS (reference)",
}


def prepare(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = OUT / model
    cache = folder / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return (np.asarray(saved[CASE]), np.asarray(saved["label"]),
                    np.asarray(saved["image_id"]))

    with np.load(MEAN_OUT / model / "compact_features.npz", allow_pickle=False) as saved:
        mean_matrix = np.asarray(saved["i_s_ffn_cos_mean"], dtype=np.float32)
        labels = np.asarray(saved["label"], dtype=np.int8)
        images = np.asarray(saved["image_id"], dtype=np.int64)
        responses = np.asarray(saved["response_token_idx"], dtype=np.int32)
        tokens = np.asarray(saved["target_token_id"], dtype=np.int32)
    layers = mean_matrix.shape[1] // 3
    if mean_matrix.shape != (len(labels), 3*layers) or layers < 1:
        raise ValueError(f"{model}: invalid previous feature matrix")
    keys = list(zip(images.astype(int), responses.astype(int), tokens.astype(int)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate object token keys")
    audit = json.loads((TOP32_OUT / model / "alignment.json").read_text())
    if int(audit["samples"]) != len(keys) or int(audit["layers"]) != layers:
        raise ValueError(f"{model}: previous alignment audit differs")

    if model == "llava_1_5_7b":
        if int(audit["min_visual_tokens"]) != int(audit["max_visual_tokens"]):
            raise ValueError("LLaVA visual count is not constant")
        counts = np.full(len(keys), int(audit["min_visual_tokens"]), dtype=np.int32)
    else:
        by_key = {target_key: index for index, target_key in enumerate(keys)}
        counts = np.zeros(len(keys), dtype=np.int32)
        for row in iter_source_rows(model):
            row_key = key(row)
            index = by_key.get(row_key)
            if index is None:
                raise ValueError(f"{model}: unmatched source target {row_key}")
            if counts[index]:
                continue
            if int(row["label"]) != int(labels[index]):
                raise ValueError(f"{model}/{row_key}: label differs")
            source = row["matrices"]["source_hpre"]
            if source.ndim != 2 or source.shape[0] != layers:
                raise ValueError(f"{model}/{row_key}: invalid source matrix")
            counts[index] = int(source.shape[1])
        if (counts == 0).any():
            raise ValueError(f"{model}: {(counts == 0).sum()} targets have no visual count")
        if int(counts.min()) != int(audit["min_visual_tokens"]) or int(counts.max()) != int(audit["max_visual_tokens"]):
            raise ValueError(f"{model}: visual counts differ from prior full audit")
    fmean = mean_matrix[:, 2*layers:]
    fsum = (fmean.astype(np.float64) * counts[:, None]).astype(np.float32)
    if not np.isfinite(fsum).all():
        raise ValueError(f"{model}: nonfinite cosine sums")
    matrix = np.concatenate((mean_matrix[:, :2*layers], fsum), axis=1)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{CASE: matrix}, ffn_cosine_sum=fsum,
                        visual_count=counts, label=labels, image_id=images,
                        response_token_idx=responses, target_token_id=tokens)
    protocol.atomic_json(folder / "alignment.json", {
        "model": model,
        "samples": len(keys), "layers": layers,
        "min_visual_tokens": int(counts.min()),
        "max_visual_tokens": int(counts.max()),
        "definition": "sum_m cos(o_ffn(object),hpre(visual_m))",
        "computation": "visual_count * previously extracted all-visual cosine mean",
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "previous_features": str(MEAN_OUT / model / "compact_features.npz"),
    })
    print(f"[{model}] prepared {len(keys)} targets, visual count {counts.min()}–{counts.max()}", flush=True)
    return matrix, labels, images


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    protocol.DISPLAY[CASE] = NAMES[CASE]
    matrix, labels, images = prepare(model)
    for seed in SEEDS:
        split_list = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
        token_counts = {name: int(np.isin(images, list(ids)).sum()) for name, ids in split.items()}
        previous = json.loads((MEAN_OUT / model / f"seed{seed}" / "i_s_ffn_cos_mean" / "result.json").read_text())
        if previous["token_counts"] != token_counts or previous["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
            raise ValueError(f"{model}/seed{seed}: previous cohort differs")
        destination = OUT / model / f"seed{seed}" / CASE
        if (destination / "result.json").exists():
            saved = json.loads((destination / "result.json").read_text())
            if saved["token_counts"] != token_counts:
                raise ValueError(f"{model}/seed{seed}: saved cohort differs")
            continue
        protocol.train_case(model=model, case=CASE, matrix=matrix, labels=labels,
                            image_ids=images, split=split, seed=seed,
                            device=device, output_dir=destination,
                            split_protocol="seeded_image_level_811")
        print(f"[{model}] completed seed{seed}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO 8:1:1：FFN 视觉对齐相似度直接求和", "",
        "逐层 F_sum = Σ_m cos(o_ffn(object), hpre(visual_m))；模型为 LLaVA-1.5-7B 和 Qwen3-VL-8B。特征输入为 [I,S,F_sum]，其中 I=Σ_m||a_lm||、S=Σ_m P(y_t|final_norm(hpre_visual_m))。F_sum 由已保存的 F_mean 乘以每个样本的视觉 token 数构造，与原始 cosine 直接求和数学等价。", "",
        "沿用相同 ENDAC object token、种子 43/44/45 控制的 3200/400/400 图片划分与 MLP 初始化；单隐藏层宽 128、dropout 0.3、batch 256、学习率 0.001、weight decay 1e-5，训练集标准化，最小验证损失选 checkpoint。表中指标为测试集百分数；± 为三种子的总体标准差。", "",
        "| 模型 | 特征 | AUROC | Real AUPR | Hall AUPR | Real F1 | Real Recall | Hall F1 | Hall Recall | Accuracy (Real threshold) | Accuracy (Hall threshold) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        by_case = {}
        for case in ("i_s", "i_s_ffn_cos_mean", "i_s_ffn_cos_top32_i", CASE, "sar_ems"):
            values = []
            for seed in SEEDS:
                if case == "sar_ems":
                    path = REFERENCE / model / f"seed{seed}" / case / "result.json"
                elif case == "i_s_ffn_cos_mean":
                    path = MEAN_OUT / model / f"seed{seed}" / case / "result.json"
                elif case in ("i_s", "i_s_ffn_cos_top32_i"):
                    path = TOP32_OUT / model / f"seed{seed}" / case / "result.json"
                else:
                    path = OUT / model / f"seed{seed}" / case / "result.json"
                metrics = protocol.flatten(json.loads(path.read_text()))
                values.append(metrics)
                rows.append({"model": model, "case": case, "seed": seed, **metrics})
            by_case[case] = values
            formatted = " | ".join(
                f"{100*np.mean([v[metric] for v in values]):.2f} ± {100*np.std([v[metric] for v in values]):.2f}"
                for metric in METRICS
            )
            lines.append(f"| {model} | {NAMES[case]} | {formatted} |")
        deltas = [f"{100*np.mean([a[metric]-b[metric] for a,b in zip(by_case[CASE],by_case['i_s_ffn_cos_mean'])]):+.2f}"
                  for metric in METRICS]
        lines.append(f"| {model} | F_sum − F_mean (paired) | " + " | ".join(deltas) + " |")
    lines += ["", "## 逐种子测试指标（%）", "",
              "| 模型 | 特征 | 种子 | " + " | ".join(METRICS) + " |",
              "|---|---|---:|" + "---:|"*len(METRICS)]
    for row in rows:
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | " +
                     " | ".join(f"{100*row[metric]:.3f}" for metric in METRICS) + " |")
    lines += ["", "LLaVA 的视觉 token 数恒为 576，训练集标准化会抵消求和与均值之间的固定比例；Qwen3 的视觉 token 数随样本变化，求和会携带该数量信息。", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model required unless --summarize")


if __name__ == "__main__":
    main()
