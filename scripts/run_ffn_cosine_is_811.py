#!/usr/bin/env python3
"""Train I+S+FFN cosine mean/max detectors on seeded COCO 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts.run_log_s_source_correspondence_ablation_811 import METRICS, MODELS, REFERENCE, SEEDS
from utils.io_utils import load_pkl


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
EXTRACTED = GF_ROOT / "outputs/coco4000_512_ffn_cosine_mean_max_811"
OUT = ROOT / "outputs/ffn_cosine_i_s_811"
CASES = ("i_s_ffn_cos_mean", "i_s_ffn_cos_max")
NAMES = {
    "i_s_ffn_cos_mean": "I + S + F_mean",
    "i_s_ffn_cos_max": "I + S + F_max",
    "sar_ems": "SAR + EMS (reference)",
}


def row_key(row: dict) -> tuple[int, int, int]:
    return (int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"]))


def prepare(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cache = OUT / model / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            data = {key: np.asarray(saved[key]) for key in saved.files}
        return {case: data[case] for case in CASES}, data["label"], data["image_id"]
    mechanism_path = ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"
    semantic_path = ROOT / "outputs/norm_semantic_transform_curves_by_label" / model / "compact_features.npz"
    extracted_path = EXTRACTED / model / "features.pkl"
    rows = load_pkl(extracted_path)
    by_key = {row_key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError(f"{model}: duplicate FFN cosine target key")
    with np.load(mechanism_path, allow_pickle=False) as source, np.load(semantic_path, allow_pickle=False) as semantic:
        labels = np.asarray(source["label"], dtype=np.int8)
        images = np.asarray(source["image_id"], dtype=np.int64)
        response = np.asarray(source["response_token_idx"], dtype=np.int32)
        token = np.asarray(source["target_token_id"], dtype=np.int32)
        layers = source["ems"].shape[1]
        visual_write = np.asarray(source["i_ems"][:, :layers], dtype=np.float32)
        semantic_mass = np.asarray(semantic["semantic_mass"], dtype=np.float32)
        if not (np.array_equal(images, semantic["image_id"]) and
                np.array_equal(labels, semantic["label"]) and
                visual_write.shape == semantic_mass.shape == (len(labels), layers)):
            raise ValueError(f"{model}: I/S metadata or dimensions differ")
        keys = list(zip(images.astype(int), response.astype(int), token.astype(int)))
        if len(set(keys)) != len(keys) or set(keys) != set(by_key):
            raise ValueError(f"{model}: extracted and reference target keys differ")
        aligned = [by_key[key] for key in keys]
        if any(int(row["label"]) != int(label) for row, label in zip(aligned, labels)):
            raise ValueError(f"{model}: extracted and reference labels differ")
        fmean = np.stack([np.asarray(row["ffn_cosine_mean"], dtype=np.float32) for row in aligned])
        fmax = np.stack([np.asarray(row["ffn_cosine_max"], dtype=np.float32) for row in aligned])
        if (fmean.shape != visual_write.shape or fmax.shape != visual_write.shape or
                np.any(fmax + 1e-6 < fmean)):
            raise ValueError(f"{model}: cosine shape or max/mean ordering invalid")
        matrices = {
            CASES[0]: np.concatenate((visual_write, semantic_mass, fmean), axis=1),
            CASES[1]: np.concatenate((visual_write, semantic_mass, fmax), axis=1),
        }
    if any(not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError(f"{model}: nonfinite input")
    audit = {
        "model": model, "samples": len(labels), "layers": layers,
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "input_order": ["I", "S", "F"],
        "I": "sum_m ||a_lm||",
        "S": "sum_m P(target | final_norm(hpre_visual_m))",
        "F_mean": "mean_m cos(o_ffn(object),hpre_visual_m)",
        "F_max": "max_m cos(o_ffn(object),hpre_visual_m)",
        "mechanism_source": str(mechanism_path),
        "semantic_source": str(semantic_path),
        "ffn_cosine_source": str(extracted_path),
        "fmean_range": [float(fmean.min()), float(fmean.max())],
        "fmax_range": [float(fmax.min()), float(fmax.max())],
    }
    output = OUT / model
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **matrices, label=labels, image_id=images,
                        response_token_idx=response, target_token_id=token)
    protocol.atomic_json(output / "alignment.json", audit)
    return matrices, labels, images


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    protocol.DISPLAY.update({case: NAMES[case] for case in CASES})
    matrices, labels, images = prepare(model)
    for seed in SEEDS:
        reference = json.loads((REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json").read_text())
        split_list = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
        token_counts = {name: int(np.isin(images, list(ids)).sum()) for name, ids in split.items()}
        if (reference["image_counts"] != {"train": 3200, "val": 400, "test": 400} or
                reference["token_counts"] != token_counts or sum(token_counts.values()) != len(labels)):
            raise ValueError(f"{model}/seed{seed}: reference cohort differs")
        for case in CASES:
            folder = OUT / model / f"seed{seed}" / case
            if (folder / "result.json").exists():
                saved = json.loads((folder / "result.json").read_text())
                if saved["token_counts"] != token_counts:
                    raise ValueError(f"{model}/seed{seed}/{case}: saved cohort differs")
                print(f"[{model}] reuse seed{seed} {case}", flush=True)
                continue
            protocol.train_case(
                model=model, case=case, matrix=matrices[case], labels=labels,
                image_ids=images, split=split, seed=seed, device=device,
                output_dir=folder, split_protocol="seeded_image_level_811",
            )
            print(f"[{model}] completed seed{seed} {case}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO 8:1:1：视觉来源 FFN 对齐均值与最大值", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；4000 图像、种子 43/44/45 同时控制 3200/400/400 图片划分与 MLP 初始化。输入分别为 [I,S,F_mean] 和 [I,S,F_max]，逐层拼接；F 是物体 token 的 FFN 输出与所有视觉 hpre 表示的原始余弦相似度，不使用 softmax 后的来源分布。I=∑ₘ||aₘ||，S=∑ₘP(yₜ|final_norm(hpre_m))。不拼接 SAR 或 EMS。",
        "所有方法使用相同 object token、标签、训练器；单隐藏层宽度 128、dropout 0.3、batch 256、学习率 0.001、weight decay 1e-5，训练集标准化、最小验证损失选 checkpoint、Real/Hall F1 阈值分别从验证集选取。指标为测试集百分数，± 为三种子的总体标准差。SAR+EMS 仅供参照。", "",
        "| 模型 | 特征 | AUROC | Real AUPR | Hall AUPR | Real F1 | Real Recall | Hall F1 | Hall Recall | Accuracy (Real threshold) | Accuracy (Hall threshold) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for case in (*CASES, "sar_ems"):
            values = []
            for seed in SEEDS:
                path = (REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json" if case == "sar_ems" else OUT / model / f"seed{seed}" / case / "result.json")
                result = json.loads(path.read_text())
                flat = protocol.flatten(result)
                values.append(flat)
                rows.append({"model": model, "case": case, "seed": seed, **flat})
            formatted = " | ".join(f"{100*np.mean([value[key] for value in values]):.2f} ± {100*np.std([value[key] for value in values]):.2f}" for key in METRICS)
            lines.append(f"| {model} | {NAMES[case]} | {formatted} |")
    lines += ["", "## 逐种子测试指标", "", "| 模型 | 特征 | 种子 | " + " | ".join(METRICS) + " |", "|---|---|---:|" + "---:|"*len(METRICS)]
    for row in rows:
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | " + " | ".join(f"{100*row[key]:.3f}" for key in METRICS) + " |")
    lines += ["", "特征审计见各模型的 `alignment.json`；检测器原始逐种子结果见各条件目录。", ""]
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
