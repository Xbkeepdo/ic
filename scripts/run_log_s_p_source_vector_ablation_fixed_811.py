#!/usr/bin/env python3
"""Compare four EMS source vectors on a fixed COCO 8:1:1 image split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts import run_log_s_p_source_vector_ablation_811 as source_ablation


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/log_s_mechanism_ablation_811"
OUT = ROOT / "outputs/log_s_p_source_vector_ablation_811_fixed_split"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
MODES = source_ablation.MODES
DISPLAY = source_ablation.DISPLAY
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def result_path(model: str, seed: int, mode: str) -> Path:
    if mode == "ffn_output":
        return REFERENCE / model / "sar_ems" / f"seed{seed}" / "result.json"
    if mode == "pre_ffn_residual":
        return REFERENCE / model / "sar_ems_ffn_input_source" / f"seed{seed}" / "result.json"
    return OUT / model / mode / f"seed{seed}" / "result.json"


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    base = source_ablation.load_base(model)
    new = source_ablation.load_new(model, base)
    if not np.array_equal(new["ffn_output"], base["ems"]):
        raise ValueError(f"{model}: FFN-source risk differs from original")
    layer_count = base["sar"].shape[1]
    if not np.array_equal(new["pre_ffn_residual"],
                          base["sar_ems_ffn_input_source"][:, layer_count:]):
        raise ValueError(f"{model}: pre-FFN-source risk differs from original")
    split = protocol.load_split()
    counts = {name: int(np.isin(base["image_id"], list(ids)).sum())
              for name, ids in split.items()}
    for seed in SEEDS:
        for mode in ("ffn_output", "pre_ffn_residual"):
            original = json.loads(result_path(model, seed, mode).read_text())
            if (original["split"] != "shared_image_level_811" or
                original["token_counts"] != counts or
                original["image_counts"] != {"train": 3200, "val": 400, "test": 400} or
                original["training"]["input_dim"] != 2 * layer_count):
                raise ValueError(f"{model}/seed{seed}: fixed-split reference differs")
        for mode in ("post_ffn_residual", "attention_output"):
            protocol.DISPLAY[mode] = DISPLAY[mode]
            folder = result_path(model, seed, mode).parent
            result_file = folder / "result.json"
            if result_file.exists():
                saved = json.loads(result_file.read_text())
                if saved["split"] != "shared_image_level_811" or saved["token_counts"] != counts:
                    raise ValueError(f"{model}/seed{seed}/{mode}: saved run differs")
                print(f"reuse {model} seed{seed} {mode}", flush=True)
                continue
            matrix = np.concatenate((base["sar"], new[mode]), axis=1).astype(np.float32)
            protocol.train_case(
                model=model, case=mode, matrix=matrix,
                labels=base["label"], image_ids=base["image_id"],
                split=split, seed=seed, device=device, output_dir=folder,
                split_protocol="shared_image_level_811",
            )
            print(f"completed {model} seed{seed} {mode}", flush=True)
    OUT.joinpath(model).mkdir(parents=True, exist_ok=True)
    protocol.atomic_json(OUT / model / "audit.json", {
        "model": model,
        "samples": len(base["label"]),
        "layers": layer_count,
        "seeds": list(SEEDS),
        "split": "same 3200/400/400 images for every seed",
        "token_counts": counts,
        "u_and_r_identical_to_original_cache": True,
        "training": "same train_case as original fixed-split 8:1:1",
    })


def summarize() -> None:
    rows = []
    lines = [
        "# COCO4000 固定 8:1:1：EMS 源分布 P 构造向量对比", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；固定同一份图片级 3200/400/400 训练/验证/测试划分，种子 43/44/45 只控制 MLP 初始化、训练样本顺序与 dropout。每种 $P$ 均在当前图像的视觉 token 上 softmax。只改变构造 $P$ 的向量 $u,r,r+u,o_A$；SAR、$T$、cost、$K=32$、$\\tau=0.03$ 和原训练器不变。$u/r$ 复用逐层完全一致的已有固定划分实验；$r+u/o_A$ 新训。", "",
        "测试集指标单位 %，± 为三种子总体标准差。", "",
        "| 模型 | P 构造向量 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | ΔAUROC vs u (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        grouped = {}
        for mode in MODES:
            values = []
            for seed in SEEDS:
                result = json.loads(result_path(model, seed, mode).read_text())
                measure = protocol.flatten(result)
                values.append(measure)
                rows.append({"model": model, "source_vector": mode, "seed": seed, **measure})
            grouped[mode] = values
            cells = " | ".join(
                f"{100*np.mean([v[key] for v in values]):.2f} ± "
                f"{100*np.std([v[key] for v in values]):.2f}"
                for key in METRICS
            )
            delta = 100 * (np.mean([v["auroc"] for v in values]) -
                           np.mean([v["auroc"] for v in grouped["ffn_output"]]))
            lines.append(f"| {model} | {DISPLAY[mode]} | {cells} | {delta:+.2f} |")
    lines += ["", "## 逐种子", "",
              "| 模型 | P 构造向量 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {DISPLAY[row['source_vector']]} | {row['seed']} | "
            + " | ".join(f"{100*row[key]:.3f}" for key in METRICS) + " |"
        )
    lines += ["", "验证阈值、逐样本分数和训练配置在各模型/向量/种子目录；`audit.json` 记录缓存和划分核对，`per_seed.csv` 保存全精度结果。", ""]
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model or --summarize required")


if __name__ == "__main__":
    main()
