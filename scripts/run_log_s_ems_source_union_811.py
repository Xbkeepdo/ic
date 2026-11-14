#!/usr/bin/env python3
"""Test whether FFN-source EMS adds information beyond attention-source EMS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts import run_log_s_p_source_vector_ablation_811 as source_ablation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_ems_source_union_811"
REF_F = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
REF_A = ROOT / "outputs/log_s_p_source_vector_ablation_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
CASES = (
    "sar_ea", "sar_ef", "sar_ea_ef", "sar_ea_epre", "sar_ea_ea",
)
DISPLAY = {
    "sar_ea": "SAR + E_A",
    "sar_ef": "SAR + E_F",
    "sar_ea_ef": "SAR + E_A + E_F",
    "sar_ea_epre": "SAR + E_A + E_pre",
    "sar_ea_ea": "SAR + E_A + E_A (duplicate control)",
}
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def reference_path(model: str, seed: int, case: str) -> Path:
    if case == "sar_ea":
        return REF_A / model / f"seed{seed}" / "attention_output" / "result.json"
    if case == "sar_ef":
        return REF_F / model / f"seed{seed}" / "sar_ems" / "result.json"
    return OUT / model / f"seed{seed}" / case / "result.json"


def train_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    base = source_ablation.load_base(model)
    new = source_ablation.load_new(model, base)
    sar = base["sar"]
    ea, ef, epre = (new["attention_output"], new["ffn_output"],
                    new["pre_ffn_residual"])
    if not (sar.shape == ea.shape == ef.shape == epre.shape):
        raise ValueError(f"{model}: layer feature shape mismatch")
    matrices = {
        "sar_ea_ef": np.concatenate((sar, ea, ef), axis=1),
        "sar_ea_epre": np.concatenate((sar, ea, epre), axis=1),
        "sar_ea_ea": np.concatenate((sar, ea, ea), axis=1),
    }
    for seed in SEEDS:
        split_list = json.loads((REF_F / "splits" / f"seed{seed}.json").read_text())
        split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
        ref_a = json.loads(reference_path(model, seed, "sar_ea").read_text())
        ref_f = json.loads(reference_path(model, seed, "sar_ef").read_text())
        counts = {name: int(np.isin(base["image_id"], list(ids)).sum()) for name, ids in split.items()}
        for reference in (ref_a, ref_f):
            if (reference["split"] != "seeded_image_level_811" or
                reference["image_counts"] != {"train": 3200, "val": 400, "test": 400} or
                reference["token_counts"] != counts or
                reference["training"]["hidden_sizes"] != [128]):
                raise ValueError(f"{model}/seed{seed}: reference protocol differs")
        for case, matrix in matrices.items():
            protocol.DISPLAY[case] = DISPLAY[case]
            folder = OUT / model / f"seed{seed}" / case
            result_path = folder / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if (saved["token_counts"] != counts or saved["case"] != case or
                    saved["training"]["input_dim"] != matrix.shape[1]):
                    raise ValueError(f"{model}/seed{seed}/{case}: saved run differs")
                print(f"reuse {model} seed{seed} {case}", flush=True)
                continue
            protocol.train_case(
                model=model, case=case, matrix=matrix.astype(np.float32),
                labels=base["label"], image_ids=base["image_id"],
                split=split, seed=seed, device=device, output_dir=folder,
                split_protocol="seeded_image_level_811",
            )
            print(f"completed {model} seed{seed} {case}", flush=True)


def summarize() -> None:
    rows = []
    report = [
        "# COCO4000 8:1:1：Attention 与 FFN 来源的 EMS 联合实验", "",
        "固定 SAR、$T$、cost、$K=32$、$\\tau=0.03$、同一 object token、8:1:1 图片划分及原训练器。种子 43/44/45 同时控制划分和 MLP 初始化。$E_A$、$E_F$、$E_{pre}$ 分别是以 attention 输出、FFN 输出、FFN 前残差构造 $P$ 得到的逐层 EMS 风险。$E_A$ 重复输入是等维度、等参数量但无新增特征信息的容量对照。", "",
        "指标单位为 %；± 为三种子总体标准差。输入维度 2L 或 3L；固定隐藏层宽度 128、单隐藏层、无 BatchNorm，参数量分别为 $128(2L)+257$ 和 $128(3L)+257$。", "",
        "| 模型 | 输入 | 维度 | 参数量 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | ΔAUROC vs SAR+E_A (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        by_case = {}
        for case in CASES:
            values = []
            for seed in SEEDS:
                result = json.loads(reference_path(model, seed, case).read_text())
                metric = protocol.flatten(result)
                values.append(metric)
                rows.append({"model": model, "case": case, "seed": seed,
                             "input_dim": result["training"]["input_dim"], **metric})
            by_case[case] = values
            input_dim = int(rows[-1]["input_dim"])
            parameters = input_dim * 128 + 257
            cells = " | ".join(
                f"{100 * np.mean([v[field] for v in values]):.2f} ± "
                f"{100 * np.std([v[field] for v in values]):.2f}"
                for field in METRICS
            )
            baseline = by_case["sar_ea"]
            delta = 100 * (np.mean([v["auroc"] for v in values]) -
                           np.mean([v["auroc"] for v in baseline]))
            report.append(f"| {model} | {DISPLAY[case]} | {input_dim} | {parameters} | {cells} | {delta:+.2f} |")
    report += ["", "## 逐种子结果", "",
               "| 模型 | 输入 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(
            f"| {row['model']} | {DISPLAY[row['case']]} | {row['seed']} | "
            + " | ".join(f"{100 * row[field]:.3f}" for field in METRICS) + " |"
        )
    report += ["", "训练配置、验证阈值、逐样本分数在每个种子的对应目录。完整逐种子指标见 `per_seed.csv`。", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(report))
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
        train_model(args.model, args.device)
    else:
        parser.error("--model or --summarize required")


if __name__ == "__main__":
    main()
