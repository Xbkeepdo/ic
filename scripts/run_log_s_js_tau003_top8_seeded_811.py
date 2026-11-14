#!/usr/bin/env python3
"""Train tau=0.03 Top8 union JS on seeded COCO 8:1:1, against saved OT+R_q."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_log_s_js_tau007_top8_seeded_811_82 as previous

OUT = ROOT / "outputs/log_s_js_tau003_top8_seeded_811_43_45"
GF = ROOT.parent / "gfchair/outputs/coco4000_512_log_s_js_tau003_top8_811"
REF = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def flatten(path: Path) -> dict[str, float]:
    test = json.loads(path.read_text())["test"]
    return {
        "auroc": float(test["auroc"]),
        "real_aupr": float(test["real"]["aupr"]),
        "hall_aupr": float(test["hall"]["aupr"]),
        "real_f1": float(test["real"]["f1"]),
        "hall_f1": float(test["hall"]["f1"]),
    }


def run_model(model: str, device: str) -> None:
    import torch
    from scripts import run_log_s_mechanism_ablation_811 as protocol

    torch.set_num_threads(1)
    previous.GF = GF
    matrices, _, labels, image_ids, _, _ = previous.load_features(model)
    case = "sar_js_union8"
    protocol.DISPLAY[case] = "SAR + JS Top8 union tau=0.03"
    for seed in SEEDS:
        split = previous.load_811_split(seed)
        expected = {part: int(np.isin(image_ids, list(ids)).sum()) for part, ids in split.items()}
        ot = json.loads((REF / model / f"seed{seed}" / "sar_ems/result.json").read_text())
        if ot["split"] != "seeded_image_level_811" or ot["token_counts"] != expected:
            raise ValueError(f"{model}/seed{seed}: OT split differs")
        folder = OUT / model / f"seed{seed}" / case
        if (folder / "result.json").exists():
            prior = json.loads((folder / "result.json").read_text())
            if prior["token_counts"] != expected or prior["split"] != "seeded_image_level_811":
                raise ValueError(f"{model}/seed{seed}: existing Top8 result uses another split")
        else:
            protocol.train_case(
                model=model, case=case, matrix=matrices["union8"], labels=labels,
                image_ids=image_ids, split=split, seed=seed, device=device,
                output_dir=folder, split_protocol="seeded_image_level_811",
            )
        print(f"{model} seed{seed} tau=0.03 Top8 done", flush=True)


def summarize() -> None:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for case, folder in (
                ("ot", REF / model / f"seed{seed}" / "sar_ems"),
                ("js_top8_tau003", OUT / model / f"seed{seed}" / "sar_js_union8"),
            ):
                result = json.loads((folder / "result.json").read_text())
                if result["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
                    raise ValueError(f"{model}/seed{seed}/{case}: wrong image split")
                rows.append({"model": model, "seed": seed, "case": case, **flatten(folder / "result.json")})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# COCO 8:1:1：OT+R_q vs Top8 JS（tau=0.03）",
        "",
        "LLaVA-1.5-7B、Qwen3-VL-8B；种子 43/44/45 同时控制 3200/400/400 图片划分及 MLP 初始化。OT+R_q 使用原 K=32、tau=0.03 检测器。JS 在 Top8(P)∪Top8(T) 视觉支持集上分别重归一化后计算自然对数 JS，不加 R_q；P_tau=0.03 从缓存 P_tau=0.07 按 P_0.03 ∝ P_0.07^(0.07/0.03) 重建，T 与 SAR 不变。数值为测试均值 ± 总体标准差，单位百分数。",
        "",
        "| 模型 | 方法 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for case in ("ot", "js_top8_tau003"):
            subset = [row for row in rows if row["model"] == model and row["case"] == case]
            values = [f"{100*mean(row[metric] for row in subset):.2f} ± {100*pstdev(row[metric] for row in subset):.2f}" for metric in METRICS]
            lines.append(f"| {model} | {case} | {' | '.join(values)} |")
    lines += ["", "逐种子完整指标见 per_seed.csv；配对逐图 Bootstrap 见 bootstrap_3000。", ""]
    (OUT / "complete_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
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
        parser.error("--model is required unless --summarize is set")
