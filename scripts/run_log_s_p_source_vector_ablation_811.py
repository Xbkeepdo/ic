#!/usr/bin/env python3
"""Train SAR+EMS for four source-P vectors on seeded COCO ENDAC 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from utils.io_utils import load_pkl


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
EXTRACTED = GF_ROOT / "outputs/coco4000_512_log_s_source_vectors_811"
BASE = ROOT / "outputs/log_s_mechanism_ablation_811"
REFERENCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
OUT = ROOT / "outputs/log_s_p_source_vector_ablation_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
MODES = ("ffn_output", "pre_ffn_residual", "post_ffn_residual", "attention_output")
DISPLAY = {
    "ffn_output": "FFN update u (original)",
    "pre_ffn_residual": "Pre-FFN residual r",
    "post_ffn_residual": "Post-FFN residual r+u",
    "attention_output": "Attention output o_A",
}
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def key(row: dict) -> tuple[int, int, int]:
    return int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"])


def load_base(model: str) -> dict[str, np.ndarray]:
    with np.load(BASE / model / "compact_features.npz", allow_pickle=False) as saved:
        base = {field: np.asarray(saved[field]) for field in (
            "sar", "ems", "sar_ems", "sar_ems_ffn_input_source",
            "label", "image_id", "response_token_idx", "target_token_id",
        )}
    layers = base["sar"].shape[1]
    if not (np.array_equal(base["sar_ems"], np.concatenate((base["sar"], base["ems"]), axis=1))
            and base["sar_ems_ffn_input_source"].shape == (len(base["label"]), 2 * layers)):
        raise ValueError(f"{model}: original SAR/EMS cache shape or order differs")
    return base


def load_new(model: str, base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cache = OUT / model / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            result = {field: np.asarray(saved[field]) for field in saved.files}
        if not np.array_equal(result["image_id"], base["image_id"]):
            raise ValueError(f"{model}: compact cache image order differs")
        return result
    if not (EXTRACTED / model / "summary.json").exists():
        raise FileNotFoundError(f"{model}: source-vector extraction incomplete")
    rows = load_pkl(str(EXTRACTED / model / "features.pkl"))
    by_key = {key(row): row for row in rows}
    expected_keys = list(zip(
        base["image_id"].astype(int), base["response_token_idx"].astype(int),
        base["target_token_id"].astype(int),
    ))
    if len(by_key) != len(rows) or set(by_key) != set(expected_keys):
        raise ValueError(f"{model}: new source-vector targets differ from base")
    ordered = [by_key[target_key] for target_key in expected_keys]
    if not np.array_equal([int(row["label"]) for row in ordered], base["label"]):
        raise ValueError(f"{model}: label alignment differs")
    risks = {
        mode: np.stack([np.asarray(row["features"][mode], dtype=np.float32) for row in ordered])
        for mode in MODES
    }
    layers = base["sar"].shape[1]
    if any(value.shape != base["ems"].shape or not np.isfinite(value).all() for value in risks.values()):
        raise ValueError(f"{model}: source-vector risk shape/nonfinite values")
    old_pre = base["sar_ems_ffn_input_source"][:, layers:]
    errors = {
        "ffn_output": float(np.max(np.abs(risks["ffn_output"] - base["ems"]))),
        "pre_ffn_residual": float(np.max(np.abs(risks["pre_ffn_residual"] - old_pre))),
    }
    if any(value > 1e-5 for value in errors.values()):
        raise ValueError(f"{model}: unchanged u/r risk differs from prior extraction: {errors}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    protocol.atomic_npz(cache, image_id=base["image_id"], label=base["label"], **risks)
    protocol.atomic_json(cache.parent / "alignment.json", {
        "model": model, "samples": len(rows), "layers": layers,
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "original_risk_max_abs_error": errors,
        "same_labels_and_targets": True,
        "source": str(EXTRACTED / model / "features.pkl"),
    })
    return {"image_id": base["image_id"], "label": base["label"], **risks}


def run_model(model: str, selected_modes: tuple[str, ...], device: str) -> None:
    import torch

    torch.set_num_threads(1)
    base = load_base(model)
    new = None
    if any(mode in ("post_ffn_residual", "attention_output") for mode in selected_modes):
        new = load_new(model, base)
    for mode in selected_modes:
        if mode == "ffn_output":
            continue  # Exact reference run already exists.
        if mode == "pre_ffn_residual":
            risk = base["sar_ems_ffn_input_source"][:, base["sar"].shape[1]:]
        else:
            if new is None:
                raise AssertionError("new extraction was not loaded")
            risk = new[mode]
        matrix = np.concatenate((base["sar"], risk), axis=1).astype(np.float32)
        protocol.DISPLAY[mode] = DISPLAY[mode]
        for seed in SEEDS:
            split_list = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
            split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
            folder = OUT / model / f"seed{seed}" / mode
            result_path = folder / "result.json"
            reference = json.loads((REFERENCE / model / f"seed{seed}/sar_ems/result.json").read_text())
            counts = {name: int(np.isin(base["image_id"], list(ids)).sum()) for name, ids in split.items()}
            if reference["image_counts"] != {"train": 3200, "val": 400, "test": 400} or reference["token_counts"] != counts:
                raise ValueError(f"{model}/seed{seed}: reference split or token counts differ")
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved["token_counts"] != counts or saved["case"] != mode:
                    raise ValueError(f"{model}/seed{seed}/{mode}: saved run differs")
                print(f"[{model}] reuse seed{seed} {mode}", flush=True)
                continue
            protocol.train_case(
                model=model, case=mode, matrix=matrix,
                labels=base["label"], image_ids=base["image_id"],
                split=split, seed=seed, device=device, output_dir=folder,
                split_protocol="seeded_image_level_811",
            )
            print(f"[{model}] completed seed{seed} {mode}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO4000 8:1:1：EMS 源分布 P 的构造向量对比", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；种子 43/44/45 同时控制图片划分与 MLP 初始化。每种方法用同一图片内视觉 token 的 hpre 状态构造 $P_m=\\operatorname{softmax}_m(\\cos(x,v_m)/0.03)$。四个 $x$ 分别为 FFN 输出 $u$、FFN 前残差 $r=q+o_A$、FFN 后残差 $r+u$、attention 输出 $o_A$。只替换 $P$；SAR、语义门控 $s=\\log(1+Cp)$、$T=\\operatorname{norm}(A s)$、代价、Top16(P)∪Top16(T)、K=32、训练器、样本和划分不变。指标单位为百分数，± 为三种子总体标准差。", "",
        "| 模型 | P 构造向量 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | ΔAUROC vs u (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        scores = {}
        for mode in MODES:
            values = []
            for seed in SEEDS:
                path = (REFERENCE / model / f"seed{seed}/sar_ems/result.json" if mode == "ffn_output"
                        else OUT / model / f"seed{seed}/{mode}/result.json")
                result = json.loads(path.read_text())
                measures = protocol.flatten(result)
                values.append(measures)
                rows.append({"model": model, "source_vector": mode, "seed": seed, **measures})
            scores[mode] = values
            display = " | ".join(
                f"{100 * np.mean([v[field] for v in values]):.2f} ± {100 * np.std([v[field] for v in values]):.2f}"
                for field in METRICS
            )
            delta = 100 * np.mean([v["auroc"] for v in values]) - 100 * np.mean([
                v["auroc"] for v in scores["ffn_output"]
            ])
            lines.append(f"| {model} | {DISPLAY[mode]} | {display} | {delta:+.2f} |")
    lines += ["", "## 逐种子结果", "",
              "| 模型 | P 构造向量 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {DISPLAY[row['source_vector']]} | {row['seed']} | "
            + " | ".join(f"{100 * row[field]:.3f}" for field in METRICS) + " |"
        )
    lines += ["", "原始训练配置、阈值与分数见模型目录；数值锚点核对见 `alignment.json`；全精度逐种子指标见 `per_seed.csv`。", ""]
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
    parser.add_argument("--case", choices=MODES, action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        selected = tuple(args.case) if args.case else MODES
        run_model(args.model, selected, args.device)
    else:
        parser.error("--model required unless --summarize")


if __name__ == "__main__":
    main()
