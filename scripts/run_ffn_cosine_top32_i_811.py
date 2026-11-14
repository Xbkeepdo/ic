#!/usr/bin/env python3
"""Evaluate FFN cosine on the 32 visual tokens with largest write norm."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

from scripts import run_log_s_mechanism_ablation_811 as protocol
from scripts.run_ffn_cosine_is_811 import NAMES as PREVIOUS_NAMES
from scripts.run_ffn_cosine_is_811 import OUT as PREVIOUS
from scripts.run_log_s_source_correspondence_ablation_811 import METRICS, REFERENCE, SEEDS


ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
SOURCE = GF_ROOT / "outputs/coco4000_512_linear_cost_log_s_risk_811"
OUT = ROOT / "outputs/ffn_cosine_top32_i_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
TOP_K = 32
TAU = 0.07
CASES = ("i_s", "i_s_ffn_cos_top32_i")
NAMES = {
    "i_s": "I + S",
    "i_s_ffn_cos_top32_i": "I + S + F_top32(I)",
    **PREVIOUS_NAMES,
}


def key(row: dict) -> tuple[int, int, int]:
    return (int(row["image_id"]), int(row["response_token_idx"]),
            int(row["target_token_id"]))


def iter_source_rows(model: str):
    folder = SOURCE / model
    parts = sorted(folder.glob("features.part*.pkl"))
    paths = parts if parts else [folder / "features.pkl"]
    for path in paths:
        with path.open("rb") as handle:
            while True:
                try:
                    block = pickle.load(handle)
                except EOFError:
                    break
                for row in block:
                    yield row


def prepare(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    output = OUT / model
    cache = output / "compact_features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return ({case: np.asarray(saved[case]) for case in CASES},
                    np.asarray(saved["label"]), np.asarray(saved["image_id"]))

    with np.load(PREVIOUS / model / "compact_features.npz", allow_pickle=False) as saved:
        labels = np.asarray(saved["label"], dtype=np.int8)
        images = np.asarray(saved["image_id"], dtype=np.int64)
        response = np.asarray(saved["response_token_idx"], dtype=np.int32)
        tokens = np.asarray(saved["target_token_id"], dtype=np.int32)
        mean_matrix = np.asarray(saved["i_s_ffn_cos_mean"], dtype=np.float32)
        max_matrix = np.asarray(saved["i_s_ffn_cos_max"], dtype=np.float32)
    layers = mean_matrix.shape[1] // 3
    if mean_matrix.shape != max_matrix.shape or mean_matrix.shape != (len(labels), 3*layers):
        raise ValueError(f"{model}: previous compact feature dimensions differ")
    keys = list(zip(images.astype(int), response.astype(int), tokens.astype(int)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate target keys in previous features")
    by_key = {target_key: index for index, target_key in enumerate(keys)}
    fmean = mean_matrix[:, 2*layers:]
    fmax = max_matrix[:, 2*layers:]
    top32 = np.full((len(keys), layers), np.nan, dtype=np.float32)
    seen = np.zeros(len(keys), dtype=bool)
    processed = 0
    max_error = 0.0
    min_visual_count = None
    max_visual_count = 0
    for row in iter_source_rows(model):
        row_key = key(row)
        index = by_key.get(row_key)
        if index is None:
            raise ValueError(f"{model}: source contains an unmatched target: {row_key}")
        if seen[index]:
            continue
        if int(row["label"]) != int(labels[index]):
            raise ValueError(f"{model}/{row_key}: label differs")
        p = np.asarray(row["matrices"]["source_hpre"], dtype=np.float64)
        write = np.asarray(row["matrices"]["am_norm"], dtype=np.float64)
        if p.shape != write.shape or p.shape[0] != layers or p.shape[1] < TOP_K:
            raise ValueError(f"{model}/{row_key}: invalid per-source dimensions")
        if not (np.isfinite(p).all() and np.isfinite(write).all() and
                (p > 0).all() and (write >= 0).all()):
            raise ValueError(f"{model}/{row_key}: nonfinite or invalid source values")
        logp = np.log(p)
        cosine = fmean[index, :, None].astype(np.float64) + TAU * (
            logp - logp.mean(axis=1, keepdims=True)
        )
        error = float(np.max(np.abs(cosine.max(axis=1) - fmax[index])))
        max_error = max(max_error, error)
        if error > 2e-4:
            raise ValueError(f"{model}/{row_key}: recovered maximum differs by {error:.6g}")
        selected = np.argpartition(-write, TOP_K-1, axis=1)[:, :TOP_K]
        top32[index] = np.take_along_axis(cosine, selected, axis=1).mean(axis=1)
        seen[index] = True
        processed += 1
        visual_count = p.shape[1]
        min_visual_count = visual_count if min_visual_count is None else min(min_visual_count, visual_count)
        max_visual_count = max(max_visual_count, visual_count)
        if processed % 2000 == 0:
            print(f"[{model}] computed {processed}/{len(keys)} targets", flush=True)
    if not seen.all():
        raise ValueError(f"{model}: {int((~seen).sum())} targets missing from source")
    if not np.isfinite(top32).all() or np.any(top32 > fmax + 1e-5):
        raise ValueError(f"{model}: invalid top32 values")
    matrices = {
        "i_s": mean_matrix[:, :2*layers],
        "i_s_ffn_cos_top32_i": np.concatenate((mean_matrix[:, :2*layers], top32), axis=1),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **matrices, ffn_cosine_top32_i=top32,
                        label=labels, image_id=images,
                        response_token_idx=response, target_token_id=tokens)
    protocol.atomic_json(output / "alignment.json", {
        "model": model, "samples": len(keys), "layers": layers,
        "min_visual_tokens": min_visual_count, "max_visual_tokens": max_visual_count,
        "top_k": TOP_K, "source_tau": TAU,
        "definition": "mean cos(o_ffn(object),hpre(visual_m)) over 32 highest ||a_lm|| visual tokens per layer",
        "reconstruction": "F_mean + tau * (log(source_hpre) - mean_visual(log(source_hpre)))",
        "max_reconstruction_error": max_error,
        "alignment_key": ["image_id", "response_token_idx", "target_token_id"],
        "source": str(SOURCE / model),
        "previous_features": str(PREVIOUS / model / "compact_features.npz"),
    })
    print(f"[{model}] prepared {len(keys)} targets; maximum reconstruction error={max_error:.3g}", flush=True)
    return matrices, labels, images


def run_model(model: str, device: str, *, prepare_only: bool = False) -> None:
    import torch

    torch.set_num_threads(1)
    protocol.DISPLAY.update({case: NAMES[case] for case in CASES})
    matrices, labels, images = prepare(model)
    if prepare_only:
        return
    for seed in SEEDS:
        split_list = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        split = {name: set(map(int, split_list[name])) for name in ("train", "val", "test")}
        token_counts = {name: int(np.isin(images, list(ids)).sum()) for name, ids in split.items()}
        previous = json.loads((PREVIOUS / model / f"seed{seed}" / "i_s_ffn_cos_mean" / "result.json").read_text())
        if previous["token_counts"] != token_counts:
            raise ValueError(f"{model}/seed{seed}: split cohort differs from previous FFN run")
        for case in CASES:
            folder = OUT / model / f"seed{seed}" / case
            if (folder / "result.json").exists():
                saved = json.loads((folder / "result.json").read_text())
                if saved["token_counts"] != token_counts:
                    raise ValueError(f"{model}/seed{seed}/{case}: saved cohort differs")
                continue
            protocol.train_case(model=model, case=case, matrix=matrices[case],
                                labels=labels, image_ids=images, split=split,
                                seed=seed, device=device, output_dir=folder,
                                split_protocol="seeded_image_level_811")
            print(f"[{model}] completed seed{seed} {case}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO 8:1:1：高写入视觉区域的 FFN 对齐特征", "",
        "在 LLaVA-1.5-7B 与 Qwen3-VL-8B 上，逐层选择 ||a_lm|| 最大的 32 个视觉 token，计算物体 token 的 FFN 输出与其 hpre 的原始 cosine 均值，记为 F_top32(I)。与 I、S 逐层拼接；I=∑ₘ||a_lm||，S=∑ₘP(yₜ|final_norm(hpre_m))。", "",
        "相同 ENDAC object token、三种子 43/44/45 同时控制 3200/400/400 图片划分及 MLP 初始化；单隐藏层宽度 128、dropout 0.3、batch 256、学习率 0.001、weight decay 1e-5、训练集标准化、最小验证损失选 checkpoint、验证集分别选 Real/Hall F1 阈值。指标为测试集百分数；± 为三种子总体标准差。", "",
        "| 模型 | 特征 | AUROC | Real AUPR | Hall AUPR | Real F1 | Real Recall | Hall F1 | Hall Recall | Accuracy (Real threshold) | Accuracy (Hall threshold) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        by_case = {}
        for case in ("i_s", "i_s_ffn_cos_top32_i", "i_s_ffn_cos_mean", "i_s_ffn_cos_max", "sar_ems"):
            values = []
            for seed in SEEDS:
                if case == "sar_ems":
                    path = REFERENCE / model / f"seed{seed}" / case / "result.json"
                elif case in ("i_s_ffn_cos_mean", "i_s_ffn_cos_max"):
                    path = PREVIOUS / model / f"seed{seed}" / case / "result.json"
                else:
                    path = OUT / model / f"seed{seed}" / case / "result.json"
                flat = protocol.flatten(json.loads(path.read_text()))
                values.append(flat)
                rows.append({"model": model, "case": case, "seed": seed, **flat})
            by_case[case] = values
            formatted = " | ".join(
                f"{100*np.mean([value[metric] for value in values]):.2f} ± {100*np.std([value[metric] for value in values]):.2f}"
                for metric in METRICS
            )
            lines.append(f"| {model} | {NAMES[case]} | {formatted} |")
        delta = [f"{100*np.mean([a[metric]-b[metric] for a,b in zip(by_case['i_s_ffn_cos_top32_i'],by_case['i_s_ffn_cos_mean'])]):+.2f}"
                 for metric in METRICS]
        lines.append(f"| {model} | Top32(I) − all-visual mean (paired) | " + " | ".join(delta) + " |")
    lines += ["", "## 逐种子测试指标（%）", "",
              "| 模型 | 特征 | 种子 | " + " | ".join(METRICS) + " |",
              "|---|---|---:|" + "---:|"*len(METRICS)]
    for row in rows:
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | " +
                     " | ".join(f"{100*row[metric]:.3f}" for metric in METRICS) + " |")
    lines += ["", "特征计算及恢复误差见每个模型的 `alignment.json`；逐种子检测器原始指标见各条件目录。", ""]
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device, prepare_only=args.prepare_only)
    else:
        parser.error("--model required unless --summarize")


if __name__ == "__main__":
    main()
