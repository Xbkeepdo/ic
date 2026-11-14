#!/usr/bin/env python3
"""Train SAR plus six JS variants with P_tau=0.07 on seeded COCO 8:1:1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GF_FEATURES = ROOT.parent / "gfchair/outputs/coco4000_512_log_s_js_tau007_variants_811"
CACHE = ROOT / "outputs/log_s_mechanism_ablation_811"
REFERENCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
OUT = ROOT / "outputs/log_s_js_tau007_variants_seeded_811_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
NEW = ("union16", "union32", "rank16")
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def key_of(row: dict) -> tuple[int, int, int]:
    return (int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"]))


def load_split(seed: int) -> dict[str, set[int]]:
    raw = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
    if {name: len(ids) for name, ids in raw.items()} != {"train": 3200, "val": 400, "test": 400}:
        raise ValueError(f"seed{seed}: invalid image split size")
    split = {name: set(ids) for name, ids in raw.items()}
    if len(set.union(*split.values())) != 4000:
        raise ValueError(f"seed{seed}: image split overlaps")
    return split


def load_matrices(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(CACHE / model / "compact_features.npz", allow_pickle=False) as data:
        sar, ems = np.asarray(data["sar"]), np.asarray(data["ems"])
        n = sar.shape[1]
        rq = ems - np.asarray(data["sar_ot_geometry"][:, n:])
        labels, image_ids = np.asarray(data["label"]), np.asarray(data["image_id"])
        keys = list(zip(image_ids.tolist(), data["response_token_idx"].tolist(), data["target_token_id"].tolist()))
    with (GF_FEATURES / model / "features.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    by_key = {key_of(row): row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != set(keys):
        raise ValueError(f"{model}: new and cached object-token keys differ")
    aligned = [by_key[key] for key in keys]
    if any(int(row["label"]) != int(label) for row, label in zip(aligned, labels)):
        raise ValueError(f"{model}: label mismatch")
    extracted = {
        name: np.stack([np.asarray(row["features"][name], dtype=np.float32) for row in aligned])
        for name in ("js_union16", "js_union32", "js_rank16")
    }
    matrices = {}
    for name in NEW:
        js = extracted[f"js_{name}"]
        if js.shape != sar.shape or not np.isfinite(js).all():
            raise ValueError(f"{model}: invalid {name} JS feature")
        matrices[f"sar_js_{name}"] = np.concatenate((sar, js), axis=1)
        matrices[f"sar_js_{name}_rq"] = np.concatenate((sar, js + rq), axis=1)
    (OUT / model).mkdir(parents=True, exist_ok=True)
    (OUT / model / "alignment.json").write_text(json.dumps({
        "model": model, "samples": len(labels), "layers": n,
        "exact_key_and_label_alignment": True,
        "js_source_tau": 0.07,
        "ot_and_rq_source_tau": 0.03,
    }, indent=2) + "\n")
    return matrices, labels, image_ids


def run_model(model: str, device: str) -> None:
    import torch
    from scripts import run_log_s_mechanism_ablation_811 as protocol

    torch.set_num_threads(1)
    matrices, labels, image_ids = load_matrices(model)
    protocol.DISPLAY.update({name: name for name in matrices})
    print(f"[{model}] aligned {len(labels)} object tokens", flush=True)
    for seed in SEEDS:
        split = load_split(seed)
        ot = json.loads((REFERENCE / model / f"seed{seed}" / "sar_ems/result.json").read_text())
        counts = {part: int(np.isin(image_ids, list(ids)).sum()) for part, ids in split.items()}
        if ot["token_counts"] != counts:
            raise ValueError(f"{model}/seed{seed}: OT split differs")
        for case, matrix in matrices.items():
            folder = OUT / model / f"seed{seed}" / case
            path = folder / "result.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["token_counts"] != counts:
                    raise ValueError(f"{model}/seed{seed}/{case}: existing split differs")
                print(f"[{model}] reuse {case} seed{seed}", flush=True)
                continue
            protocol.train_case(
                model=model, case=case, matrix=matrix, labels=labels, image_ids=image_ids,
                split=split, seed=seed, device=device, output_dir=folder,
                split_protocol="seeded_image_level_811",
            )
            print(f"[{model}] completed {case} seed{seed}", flush=True)


def result_path(model: str, seed: int, case: str) -> Path:
    if case == "ot":
        return REFERENCE / model / f"seed{seed}" / "sar_ems/result.json"
    return OUT / model / f"seed{seed}" / f"sar_js_{case}" / "result.json"


def summarize() -> None:
    cases = ("ot", "union16", "union16_rq", "union32", "union32_rq", "rank16", "rank16_rq")
    display = {
        "ot": "SAR+OT_G+R_q", "union16": "SAR+JS Top16 union",
        "union16_rq": "SAR+JS Top16 union+R_q",
        "union32": "SAR+JS Top32 union", "union32_rq": "SAR+JS Top32 union+R_q",
        "rank16": "SAR+JS Top16 rank", "rank16_rq": "SAR+JS Top16 rank+R_q",
    }
    rows = []
    for model in MODELS:
        for case in cases:
            for seed in SEEDS:
                result = json.loads(result_path(model, seed, case).read_text())
                test = result["test"]
                flat = {
                    "auroc": float(test["auroc"]),
                    "real_aupr": float(test["real"]["aupr"]),
                    "hall_aupr": float(test["hall"]["aupr"]),
                    "real_f1": float(test["real"]["f1"]),
                    "hall_f1": float(test["hall"]["f1"]),
                }
                rows.append({"model": model, "case": case, "seed": seed, **{k: flat[k] for k in METRICS}})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# COCO 8:1:1：源分布 tau=0.07 的六种 JS 构造", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；种子 43/44/45 联动控制 3200/400/400 图片划分与 MLP 初始化。OT 固定为原 tau=0.03 的 SAR+OT_G+R_q；六种 JS 用缓存的 P_tau=0.07 和相同 T，分别训练原值及 +R_q。R_q 固定为原 K=32、tau=0.03 的项，隔离 JS 源分布温度的影响。Top16 rank 将 P、T 各自前 16 个概率降序排列后按排名配对。所有方案共用 SAR、object token、宽度 128 单隐藏层 MLP 和验证集选阈值规则。", "",
        "三种子测试集均值 ± 总体标准差，单位为百分数。", "",
        "| 模型 | 方法 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for case in cases:
            group = [row for row in rows if row["model"] == model and row["case"] == case]
            metrics = [f"{100*np.mean([row[key] for row in group]):.2f} ± {100*np.std([row[key] for row in group]):.2f}" for key in METRICS]
            lines.append(f"| {model} | {display[case]} | {' | '.join(metrics)} |")
    lines += ["", "## 逐种子测试结果", "",
              "| 模型 | 方法 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {display[row['case']]} | {row['seed']} | {' | '.join(f'{100*row[key]:.2f}' for key in METRICS)} |")
    lines += ["", "每模型的 `alignment.json` 记录 object-token 键及标签对齐；完整逐种子数据见 `per_seed.csv`。", ""]
    (OUT / "complete_results.md").write_text("\n".join(lines))
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
        parser.error("--model is required unless --summarize is set")


if __name__ == "__main__":
    main()
