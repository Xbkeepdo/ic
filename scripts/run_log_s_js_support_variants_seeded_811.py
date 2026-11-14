#!/usr/bin/env python3
"""Train SAR plus new JS support variants on seeded COCO 8:1:1 splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_log_s_mechanism_ablation_811 as protocol  # noqa: E402
from scripts.run_log_s_ot_js_seeded_811 import load_split  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402

GF_FEATURES = ROOT.parent / "gfchair/outputs/coco4000_512_log_s_js_support_variants_811"
CACHE = ROOT / "outputs/log_s_mechanism_ablation_811"
PRIOR = ROOT / "outputs/log_s_ot_js_seeded_811_43_45"
REFERENCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
OUT = ROOT / "outputs/log_s_js_support_variants_seeded_811_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
NEW = ("union32", "rank16")
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def key_of(row: dict) -> tuple[int, int, int]:
    return (int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"]))


def load_matrices(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(CACHE / model / "compact_features.npz", allow_pickle=False) as data:
        sar, ems = np.asarray(data["sar"]), np.asarray(data["ems"])
        n = sar.shape[1]
        rq = ems - np.asarray(data["sar_ot_geometry"][:, n:])
        original_js = np.asarray(data["sar_union_js"][:, n:])
        labels, image_ids = np.asarray(data["label"]), np.asarray(data["image_id"])
        keys = list(zip(image_ids.tolist(), data["response_token_idx"].tolist(), data["target_token_id"].tolist()))
    rows = load_pkl(GF_FEATURES / model / "features.pkl")
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
    anchor_error = np.abs(extracted["js_union16"].astype(np.float64) - original_js)
    delta = float(anchor_error.max())
    outside_tolerance = ~np.isclose(extracted["js_union16"], original_js, rtol=1e-5, atol=1e-6)
    mismatch_count = int(outside_tolerance.sum())
    mismatch_fraction = mismatch_count / int(original_js.size)
    # Reconstructing tau=.03 from cached float32 tau=.07 probabilities can
    # change a top-k boundary in a few cells; record this, never claim exact JS.
    if mismatch_fraction >= 1e-4 or delta >= 1e-2:
        raise ValueError(
            f"{model}: cached-P reconstruction differs from old JS too often: "
            f"{mismatch_count}/{original_js.size}, max={delta}"
        )
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
        "legacy_union16_js_max_abs_delta": delta,
        "legacy_union16_js_cells_outside_tolerance": mismatch_count,
        "legacy_union16_js_fraction_outside_tolerance": mismatch_fraction,
        "legacy_union16_tolerance": {"rtol": 1e-5, "atol": 1e-6},
    }, indent=2) + "\n")
    return matrices, labels, image_ids


def run_model(model: str, device: str) -> None:
    import torch

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
    if case in ("union16", "union16_rq"):
        old = "sar_union_js" if case == "union16" else "sar_js_rq"
        return PRIOR / model / f"seed{seed}" / old / "result.json"
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
                flat = protocol.flatten(result)
                rows.append({"model": model, "case": case, "seed": seed, **{k: flat[k] for k in METRICS}})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# JS 支持集对比：Top16 并集、Top32 并集、Top16 排名", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；沿用种子 43/44/45 联动的 3200/400/400 图片划分与 MLP 初始化。OT 始终为 SAR+OT_G+R_q；新 JS 各试原值与 +R_q。Top16 rank 将 P、T 各自前 16 个概率降序排列、各自归一化后按排名配对，忽略视觉 token 位置身份。所有方案共用 SAR、object token、宽度 128 单隐藏层 MLP 和验证集选阈值规则。", "",
        "新 JS 从已保存的逐层 P/T 矩阵计算。原缓存 P 的温度为 0.07，通过 P_0.03 ∝ P_0.07^(0.07/0.03) 重建；T 直接复用。与重新前向计算的旧 Top16 并集 JS 做全量核对，极少数 top-k 边界因 float32 缓存精度产生差异；精确数量与最大差异见每模型 alignment.json。旧 Top16 并集对照使用原始缓存实验结果。+R_q 实验复用原 K=32 OT 的 R_q，不随 JS 支持集改变。", "",
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
    lines += ["", "每模型的 `alignment.json` 记录旧 JS 逐值核对；完整逐种子数据见 `per_seed.csv`。", ""]
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
