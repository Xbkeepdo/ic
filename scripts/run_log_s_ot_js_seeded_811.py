#!/usr/bin/env python3
"""Compare cached SAR+OT and SAR+JS on the seeded COCO 8:1:1 split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_log_s_mechanism_ablation_811 as protocol


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs/log_s_mechanism_ablation_811"
REFERENCE = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
OUT = ROOT / "outputs/log_s_ot_js_seeded_811_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def paths(model: str, seed: int) -> tuple[Path, Path, Path]:
    ot = REFERENCE / model / f"seed{seed}" / "sar_ems" / "result.json"
    js = OUT / model / f"seed{seed}" / "sar_union_js" / "result.json"
    js_rq = OUT / model / f"seed{seed}" / "sar_js_rq" / "result.json"
    return ot, js, js_rq


def load_matrix(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(FEATURES / model / "compact_features.npz", allow_pickle=False) as data:
        sar, ems = data["sar"], data["ems"]
        ot, js = data["sar_ems"], data["sar_union_js"]
        geometry = data["sar_ot_geometry"]
        labels, image_ids = data["label"], data["image_id"]
        keys = list(zip(
            image_ids.tolist(), data["response_token_idx"].tolist(),
            data["target_token_id"].tolist(),
        ))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate object-token keys")
    if ot.shape != js.shape or not np.array_equal(ot[:, :sar.shape[1]], sar):
        raise ValueError(f"{model}: OT/SAR feature layout differs")
    if not np.array_equal(ot[:, sar.shape[1]:], ems):
        raise ValueError(f"{model}: OT feature differs from cache")
    if not np.array_equal(js[:, :sar.shape[1]], sar):
        raise ValueError(f"{model}: JS and OT use different SAR features")
    if geometry.shape != ot.shape or not np.array_equal(geometry[:, :sar.shape[1]], sar):
        raise ValueError(f"{model}: geometry OT feature layout differs")
    # EMS is OT_G + R_q; geometry contains OT_G on the same top-K union.
    rq = ems - geometry[:, sar.shape[1]:]
    js_rq = js.copy()
    js_rq[:, sar.shape[1]:] += rq
    if any(not np.isfinite(value).all() for value in (js, js_rq, geometry)):
        raise ValueError(f"{model}: nonfinite comparison features")
    return {"sar_union_js": js, "sar_js_rq": js_rq}, labels, image_ids


def load_split(seed: int) -> dict[str, set[int]]:
    raw = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
    if {name: len(ids) for name, ids in raw.items()} != {"train": 3200, "val": 400, "test": 400}:
        raise ValueError(f"seed{seed}: invalid image split size")
    split = {name: set(ids) for name, ids in raw.items()}
    if len(set.union(*split.values())) != 4000:
        raise ValueError(f"seed{seed}: image split overlaps")
    return split


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    matrices, labels, image_ids = load_matrix(model)
    protocol.DISPLAY["sar_js_rq"] = "SAR + JS + R_q"
    print(f"[{model}] {len(labels)} object tokens", flush=True)
    for seed in SEEDS:
        split = load_split(seed)
        ot_path, js_path, js_rq_path = paths(model, seed)
        ot = json.loads(ot_path.read_text())
        expected_counts = {
            name: int(np.isin(image_ids, list(ids)).sum())
            for name, ids in split.items()
        }
        if ot["split"] != "seeded_image_level_811" or ot["token_counts"] != expected_counts:
            raise ValueError(f"{model}/seed{seed}: OT result and JS feature split differ")
        for case, path in (("sar_union_js", js_path), ("sar_js_rq", js_rq_path)):
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["token_counts"] != expected_counts or saved["split"] != "seeded_image_level_811":
                    raise ValueError(f"{model}/seed{seed}: existing {case} result uses another split")
                print(f"[{model}] reuse {case} seed{seed}", flush=True)
                continue
            protocol.train_case(
                model=model, case=case, matrix=matrices[case], labels=labels,
                image_ids=image_ids, split=split, seed=seed, device=device,
                output_dir=path.parent, split_protocol="seeded_image_level_811",
            )
            print(f"[{model}] completed {case} seed{seed}", flush=True)


def flatten(result: dict) -> dict[str, float]:
    test = result["test"]
    return {
        "auroc": float(test["auroc"]),
        "real_aupr": float(test["real"]["aupr"]),
        "hall_aupr": float(test["hall"]["aupr"]),
        "real_f1": float(test["real"]["f1"]),
        "hall_f1": float(test["hall"]["f1"]),
    }


def summarize() -> None:
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            ot_path, js_path, js_rq_path = paths(model, seed)
            for method, path in (
                ("SAR+OT", ot_path),
                ("SAR+JS", js_path),
                ("SAR+JS+R_q", js_rq_path),
            ):
                result = json.loads(path.read_text())
                if result["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
                    raise ValueError(f"{model}/{method}/seed{seed}: invalid image counts")
                rows.append({"model": model, "method": method, "seed": seed, **flatten(result)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# OT 始终含 R_q：SAR+OT 与 SAR+JS（JS 加/不加 R_q）", "",
        "LLaVA-1.5-7B、Qwen3-VL-8B；种子 43/44/45 同时控制 3200/400/400 图像划分及 MLP 初始化。复用未优化的 Log-S K=32、τ=0.03 特征；三组的 SAR 完全相同，第二部分分别为 OT_G+R_q (EMS)、top-K 并集自然对数 JS、JS+R_q。OT 始终含 R_q；JS 分别不加和加 R_q。R_q 从同一缓存的 EMS−geometry OT 求得，即 R_q(P,T)=Σ_i P_i(1−cos(q,v_i))+Σ_j T_j(1−cos(q,v_j))；JS、OT 与 R_q 使用相同源/目标分布及 top-K 并集支持。三组使用相同的单隐藏层宽度 128 MLP、训练设置与 object-token 样本。完整 EMS 结果直接复用先前同协议实验，两个 JS 方案在该协议下训练。Real/Hall F1 的阈值分别在验证集选取；测试集不参与阈值或超参选择。", "",
        "数值为三种子均值 ± 总体标准差，单位为百分数。", "",
        "| 模型 | 方法 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for method in ("SAR+OT", "SAR+JS", "SAR+JS+R_q"):
            group = [row for row in rows if row["model"] == model and row["method"] == method]
            values = [f"{100*np.mean([r[k] for r in group]):.2f} ± {100*np.std([r[k] for r in group]):.2f}" for k in METRICS]
            lines.append(f"| {model} | {method} | {' | '.join(values)} |")
    lines += ["", "## 配对差值（百分点）", "",
              "| 模型 | 对比 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        for left, right, label in (
            ("SAR+OT", "SAR+JS", "OT − JS"),
            ("SAR+OT", "SAR+JS+R_q", "OT − (JS+R_q)"),
            ("SAR+JS+R_q", "SAR+JS", "(JS+R_q) − JS"),
        ):
            differences = [
                100 * np.mean([
                    next(r[k] for r in group if r["seed"] == seed and r["method"] == left)
                    - next(r[k] for r in group if r["seed"] == seed and r["method"] == right)
                    for seed in SEEDS
                ]) for k in METRICS
            ]
            lines.append(f"| {model} | {label} | {' | '.join(f'{v:+.2f}' for v in differences)} |")
    lines += ["", "## 逐种子结果", "",
              "| 模型 | 方法 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['method']} | {row['seed']} | {' | '.join(f'{100*row[k]:.2f}' for k in METRICS)} |")
    lines += ["", "注意：两次比较中的 OT 都是 OT_G+R_q；变化只在 JS 是否加 R_q。原 SAR+JS 未包含 R_q，因此第一组不是两侧均有 R_q 的纯距离函数对照。", ""]
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
