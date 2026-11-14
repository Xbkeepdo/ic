#!/usr/bin/env python3
"""Compare saved OT with newly trained rank-top16 JS+R_q on seeded COCO 8:2."""

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
OUT = ROOT / "outputs/log_s_ot_rank16_rq_seeded_82_43_45"
SOURCE_82 = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
CACHE = ROOT / "outputs/log_s_mechanism_ablation_811"
OLD_CACHE = ROOT / "outputs/log_s_hpre_k_tau_ablation"
GF_FEATURES = ROOT.parent / "gfchair/outputs/coco4000_512_log_s_js_support_variants_811"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)


def _key(row: dict) -> tuple[int, int, int]:
    return int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"])


def load_features(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(CACHE / model / "compact_features.npz", allow_pickle=False) as saved:
        sar = np.asarray(saved["sar"], dtype=np.float32)
        ems = np.asarray(saved["ems"], dtype=np.float32)
        n = sar.shape[1]
        geometry = np.asarray(saved["sar_ot_geometry"][:, n:], dtype=np.float32)
        union16 = np.asarray(saved["sar_union_js"][:, n:], dtype=np.float32)
        image_ids = np.asarray(saved["image_id"], dtype=np.int64)
        labels = np.asarray(saved["label"], dtype=np.int8)
        response_idx = np.asarray(saved["response_token_idx"], dtype=np.int32)
        target_ids = np.asarray(saved["target_token_id"], dtype=np.int32)
        keys = list(zip(image_ids.tolist(), response_idx.tolist(), target_ids.tolist()))
    with np.load(OLD_CACHE / model / "compact_features.npz", allow_pickle=False) as prior:
        old_keys = list(zip(prior["image_id"].tolist(), prior["response_token_idx"].tolist(), prior["target_token_id"].tolist()))
        if keys != old_keys or not np.array_equal(labels, prior["label"]):
            raise ValueError(f"{model}: current and 8:2 OT object tokens differ")
        if not np.array_equal(sar, prior["f_log"]) or not np.array_equal(ems, prior["log_s_hpre_k032_tau003"]):
            raise ValueError(f"{model}: current SAR/EMS differs from saved 8:2 OT")
    with (GF_FEATURES / model / "features.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    by_key = {_key(row): row for row in rows}
    if len(by_key) != len(rows) or set(by_key) != set(keys):
        raise ValueError(f"{model}: JS object-token keys differ")
    ordered = [by_key[key] for key in keys]
    if any(int(row["label"]) != int(label) for row, label in zip(ordered, labels)):
        raise ValueError(f"{model}: JS labels differ")
    union32 = np.stack([np.asarray(row["features"]["js_union32"], dtype=np.float32) for row in ordered])
    rank16 = np.stack([np.asarray(row["features"]["js_rank16"], dtype=np.float32) for row in ordered])
    if any(value.shape != sar.shape or not np.isfinite(value).all() for value in (union16, union32, rank16)):
        raise ValueError(f"{model}: invalid JS matrix")
    rq = ems - geometry
    # The saved strict-8:2 Ours checkpoint was trained in [EMS, SAR] order.
    matrices = {"ot": np.concatenate((ems, sar), axis=1)}
    for name, js in (("union16", union16), ("union32", union32), ("rank16", rank16)):
        matrices[name] = np.concatenate((js, sar), axis=1)
        matrices[f"{name}_rq"] = np.concatenate((js + rq, sar), axis=1)
    return matrices, labels, image_ids, np.stack((response_idx, target_ids), axis=1)


def test_scores(folder: Path, matrix: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    import torch
    from scripts.train_torch_probe_feature_sets import DGSTStyleProbe

    config = json.loads((folder / "config.json").read_text())
    if config["positive_class"] != "real" or not config["standardize_input"]:
        raise ValueError(f"{folder}: checkpoint convention differs")
    with np.load(folder / "input_scaler.npz", allow_pickle=False) as scaler:
        mean, scale = scaler["mean"], scaler["scale"]
    inputs = ((matrix.astype(np.float64) - mean) / scale).astype(np.float32)
    net = DGSTStyleProbe(
        input_dim=matrix.shape[1], hidden_sizes=tuple(config["hidden_sizes"]),
        dropout=float(config["dropout"]), batch_norm=bool(config["batch_norm"]),
    ).to(device)
    net.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
    net.eval()
    values = []
    batch_size = int(config["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            chunk = torch.from_numpy(inputs[start:start + batch_size]).to(device)
            values.append(torch.sigmoid(net(chunk)).squeeze(-1).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _auc_from_report(folder: Path) -> float:
    saved = json.loads((folder / "metrics.json").read_text())
    return float(saved["threshold_reports"]["train_f1"]["test_metrics"]["real_positive"]["auc"])


def run_model(model: str, device: str) -> None:
    import torch
    from sklearn.metrics import roc_auc_score
    from scripts.train_log_s_hpre_k_tau_ablation import _fit
    from utils.config_utils import load_config

    torch.set_num_threads(1)
    matrices, labels, image_ids, token_indices = load_features(model)
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    model_out = OUT / model
    model_out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        split = json.loads((SOURCE_82 / "splits" / f"seed{seed}.json").read_text())
        train_ids, test_ids = set(split["train"]), set(split["test"])
        if len(train_ids) != 3200 or len(test_ids) != 800 or train_ids & test_ids:
            raise ValueError(f"seed{seed}: invalid 8:2 split")
        fit_mask, test_mask = np.isin(image_ids, list(train_ids)), np.isin(image_ids, list(test_ids))
        if not np.all(fit_mask ^ test_mask):
            raise ValueError(f"{model}/seed{seed}: object tokens are not partitioned")
        ot_folder = SOURCE_82 / model / f"seed{seed}" / "ours"
        js_folder = model_out / f"seed{seed}" / "rank16_rq"
        _fit(
            matrix=matrices["rank16_rq"], labels=labels,
            fit_mask=fit_mask, eval_mask=test_mask,
            seed=seed, root_config=root_config,
            device=torch.device(device), folder=js_folder,
        )
        old_config = json.loads((ot_folder / "config.json").read_text())
        new_config = json.loads((js_folder / "config.json").read_text())
        if old_config != new_config:
            raise ValueError(f"{model}/seed{seed}: OT and JS training configurations differ")
        test_labels = labels[test_mask]
        ot_scores = test_scores(ot_folder, matrices["ot"][test_mask], device=device)
        js_scores = test_scores(js_folder, matrices["rank16_rq"][test_mask], device=device)
        ot_auc = float(roc_auc_score(test_labels, ot_scores))
        js_auc = float(roc_auc_score(test_labels, js_scores))
        if abs(ot_auc - _auc_from_report(ot_folder)) > 1e-8 or abs(js_auc - _auc_from_report(js_folder)) > 1e-8:
            raise ValueError(
                f"{model}/seed{seed}: recovered score AUC differs from saved result: "
                f"OT {ot_auc} vs {_auc_from_report(ot_folder)}, "
                f"JS {js_auc} vs {_auc_from_report(js_folder)}"
            )
        output = model_out / f"seed{seed}" / "paired_scores.npz"
        np.savez_compressed(
            output,
            image_id=image_ids[test_mask],
            response_token_idx=token_indices[test_mask, 0],
            target_token_id=token_indices[test_mask, 1],
            label=test_labels,
            ot_real_score=ot_scores,
            js_real_score=js_scores,
        )
        audit = {
            "model": model, "seed": seed,
            "train_images": len(train_ids), "test_images": len(test_ids),
            "train_tokens": int(fit_mask.sum()), "test_tokens": int(test_mask.sum()),
            "feature_order": "[OT or JS+R_q, SAR]",
            "same_training_configuration": True,
            "ot_auc_recovered": ot_auc, "js_auc_recovered": js_auc,
            "ot_auc_report": _auc_from_report(ot_folder),
            "js_auc_report": _auc_from_report(js_folder),
        }
        (model_out / f"seed{seed}" / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(f"{model} seed{seed}: OT {100*ot_auc:.2f}, JS+R_q {100*js_auc:.2f}, tokens {len(test_labels)}", flush=True)


def summarize() -> None:
    from statistics import mean, pstdev

    rows = []
    for model in MODELS:
        for seed in SEEDS:
            audit = json.loads((OUT / model / f"seed{seed}" / "audit.json").read_text())
            rows.append({
                "model": model, "seed": seed,
                "test_images": audit["test_images"], "test_tokens": audit["test_tokens"],
                "ot_auroc": audit["ot_auc_recovered"], "js_auroc": audit["js_auc_recovered"],
                "delta_ot_js": audit["ot_auc_recovered"] - audit["js_auc_recovered"],
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# COCO 8:2：OT 与独立 Top16 排名 JS+R_q", "",
        "LLaVA-1.5-7B 和 Qwen3-VL-8B；种子 43/44/45 同时决定 3200/800 图片划分和单隐藏层 MLP 初始化。OT 使用原 8:2 已保存的 SAR+EMS 检测器；JS+R_q 用相同训练器、配置与 object token，新训练并恢复逐 token 测试分数。OT 和 JS 分别排在 SAR 前，保持原 8:2 特征顺序。AUROC 以 real 为正类。", "",
        "| 模型 | OT AUROC (%) | JS+R_q AUROC (%) | OT−JS (pp) |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        group = [row for row in rows if row["model"] == model]
        fmt = lambda key: f"{100*mean(row[key] for row in group):.2f} ± {100*pstdev(row[key] for row in group):.2f}"
        delta = 100 * mean(row["delta_ot_js"] for row in group)
        if abs(delta) < 0.005:
            delta = 0.0
        lines.append(f"| {model} | {fmt('ot_auroc')} | {fmt('js_auroc')} | {delta:+.2f} |")
    lines += ["", "## 逐种子结果", "",
              "| 模型 | 种子 | 测试图像 | 测试 object token | OT AUROC (%) | JS+R_q AUROC (%) | OT−JS (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['seed']} | {row['test_images']} | {row['test_tokens']} | {100*row['ot_auroc']:.2f} | {100*row['js_auroc']:.2f} | {100*row['delta_ot_js']:+.2f} |")
    lines += ["", "逐 token 对齐测试分数见各模型 `seed*/paired_scores.npz`；配对逐图 Bootstrap 结果另见 `bootstrap_3000/`。", ""]
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
