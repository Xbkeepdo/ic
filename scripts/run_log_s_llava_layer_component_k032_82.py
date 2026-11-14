#!/usr/bin/env python3
"""Strict 8:2 LLaVA K=32 layer ablation for SAR, EMS, and SAR+EMS."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts.train_log_s_hpre_k_tau_ablation import _probe
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/log_s_hpre_k_tau_ablation/llava_1_5_7b/compact_features.npz"
OUT = ROOT / "outputs/log_s_layer_component_ablation_k032_82/llava_1_5_7b"
SAR_REFERENCE = ROOT / "outputs/linear_cost_log_s_risk_four_f_fusion/llava_1_5_7b/f__log_residual_fusion"
RISK_REFERENCE = ROOT / "outputs/log_s_hpre_k_tau_ablation/llava_1_5_7b/test_grid/log_s_hpre_k032_tau003"
RANGES = ((1, 8), (9, 18), (19, 27), (28, 32))
GROUPS = ("sar", "ems", "sar_ems")
SEEDS = (43, 44, 45)
METRICS = ("auc", "aupr", "f1")


def load_data() -> dict[str, np.ndarray]:
    with np.load(SOURCE, allow_pickle=False) as saved:
        data = {
            "sar": np.asarray(saved["f_log"], dtype=np.float32),
            "ems": np.asarray(saved["log_s_hpre_k032_tau003"], dtype=np.float32),
            "label": np.asarray(saved["label"], dtype=np.int64),
            "image_id": np.asarray(saved["image_id"], dtype=np.int64),
            "train_mask": np.asarray(saved["train_mask"], dtype=bool),
            "test_mask": np.asarray(saved["strict_test_mask"], dtype=bool),
        }
    if data["sar"].shape != data["ems"].shape or data["sar"].shape[1] != 32:
        raise ValueError("Expected matching 32-layer SAR and EMS matrices")
    if np.any(data["train_mask"] & data["test_mask"]):
        raise ValueError("Train/test token masks overlap")
    if not np.all(data["train_mask"] | data["test_mask"]):
        raise ValueError("Train/test masks do not cover the cached tokens")
    ids = data["image_id"]
    train_images = set(ids[data["train_mask"]])
    test_images = set(ids[data["test_mask"]])
    if train_images & test_images or len(train_images) != 3171 or len(test_images) != 797:
        raise ValueError("Unexpected image-level 8:2 split")
    if not np.isfinite(data["sar"]).all() or not np.isfinite(data["ems"]).all():
        raise ValueError("Non-finite cached features")
    return data


def matrix(data: dict[str, np.ndarray], group: str, start: int, end: int) -> np.ndarray:
    sar = data["sar"][:, start - 1:end]
    ems = data["ems"][:, start - 1:end]
    return sar if group == "sar" else ems if group == "ems" else np.concatenate((sar, ems), axis=1)


def reference(group: str, seed: int) -> Path:
    if group == "sar":
        return SAR_REFERENCE / f"seed{seed}" / "metrics.json"
    name = "risk" if group == "ems" else "risk_f_log"
    return RISK_REFERENCE / name / f"seed{seed}" / "metrics.json"


def train(group: str, device_name: str) -> None:
    torch.set_num_threads(1)
    data = load_data()
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    train_mask, test_mask = data["train_mask"], data["test_mask"]
    for start, end in RANGES:
        features = matrix(data, group, start, end)
        for seed in SEEDS:
            folder = OUT / group / f"layers_{start:02d}_{end:02d}" / f"seed{seed}"
            metrics_path = folder / "metrics.json"
            if metrics_path.exists():
                print(f"Skip {group} {start}-{end} seed {seed}", flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            config = _probe(root_config, seed)
            (folder / "config.json").write_text(json.dumps(_json_ready(asdict(config)), indent=2))
            print(f"Train {group} {start}-{end} seed {seed} on {device_name}", flush=True)
            result = train_and_evaluate_probe(
                X_train=features[train_mask], y_train=data["label"][train_mask],
                X_val=np.empty((0, features.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=features[test_mask], y_test=data["label"][test_mask],
                config=config, device=torch.device(device_name), output_dir=str(folder),
            )
            metrics_path.write_text(json.dumps(_json_ready(result), indent=2))
        print(f"Completed {group} layers {start}-{end}", flush=True)


def summarize() -> None:
    data = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "llava_1_5_7b", "top_k": 32, "tau": 0.03,
        "split_protocol": "strict_82_no_validation",
        "threshold_selection": "train_f1",
        "checkpoint_selection": "minimum_train_loss",
        "source": str(SOURCE), "seeds": list(SEEDS),
        "ranges_inclusive": [list(value) for value in RANGES],
        "train_tokens": int(data["train_mask"].sum()),
        "test_tokens": int(data["test_mask"].sum()),
        "train_images_with_object_mentions": int(np.unique(data["image_id"][data["train_mask"]]).size),
        "test_images_with_object_mentions": int(np.unique(data["image_id"][data["test_mask"]]).size),
        "full_references": {group: str(reference(group, 43).parent.parent) for group in GROUPS},
    }
    (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for group in GROUPS:
        for start, end in (*RANGES, (1, 32)):
            for seed in SEEDS:
                path = (reference(group, seed) if (start, end) == (1, 32)
                        else OUT / group / f"layers_{start:02d}_{end:02d}" / f"seed{seed}" / "metrics.json")
                if not path.exists():
                    raise FileNotFoundError(path)
                result = json.loads(path.read_text())
                real = result["real_positive"]
                rows.append({
                    "group": group, "layer_set": f"{start}-{end}", "seed": seed,
                    "input_dim": (end - start + 1) * (2 if group == "sar_ems" else 1),
                    "source_metrics": str(path),
                    **{name: float(real[name]) for name in METRICS},
                })
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# LLaVA Log-S 层段与组成消融：8:2，K=32", "",
        "四个互不重合的层段为 1–8、9–18、19–27、28–32。SAR、EMS 各自截取层段，SAR+EMS 拼接同一层段的两类特征；全层行复用相同 8:2 协议的已保存结果。",
        "COCO ENDAC 图片级严格 8:2 划分；含 object mention 的图片为训练 3171、测试 797，token 为训练 10109、测试 2577。K=32，τ=0.03，随机种 43/44/45。",
        "单隐藏层 128、训练集 z-score、无 BatchNorm、dropout 0.3、batch 256、drop_last、Adam LR 1e-3、weight decay 1e-5；按训练损失选 checkpoint（最多 100 epoch，patience 10），在训练集选 Real-F1 阈值。",
        "Real 为正类。AUC 指 AUROC；下表为测试集三种子均值 ± 总体标准差，单位为百分比。", "",
        "| 特征 | 层段 | 输入维度 | AUC | Real AUPR | Real F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for start, end in (*RANGES, (1, 32)):
            subset = [row for row in rows if row["group"] == group and row["layer_set"] == f"{start}-{end}"]
            cells = []
            for name in METRICS:
                values = np.asarray([row[name] for row in subset])
                cells.append(f"{100*values.mean():.2f} ± {100*values.std():.2f}")
            display = "SAR+EMS" if group == "sar_ems" else group.upper()
            lines.append(f"| {display} | {start}–{end} | {subset[0]['input_dim']} | " + " | ".join(cells) + " |")
    lines.extend(["", "逐种子数据见 `per_seed_results.csv`，源结果见 `source_metrics` 列。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize()
    elif args.group:
        train(args.group, args.device)
    else:
        parser.error("Provide --group or --summarize-only")


if __name__ == "__main__":
    main()
