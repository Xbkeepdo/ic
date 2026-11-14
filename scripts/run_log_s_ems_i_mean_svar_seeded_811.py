#!/usr/bin/env python3
"""Compare SAR+EMS, EMS+I+E, and native SVAR with seeded 8:1:1 image splits."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from detection.baselines import DHCPMLP
from features.baseline import baseline_config
from scripts import train_baselines
from scripts import run_log_s_mechanism_ablation_811 as ours_protocol
from scripts import run_endac_baselines_811_dual_threshold as baseline_protocol
from scripts.run_coco_seeded_image_split_auc_aupr import CachedDHCPRecordDataset
from scripts.run_log_s_ems_i_semantic_mean_811 import matrices as simple_matrices
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
REFERENCE_82 = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
MODELS = (
    "llava_1_5_7b", "qwen2_5_vl_7b", "internvl_2_5_8b", "qwen3_vl_8b",
)
SEEDS = (43, 44, 45)
OWN_CASES = ("sar_ems", "ems_i_e_sum", "ems_i_e_mean")
CASES = (
    "sar_ems", "ems_i_e_sum", "ems_i_e_mean", "metatoken_gb", "dhcp",
    "svar", "projectaway", "token_grounding_mlp", "inslen",
)
DISPLAY = {
    "sar_ems": "EMS+SAR (ours)",
    "ems_i_e_sum": "EMS+I+E_sum",
    "ems_i_e_mean": "EMS+I+E_mean",
    "metatoken_gb": "MetaToken (GB)",
    "dhcp": "DHCP (native)",
    "svar": "SVAR (native)",
    "projectaway": "ProjectAway",
    "token_grounding_mlp": "TokenGrounding (MLP)",
    "inslen": "InsLen",
}
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1")


def image_pool() -> np.ndarray:
    source = json.loads((REFERENCE_82 / "splits" / "seed43.json").read_text())
    pool = np.asarray(sorted(set(source["train"] + source["test"])), dtype=np.int64)
    if len(pool) != 4000:
        raise ValueError("expected exactly 4000 image IDs")
    return pool


def seeded_split(seed: int, pool: np.ndarray) -> dict[str, list[int]]:
    shuffled = np.random.default_rng(seed).permutation(pool)
    split = {
        "train": np.sort(shuffled[:3200]).astype(int).tolist(),
        "val": np.sort(shuffled[3200:3600]).astype(int).tolist(),
        "test": np.sort(shuffled[3600:]).astype(int).tolist(),
    }
    reference = json.loads((REFERENCE_82 / "splits" / f"seed{seed}.json").read_text())
    if split["train"] != reference["train"] or set(split["val"] + split["test"]) != set(reference["test"]):
        raise ValueError(f"seed{seed}: split differs from existing seeded 8:2 image permutation")
    path = OUT / "splits" / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != split:
        raise ValueError(f"seed{seed}: existing 8:1:1 split differs")
    if not path.exists():
        baseline_protocol.atomic_json(path, split)
    return split


def prepare_model(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
    matrices, labels, image_ids = simple_matrices(model, audit_dir=OUT)
    with np.load(ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz", allow_pickle=False) as source:
        if not (np.array_equal(labels, source["label"]) and np.array_equal(image_ids, source["image_id"])):
            raise ValueError(f"{model}: mechanism metadata differs")
        matrices["sar_ems"] = np.asarray(source["sar_ems"], dtype=np.float32)
        keys = list(zip(source["image_id"].tolist(), source["response_token_idx"].tolist(), source["target_token_id"].tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate object-token keys")
    records = baseline_protocol.load_baseline_records(model)
    baseline_labels = {baseline_protocol.key_of(row): int(row["label"]) for row in records}
    if len(baseline_labels) != len(records) or baseline_labels != dict(zip(keys, labels.astype(int).tolist())):
        raise ValueError(f"{model}: SVAR and Ours mention keys/labels differ")
    if any(len(value) != len(labels) for value in matrices.values()):
        raise ValueError(f"{model}: feature matrix length differs")
    return matrices, labels, image_ids, records


def dhcp_shard_root(model: str) -> Path:
    local = Path("/tmp/coco_seeded_dhcp_cache") / f"{model}_shards"
    if local.is_dir() and (local / "manifest.json").exists():
        return local
    source = ROOT / "outputs" / model / baseline_protocol.BASELINE_RUN / "baseline/dhcp/shards"
    if not source.is_dir():
        raise FileNotFoundError(source)
    return source


def run_dhcp(
    *, model: str, seed: int, split_records: dict, split: dict,
    baseline_cfg: dict, device: str, folder: Path,
) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "checkpoint").mkdir(parents=True, exist_ok=True)
    baseline_dir = OUT / model / "baseline"
    shard_link = baseline_dir / "dhcp/shards"
    shard_link.parent.mkdir(parents=True, exist_ok=True)
    source = dhcp_shard_root(model)
    if not shard_link.exists():
        shard_link.symlink_to(source.resolve(), target_is_directory=True)
    if shard_link.resolve() != source.resolve():
        raise ValueError(f"{model}: DHCP shard link differs from source")
    train_baselines.DHCPRecordDataset = CachedDHCPRecordDataset
    native = train_baselines._train_dhcp(
        split_records, baseline_dir, folder / "checkpoint", baseline_cfg,
        torch.device(device), "real",
    )
    checkpoint = torch.load(native["checkpoint"], map_location="cpu", weights_only=False)
    net = DHCPMLP(
        int(np.prod(checkpoint["input_shape"])), hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(torch.device(device))
    net.load_state_dict(checkpoint["state_dict"])
    shard_dir = baseline_dir / "dhcp/shards"
    batches = int(baseline_cfg.get("dhcp", {}).get("batch_size", 1024))
    scores = {}
    labels = {}
    for part in ("val", "test"):
        dataset = CachedDHCPRecordDataset(split_records[part], shard_dir)
        _, scores[part] = train_baselines._stream_scores(
            net, DataLoader(dataset, batch_size=batches, num_workers=0),
            torch.device(device), nn.CrossEntropyLoss(),
        )
        labels[part] = np.asarray([row["label"] for row in dataset.records], dtype=np.int8)
    report = baseline_protocol.dual_threshold_report(
        val_labels=labels["val"], val_hall_scores=scores["val"],
        test_labels=labels["test"], test_hall_scores=scores["test"],
    )
    np.savez_compressed(
        folder / "scores.npz",
        val_label=labels["val"], val_hall_score=scores["val"],
        test_label=labels["test"], test_hall_score=scores["test"],
    )
    return {
        "model": model,
        "method": "dhcp",
        "method_display": DISPLAY["dhcp"],
        "seed": seed,
        "split_protocol": "seeded_image_level_811",
        "image_counts": {name: len(ids) for name, ids in split.items()},
        "token_counts": {name: len(labels[name]) for name in ("val", "test")},
        "threshold_selection": {"real": "validation_real_f1", "hall": "validation_hall_f1"},
        "test": report["test"],
        "validation": report["validation"],
        "thresholds": report["thresholds"],
        "native_training": baseline_protocol.compact_native_metadata(native),
    }


def run_model(model: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)
    matrix_by_case, labels, image_ids, records = prepare_model(model)
    ours_protocol.DISPLAY.update({case: DISPLAY[case] for case in OWN_CASES})
    root_config = load_config(str(baseline_protocol.CONFIG_PATH))
    base_cfg = baseline_config(root_config)
    base_cfg["metatoken"]["classifiers"] = ["gb"]
    base_cfg["token_grounding"]["classifiers"] = ["mlp"]
    pool = image_pool()
    for seed in SEEDS:
        split_list = seeded_split(seed, pool)
        split = {key: set(values) for key, values in split_list.items()}
        split_records = baseline_protocol.split_rows(records, split)
        for name, ids in split.items():
            group_labels = set(labels[np.isin(image_ids, list(ids))].tolist())
            if group_labels != {0, 1}:
                raise ValueError(f"{model}/seed{seed}/{name}: missing class")
        for case in CASES:
            folder = OUT / model / f"seed{seed}" / case
            path = folder / "result.json"
            if path.exists():
                print(f"[{model}] reuse seed{seed} {case}", flush=True)
                continue
            if case not in OWN_CASES:
                seed_cfg = copy.deepcopy(base_cfg)
                seed_cfg["seed"] = seed
                if case == "dhcp":
                    result = run_dhcp(
                        model=model, seed=seed, split_records=split_records,
                        split=split, baseline_cfg=seed_cfg, device=device, folder=folder,
                    )
                else:
                    result = baseline_protocol.run_method(
                        method=case, model=model, seed=seed, split_records=split_records,
                        split=split, baseline_cfg=seed_cfg, device=device, output_dir=folder,
                    )
                result["split_protocol"] = "seeded_image_level_811"
                baseline_protocol.atomic_json(path, result)
            else:
                ours_protocol.train_case(
                    model=model, case=case, matrix=matrix_by_case[case], labels=labels,
                    image_ids=image_ids, split=split, seed=seed, device=device,
                    output_dir=folder, split_protocol="seeded_image_level_811",
                )
            print(f"[{model}] completed seed{seed} {case}", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO：随机种子同时控制 8:1:1 图片划分和模型训练", "",
        "比较 LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL2.5-8B、Qwen3-VL-8B 上的 EMS+SAR、两种简单拼接和六种 baseline。",
        "每个种子 43/44/45 均从同一 4000 图池独立产生 3200 train、400 val、400 test；3200 train 与先前相同种子的 8:2 实验相同。",
        "对需要训练检测器的方法，相同种子还控制初始化和训练随机性；training-free 方法仅随图片划分变化。所有方法使用相同 object token、标签和图片划分；检测阈值分别用验证集 Real/Hall F1 选取，测试集不参与选参。",
        "EMS K=32、tau=0.03。E_sum 是视觉 token 目标词概率和，E_mean 除以当前样本视觉 token 数。SVAR、DHCP 使用现有原生训练器；MetaToken 使用 GB，TokenGrounding 使用 MLP。ProjectAway 和 InsLen 为 training-free。± 为三个种子的总体标准差，指标单位为百分数。", "",
        "| 模型 | 方法 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for case in CASES:
            values = []
            for seed in SEEDS:
                path = OUT / model / f"seed{seed}" / case / "result.json"
                result = json.loads(path.read_text())
                if result["image_counts"] != {"train": 3200, "val": 400, "test": 400}:
                    raise ValueError(f"{model}/seed{seed}/{case}: invalid image counts")
                row = baseline_protocol.flatten_result(result)
                rows.append({"model": model, "case": DISPLAY[case], "seed": seed, **row})
                values.append(row)
            display = " | ".join(
                f"{100*np.mean([row[key] for row in values]):.2f} ± {100*np.std([row[key] for row in values]):.2f}"
                for key in METRICS
            )
            lines.append(f"| {model} | {DISPLAY[case]} | {display} |")
    lines.extend(["", "## 逐种子测试集结果", "",
                  "| 模型 | 方法 | 种子 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        display = " | ".join(f"{100*row[key]:.2f}" for key in METRICS)
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | {display} |")
    lines.extend(["", "逐种子划分见 `splits/`，全部指标的三种子均值与标准差见 `aggregate_all_metrics.csv`，逐种子指标见 `per_seed.csv` 和各运行目录下的 `result.json`。", ""])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete_results.md").write_text("\n".join(lines))
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = [key for key in rows[0] if key not in ("model", "case", "seed")]
    aggregates = []
    for model in MODELS:
        for case in CASES:
            values = [row for row in rows if row["model"] == model and row["case"] == DISPLAY[case]]
            aggregate = {"model": model, "case": DISPLAY[case]}
            for key in metric_keys:
                aggregate[f"{key}_mean"] = float(np.mean([row[key] for row in values]))
                aggregate[f"{key}_std"] = float(np.std([row[key] for row in values]))
            aggregates.append(aggregate)
    with (OUT / "aggregate_all_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
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
        parser.error("--model is required unless --summarize")


if __name__ == "__main__":
    main()
