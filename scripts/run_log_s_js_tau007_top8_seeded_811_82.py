#!/usr/bin/env python3
"""Train tau=0.07 Top8 union JS with/without R_q on COCO 8:1:1 and 8:2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
from statistics import mean, pstdev
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/log_s_js_tau007_top8_seeded_811_82_43_45"
GF = ROOT.parent / "gfchair/outputs/coco4000_512_log_s_js_tau007_top8_811"
CACHE = ROOT / "outputs/log_s_mechanism_ablation_811"
OLD_CACHE = ROOT / "outputs/log_s_hpre_k_tau_ablation"
REF_811 = ROOT / "outputs/log_s_ems_i_mean_svar_seeded_811_43_45"
REF_82 = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
PRIOR_82 = ROOT / "outputs/log_s_ot_rank16_rq_seeded_82_43_45"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
CASES = ("union8", "union8_rq")
METRICS = ("auroc", "real_aupr", "hall_aupr")


def key(row: dict) -> tuple[int, int, int]:
    return (int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"]))


def load_features(model: str):
    with np.load(CACHE / model / "compact_features.npz", allow_pickle=False) as saved:
        sar = np.asarray(saved["sar"], dtype=np.float32)
        ems = np.asarray(saved["ems"], dtype=np.float32)
        n = sar.shape[1]
        geometry = np.asarray(saved["sar_ot_geometry"][:, n:], dtype=np.float32)
        image_ids = np.asarray(saved["image_id"], dtype=np.int64)
        labels = np.asarray(saved["label"], dtype=np.int8)
        response_idx = np.asarray(saved["response_token_idx"], dtype=np.int32)
        target_ids = np.asarray(saved["target_token_id"], dtype=np.int32)
    keys = list(zip(image_ids.tolist(), response_idx.tolist(), target_ids.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{model}: duplicate object token keys")
    with np.load(OLD_CACHE / model / "compact_features.npz", allow_pickle=False) as old:
        old_keys = list(zip(old["image_id"].tolist(), old["response_token_idx"].tolist(), old["target_token_id"].tolist()))
        if keys != old_keys or not np.array_equal(labels, old["label"]):
            raise ValueError(f"{model}: saved 8:2 object tokens differ")
        if not np.array_equal(sar, old["f_log"]) or not np.array_equal(ems, old["log_s_hpre_k032_tau003"]):
            raise ValueError(f"{model}: saved 8:2 OT features differ")
    with (GF / model / "features.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    by_key = {key(row): row for row in raw}
    if len(by_key) != len(raw) or set(by_key) != set(keys):
        raise ValueError(f"{model}: Top8 JS object token keys differ")
    ordered = [by_key[item] for item in keys]
    if any(int(row["label"]) != int(label) for row, label in zip(ordered, labels)):
        raise ValueError(f"{model}: Top8 JS labels differ")
    js = np.stack([np.asarray(row["features"]["js_union8"], dtype=np.float32) for row in ordered])
    if js.shape != sar.shape or not np.isfinite(js).all():
        raise ValueError(f"{model}: invalid Top8 JS")
    rq = ems - geometry
    matrices_811 = {
        "union8": np.concatenate((sar, js), axis=1),
        "union8_rq": np.concatenate((sar, js + rq), axis=1),
    }
    matrices_82 = {
        "union8": np.concatenate((js, sar), axis=1),
        "union8_rq": np.concatenate((js + rq, sar), axis=1),
    }
    return matrices_811, matrices_82, labels, image_ids, response_idx, target_ids


def load_811_split(seed: int) -> dict[str, set[int]]:
    raw = json.loads((REF_811 / "splits" / f"seed{seed}.json").read_text())
    if {name: len(ids) for name, ids in raw.items()} != {"train": 3200, "val": 400, "test": 400}:
        raise ValueError(f"seed{seed}: invalid 8:1:1 split")
    split = {name: set(ids) for name, ids in raw.items()}
    if len(set.union(*split.values())) != 4000:
        raise ValueError(f"seed{seed}: overlapping 8:1:1 split")
    return split


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "real_aupr": float(average_precision_score(labels, scores)),
        "hall_aupr": float(average_precision_score(1 - labels, 1 - scores)),
    }


def run_model(model: str, device: str) -> None:
    import torch
    from scripts import run_log_s_mechanism_ablation_811 as protocol
    from scripts.run_log_s_ot_rank16_rq_seeded_82 import test_scores
    from scripts.train_log_s_hpre_k_tau_ablation import _fit
    from utils.config_utils import load_config

    torch.set_num_threads(1)
    matrices_811, matrices_82, labels, image_ids, response_idx, target_ids = load_features(model)
    protocol.DISPLAY.update({
        "sar_js_union8": "SAR + JS Top8 union",
        "sar_js_union8_rq": "SAR + JS Top8 union + R_q",
    })
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for seed in SEEDS:
        split = load_811_split(seed)
        expected = {part: int(np.isin(image_ids, list(ids)).sum()) for part, ids in split.items()}
        ot_811 = json.loads((REF_811 / model / f"seed{seed}" / "sar_ems/result.json").read_text())
        if ot_811["split"] != "seeded_image_level_811" or ot_811["token_counts"] != expected:
            raise ValueError(f"{model}/seed{seed}: 8:1:1 OT split mismatch")
        for case in CASES:
            case_name = f"sar_js_{case}"
            folder = OUT / "811" / model / f"seed{seed}" / case_name
            result_path = folder / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved["token_counts"] != expected:
                    raise ValueError(f"{model}/seed{seed}/{case}: existing 8:1:1 result mismatch")
            else:
                protocol.train_case(
                    model=model, case=case_name, matrix=matrices_811[case],
                    labels=labels, image_ids=image_ids, split=split, seed=seed,
                    device=device, output_dir=folder, split_protocol="seeded_image_level_811",
                )
            print(f"{model} 811 seed{seed} {case} done", flush=True)
        split82 = json.loads((REF_82 / "splits" / f"seed{seed}.json").read_text())
        train_ids, test_ids = set(split82["train"]), set(split82["test"])
        if len(train_ids) != 3200 or len(test_ids) != 800 or train_ids & test_ids:
            raise ValueError(f"seed{seed}: invalid 8:2 split")
        fit_mask = np.isin(image_ids, list(train_ids))
        test_mask = np.isin(image_ids, list(test_ids))
        if not np.all(fit_mask ^ test_mask):
            raise ValueError(f"{model}/seed{seed}: invalid 8:2 object split")
        ot_folder = REF_82 / model / f"seed{seed}" / "ours"
        ot_config = json.loads((ot_folder / "config.json").read_text())
        with np.load(PRIOR_82 / model / f"seed{seed}" / "paired_scores.npz", allow_pickle=False) as prior:
            expected_rows = {
                "image_id": image_ids[test_mask],
                "response_token_idx": response_idx[test_mask],
                "target_token_id": target_ids[test_mask],
                "label": labels[test_mask],
            }
            if any(not np.array_equal(prior[name], value) for name, value in expected_rows.items()):
                raise ValueError(f"{model}/seed{seed}: 8:2 OT scores not aligned")
            scores = {"ot": np.asarray(prior["ot_real_score"], dtype=np.float32)}
        reported = json.loads((ot_folder / "metrics.json").read_text())["threshold_reports"]["train_f1"]["test_metrics"]
        baseline = score_metrics(labels[test_mask], scores["ot"])
        if abs(baseline["auroc"] - float(reported["real_positive"]["auc"])) > 1e-6:
            raise ValueError(f"{model}/seed{seed}: recovered OT AUROC differs")
        metrics = {"ot": baseline}
        for case in CASES:
            folder = OUT / "82" / model / f"seed{seed}" / case
            if not (folder / "metrics.json").exists():
                _fit(
                    matrix=matrices_82[case], labels=labels, fit_mask=fit_mask,
                    eval_mask=test_mask, seed=seed, root_config=config,
                    device=torch.device(device), folder=folder,
                )
            if json.loads((folder / "config.json").read_text()) != ot_config:
                raise ValueError(f"{model}/seed{seed}/{case}: 8:2 probe config mismatch")
            scores[case] = test_scores(folder, matrices_82[case][test_mask], device=device)
            metrics[case] = score_metrics(labels[test_mask], scores[case])
            saved = json.loads((folder / "metrics.json").read_text())["threshold_reports"]["train_f1"]["test_metrics"]
            if abs(metrics[case]["auroc"] - float(saved["real_positive"]["auc"])) > 1e-6:
                raise ValueError(f"{model}/seed{seed}/{case}: recovered AUROC differs")
            print(f"{model} 82 seed{seed} {case} done", flush=True)
        destination = OUT / "82" / model / f"seed{seed}"
        np.savez_compressed(
            destination / "paired_scores.npz",
            **expected_rows, **{f"{case}_real_score": value for case, value in scores.items()},
        )
        (destination / "metrics.json").write_text(json.dumps({
            "model": model, "seed": seed, "train_images": 3200, "test_images": 800,
            "test_tokens": int(test_mask.sum()), "js_source_tau": 0.07,
            "ot_and_rq_source_tau": 0.03, "methods": metrics,
        }, indent=2) + "\n")


def summarize() -> None:
    rows = []
    for split in ("811", "82"):
        for model in MODELS:
            for seed in SEEDS:
                for case in ("ot", *CASES):
                    if split == "811":
                        folder = (REF_811 / model / f"seed{seed}" / "sar_ems") if case == "ot" else (OUT / split / model / f"seed{seed}" / f"sar_js_{case}")
                        test = json.loads((folder / "result.json").read_text())["test"]
                        metrics = {
                            "auroc": float(test["auroc"]),
                            "real_aupr": float(test["real"]["aupr"]),
                            "hall_aupr": float(test["hall"]["aupr"]),
                        }
                    else:
                        metrics = json.loads((OUT / split / model / f"seed{seed}" / "metrics.json").read_text())["methods"][case]
                    rows.append({"split": split, "model": model, "seed": seed, "case": case, **metrics})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# COCO：Top8 并集 JS（tau=0.07）与固定 OT",
        "",
        "LLaVA-1.5-7B、Qwen3-VL-8B；种子 43/44/45 同时控制图片划分与 MLP 初始化。8:1:1 为 3200/400/400，8:2 为 3200/800。JS 在 Top8(P)∪Top8(T) 支持集上分别重归一化后计算自然对数 JS；分别测试 JS 与 JS+R_q。OT 与 R_q 固定原 K=32、tau=0.03，JS 的 P 使用缓存 tau=0.07，T 与 SAR 不变。三种子测试均值 ± 总体标准差，单位百分数。",
        "",
        "| 划分 | 模型 | 方法 | AUROC | Real AUPR | Hall AUPR |",
        "|---|---|---|---:|---:|---:|",
    ]
    for split in ("811", "82"):
        for model in MODELS:
            for case in ("ot", *CASES):
                subset = [row for row in rows if row["split"] == split and row["model"] == model and row["case"] == case]
                values = [f"{100*mean(row[metric] for row in subset):.2f} ± {100*pstdev(row[metric] for row in subset):.2f}" for metric in METRICS]
                lines.append(f"| {split} | {model} | {case} | {' | '.join(values)} |")
    lines += ["", "逐种子未四舍五入指标见 per_seed.csv；配对逐图 Bootstrap 见 bootstrap_3000。", ""]
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
