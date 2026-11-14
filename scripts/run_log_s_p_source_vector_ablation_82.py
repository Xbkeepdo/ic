#!/usr/bin/env python3
"""Compare attention- versus FFN-source EMS on seeded strict COCO 8:2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_log_s_p_source_vector_ablation_811 import load_base, load_new
from scripts.train_log_s_hpre_k_tau_ablation import _fit
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
OLD_FEATURES = ROOT / "outputs/log_s_hpre_k_tau_ablation"
OUT = ROOT / "outputs/log_s_p_source_vector_ablation_82"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
FIELDS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def metrics(path: Path) -> dict[str, float]:
    test = json.loads(path.read_text())["threshold_reports"]["train_f1"]["test_metrics"]
    return {
        "auroc": float(test["real_positive"]["auc"]),
        "real_aupr": float(test["real_positive"]["aupr"]),
        "hall_aupr": float(test["hallucination_positive"]["aupr"]),
        "real_f1": float(test["real_positive"]["f1"]),
        "hall_f1": float(test["hallucination_positive"]["f1"]),
        "accuracy": float(test["accuracy"]),
    }


def result_path(model: str, seed: int, case: str) -> Path:
    if case == "sar_ef":
        return REFERENCE / model / f"seed{seed}" / "ours" / "metrics.json"
    return OUT / model / f"seed{seed}" / "sar_ea" / "metrics.json"


def run_model(model: str, device_name: str) -> None:
    torch.set_num_threads(1)
    base = load_base(model)
    new = load_new(model, base)
    with np.load(OLD_FEATURES / model / "compact_features.npz", allow_pickle=False) as old:
        metadata_equal = all(np.array_equal(base[key], old[key]) for key in (
            "image_id", "response_token_idx", "target_token_id", "label",
        ))
        risk_error = float(np.max(np.abs(base["ems"] - old["log_s_hpre_k032_tau003"])))
        sar_error = float(np.max(np.abs(base["sar"] - old["f_log"])))
    if not metadata_equal or risk_error != 0.0 or sar_error != 0.0:
        raise ValueError(f"{model}: strict 8:2 original features differ")
    if not np.array_equal(new["ffn_output"], base["ems"]):
        raise ValueError(f"{model}: extracted FFN EMS differs")
    matrix = np.concatenate((new["attention_output"], base["sar"]), axis=1).astype(np.float32)
    audit = {
        "model": model, "samples": len(base["label"]),
        "layers": int(base["sar"].shape[1]),
        "input_order": ["attention_source_EMS", "SAR"],
        "reference_input_order": ["FFN_source_EMS", "SAR"],
        "same_metadata_and_labels": metadata_equal,
        "original_FFN_EMS_max_abs_error": risk_error,
        "original_SAR_max_abs_error": sar_error,
        "seeds": list(SEEDS), "split": "strict_seeded_image_level_82",
    }
    (OUT / model).mkdir(parents=True, exist_ok=True)
    (OUT / model / "alignment.json").write_text(json.dumps(audit, indent=2) + "\n")
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    for seed in SEEDS:
        split = json.loads((REFERENCE / "splits" / f"seed{seed}.json").read_text())
        train, test = set(map(int, split["train"])), set(map(int, split["test"]))
        if (len(train), len(test)) != (3200, 800) or train & test or split["val"]:
            raise ValueError(f"seed{seed}: invalid strict 8:2 split")
        fit_mask = np.isin(base["image_id"], list(train))
        eval_mask = np.isin(base["image_id"], list(test))
        if not np.all(fit_mask ^ eval_mask):
            raise ValueError(f"{model}/seed{seed}: incomplete object-token split")
        reference = result_path(model, seed, "sar_ef")
        if not reference.exists():
            raise FileNotFoundError(reference)
        original = json.loads(reference.read_text())
        if original["split_protocol"] != "strict_82_no_validation":
            raise ValueError(f"{model}/seed{seed}: original training protocol differs")
        folder = result_path(model, seed, "sar_ea").parent
        _fit(matrix=matrix, labels=base["label"], fit_mask=fit_mask,
             eval_mask=eval_mask, seed=seed, root_config=config,
             device=torch.device(device_name), folder=folder)
        current_config = json.loads((folder / "config.json").read_text())
        old_config = json.loads((reference.parent / "config.json").read_text())
        if current_config != old_config:
            raise ValueError(f"{model}/seed{seed}: training configuration differs")
        print(f"completed {model} seed{seed}: SAR+E_A", flush=True)


def summarize() -> None:
    rows = []
    lines = [
        "# COCO4000 严格 8:2：SAR+E_A 与 SAR+E_F", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；种子 43/44/45 同时控制 3200/800 图片划分及 MLP 初始化。无独立验证集；使用原版 8:2 训练器、最小训练损失 checkpoint、训练集选 F1 阈值。两组均以 [EMS,SAR] 为输入顺序，训练集统计量标准化。$K=32$，$\\tau=0.03$；其余特征与样本完全一致。", "",
        "指标是测试集百分数；± 为三种子总体标准差。Real/Hall F1 分别使用对应正类的训练集 F1 阈值；AUROC/AUPR 无需阈值。", "",
        "| 模型 | 特征 | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        values = {}
        for case in ("sar_ef", "sar_ea"):
            values[case] = []
            for seed in SEEDS:
                row = {"model": model, "case": case, "seed": seed,
                       **metrics(result_path(model, seed, case))}
                rows.append(row)
                values[case].append(row)
            formatted = " | ".join(
                f"{100*np.mean([v[field] for v in values[case]]):.2f} ± "
                f"{100*np.std([v[field] for v in values[case]]):.2f}"
                for field in FIELDS
            )
            lines.append(f"| {model} | {'SAR+E_F (original)' if case == 'sar_ef' else 'SAR+E_A'} | {formatted} |")
        delta = " | ".join(
            f"{100*np.mean([a[field]-f[field] for a,f in zip(values['sar_ea'],values['sar_ef'])]):+.2f}"
            for field in FIELDS
        )
        lines.append(f"| {model} | Δ E_A − E_F (pp) | {delta} |")
    lines += ["", "## 逐种子", "",
              "| 模型 | 特征 | 种子 | " + " | ".join(FIELDS) + " |",
              "|---|---|---:|" + "---:|"*len(FIELDS)]
    for row in rows:
        lines.append(f"| {row['model']} | {row['case']} | {row['seed']} | " +
                     " | ".join(f"{100*row[field]:.3f}" for field in FIELDS) + " |")
    lines += ["", "各模型的 `alignment.json` 记录与原版严格 8:2 特征的逐元素核对；原始训练结果见各 `metrics.json`。", ""]
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        run_model(args.model, args.device)
    else:
        parser.error("--model or --summarize required")


if __name__ == "__main__":
    main()
