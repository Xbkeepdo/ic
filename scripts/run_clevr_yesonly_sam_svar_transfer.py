#!/usr/bin/env python3
"""Train CLEVR Yes-only SAM/SVAR once and evaluate unchanged on POPE/AMBER."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.qa_probe import classification_metrics
from features.baseline import baseline_config, svar_training_vector
from features.qa_baseline import qa_image_identity
from scripts.evaluate_coco_clevr_log_s_transfer import _predict_qa
from scripts.run_amber_exist_clevr_svar_transfer import predict as predict_svar
from scripts.run_object_position_transfer_comparison import fit_ours_82
from scripts.train_baselines import _train_svar
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/root/rivermind-data/ic/outputs/qa_benchmarks")
TARGET_ROOT = ROOT / "outputs/qa_benchmarks"
OUTPUT_ROOT = TARGET_ROOT / "clevr_yesonly_sam_svar_to_pope_amber_82"
SOURCE_RUN = "CLEVR9K-LOGS-HPR-K32-TAU003-W1-SIXBASELINES"
TARGET_RUN = "POPE-POPULAR-AMBER-EXIST-SAM-SVAR-YESONLY"
ANSWER = "prompt_last_token"
OBJECT = "question_object_pre_token"
PROTOCOL = "object_hallucination_yes_only"
SEEDS = (43, 44, 45)


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check_log_profile(payload: dict, key: str) -> None:
    expected = {
        "tau": 0.03,
        "transport_top_k": 32,
        "semantic_cost_weight": 1.0,
        "endpoint_penalty_weight": 1.0,
        "probability_mode": "final_norm_hpre",
        "cost_mode": "linear_state_plus_two_endpoint_penalties",
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"{key}: incompatible {name}: {payload.get(name)!r}")


def load_clevr(model: str, source_root: Path) -> tuple[dict, dict]:
    source = source_root / model / SOURCE_RUN / "clevr_exist_9k"
    with (source / "features.pkl").open("rb") as handle:
        method_rows = pickle.load(handle)
    baseline_path = source / "baseline" / OBJECT / PROTOCOL / "features.pkl"
    with baseline_path.open("rb") as handle:
        baseline_rows = pickle.load(handle)
    yes_rows = {
        str(row["key"]): row
        for row in method_rows
        if row.get("object_hallucination_yes_only_label") in (0, 1)
    }
    if len(yes_rows) != len(baseline_rows):
        raise ValueError(f"{model}: CLEVR Yes-only count mismatch")
    if len(yes_rows) != sum(row.get("object_hallucination_yes_only_label") in (0, 1) for row in method_rows):
        raise ValueError(f"{model}: duplicate CLEVR method key")
    if len({str(row["key"]) for row in baseline_rows}) != len(baseline_rows):
        raise ValueError(f"{model}: duplicate CLEVR SVAR key")
    if set(yes_rows) != {str(row["key"]) for row in baseline_rows}:
        raise ValueError(f"{model}: CLEVR method and SVAR keys differ")

    groups = {"train": [], "test": []}
    images: dict[str, str] = {}
    digest = hashlib.sha256()
    widths = {"sam": set(), "svar": set()}
    for baseline in baseline_rows:
        key = str(baseline["key"])
        method = yes_rows[key]
        label = int(method["object_hallucination_yes_only_label"])
        split = str(method["probe_split"])
        if split not in groups:
            raise ValueError(f"{key}: unexpected CLEVR split {split}")
        if (int(baseline["label"]) != label
                or str(baseline["probe_split"]) != split
                or baseline.get("qa_label_protocol") != PROTOCOL
                or baseline.get("qa_position_protocol") != OBJECT
                or method.get("prediction") != "yes"
                or qa_image_identity(baseline) != qa_image_identity(method)):
            raise ValueError(f"{key}: CLEVR label, position, or image mismatch")
        image = str(qa_image_identity(method))
        if images.setdefault(image, split) != split:
            raise ValueError(f"{key}: CLEVR image leakage")
        object_log = method["positions"][OBJECT]["log_s_hpre"]
        answer_log = method["positions"][ANSWER]["log_s_hpre"]
        _check_log_profile(object_log, key)
        _check_log_profile(answer_log, key)
        risk = np.asarray(object_log["risk"], dtype=np.float32).reshape(-1)
        f_log = np.asarray(answer_log["f_log"], dtype=np.float32).reshape(-1)
        if risk.size == 0 or risk.shape != f_log.shape:
            raise ValueError(f"{key}: CLEVR SAM layer mismatch")
        sam = np.concatenate((risk, f_log))
        svar = np.asarray(
            svar_training_vector(baseline["baselines"]["svar"], layer_start=5, layer_end=19),
            dtype=np.float32,
        ).reshape(-1)
        if not np.isfinite(sam).all() or not np.isfinite(svar).all():
            raise ValueError(f"{key}: non-finite CLEVR feature")
        widths["sam"].add(sam.size)
        widths["svar"].add(svar.size)
        digest.update(key.encode())
        digest.update(bytes([label]))
        digest.update(sam.tobytes())
        digest.update(svar.tobytes())
        groups[split].append({
            "key": key,
            "label": label,
            "image": image,
            "image_id": int(baseline["image_id"]),
            "response_token_idx": int(baseline["response_token_idx"]),
            "sam": sam,
            "svar": svar,
        })
    if len(widths["sam"]) != 1 or widths["svar"] != {448}:
        raise ValueError(f"{model}: inconsistent CLEVR feature widths {widths}")
    counts = {
        split: {"real": sum(row["label"] == 1 for row in rows),
                "hallucination": sum(row["label"] == 0 for row in rows),
                "images": len({row["image"] for row in rows})}
        for split, rows in groups.items()
    }
    if any(min(counts[split]["real"], counts[split]["hallucination"]) == 0 for split in groups):
        raise ValueError(f"{model}: single-class CLEVR split: {counts}")
    audit = {
        "model": model,
        "source": str(source),
        "label_protocol": PROTOCOL,
        "split_protocol": "original_CLEVR_image_level_82",
        "sam_definition": "concat(risk@question_object_pre_token,f_log@prompt_last_token)",
        "svar_definition": "SVAR@question_object_pre_token, layers [5,19)",
        "counts": counts,
        "widths": {key: next(iter(values)) for key, values in widths.items()},
        "cohort_feature_sha256": digest.hexdigest(),
    }
    return groups, audit


def load_targets(model: str, target_root: Path, source_widths: dict) -> dict:
    cohorts = {}
    for dataset in ("pope", "amber_discriminative"):
        run = target_root / model / TARGET_RUN / dataset
        summary = _json(run / "sam_svar_cross_position_summary.json")
        with np.load(run / "sam_object_risk_answer_f_log_vs_svar_object_yesonly.npz") as payload:
            keys = payload["keys"].astype(str)
            sam = payload["sam"].astype(np.float32)
            svar = payload["svar"].astype(np.float32)
            labels = payload["labels"].astype(np.int64)
            splits = payload["splits"].astype(str)
        if (len(set(keys)) != len(keys)
                or sam.shape != (len(keys), source_widths["sam"])
                or svar.shape != (len(keys), source_widths["svar"])
                or not np.isfinite(sam).all()
                or not np.isfinite(svar).all()
                or not np.isin(labels, (0, 1)).all()
                or set(splits) - {"train", "test"}):
            raise ValueError(f"{model}/{dataset}: invalid target matrix")
        if (summary["num_questions"] != len(keys)
                or summary["sam_shape"] != list(sam.shape)
                or summary["svar_shape"] != list(svar.shape)
                or summary["counts_by_label"] != {str(i): int((labels == i).sum()) for i in (0, 1)}):
            raise ValueError(f"{model}/{dataset}: target summary mismatch")
        cohorts[dataset] = {
            "keys": keys,
            "sam": sam,
            "svar": svar,
            "labels": labels,
            "splits": splits,
            "summary_path": str(run / "sam_svar_cross_position_summary.json"),
        }
    return cohorts


def _training_records(groups: dict, method: str) -> dict:
    records = {}
    for split, rows in groups.items():
        records[split] = []
        for row in rows:
            common = {"key": row["key"], "label": row["label"], "probe_split": split}
            if method == "sam":
                records[split].append({
                    **common,
                    "positions": {ANSWER: {"dgst": {"logs_risk_f_log": row["sam"]}}},
                })
            else:
                records[split].append({
                    **common,
                    "baseline_schema_version": "1.0",
                    "image_id": row["image_id"],
                    "response_token_idx": row["response_token_idx"],
                    "baselines": {"svar": {"vector": row["svar"]}},
                })
    if method == "svar":
        records["val"] = records["train"]
    return records


def train(model: str, groups: dict, audit: dict, output: Path,
          *, training_seed: int | None = None) -> None:
    config = load_config(str(ROOT / "configs/model_configs_qa_logs_k32_tau003.yaml"))
    base_cfg = baseline_config(config)
    for method in ("sam", "svar"):
        records = _training_records(groups, method)
        for seed in SEEDS:
            folder = output / model / method / f"seed_{seed}"
            report_path = folder / "result.json"
            if report_path.exists():
                report = _json(report_path)
                if (report.get("cohort_feature_sha256") != audit["cohort_feature_sha256"]
                        or (training_seed is not None and report.get("training_random_seed") != training_seed)):
                    raise ValueError(f"{folder}: incompatible source cohort")
                print(f"[reuse] {model}/{method}/seed_{seed}", flush=True)
                continue
            if method == "sam":
                report = fit_ours_82(
                    records, config["training"]["torch_probe"], seed, folder, PROTOCOL,
                    training_seed=training_seed,
                )
                report["feature_set"] = audit["sam_definition"]
            else:
                folder.mkdir(parents=True, exist_ok=True)
                run_cfg = {**base_cfg, "seed": seed, "_strict_82_no_validation": True}
                if training_seed is not None:
                    run_cfg.update(initialization_seed=seed, training_seed=training_seed)
                report = _train_svar(
                    records,
                    folder,
                    run_cfg,
                    "cpu",
                    "real",
                )
                report["feature_set"] = audit["svar_definition"]
            report["cohort_feature_sha256"] = audit["cohort_feature_sha256"]
            report["source_counts"] = audit["counts"]
            if training_seed is not None:
                report.update(initialization_seed=seed,
                              training_random_seed=training_seed,
                              seed_scope="model_initialization_only")
            _save_json(report_path, report)
            print(f"[train] {model}/{method}/seed_{seed}", flush=True)


def _metric_row(model: str, method: str, seed: int, target: str, cohort: str,
                labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = classification_metrics(labels, scores, threshold)
    return {
        "model": model,
        "method": method.upper(),
        "seed": seed,
        "target": target,
        "cohort": cohort,
        "n": len(labels),
        "real_n": int(labels.sum()),
        "hall_n": int((labels == 0).sum()),
        "threshold": threshold,
        "auroc": metrics["auroc"],
        "real_aupr": metrics["real_aupr"],
        "hall_aupr": metrics["hallucination_aupr"],
        "real_f1": metrics["real"]["f1"],
        "hall_f1": metrics["hallucination"]["f1"],
        "hall_recall": metrics["hallucination"]["recall"],
        "accuracy": metrics["accuracy"],
        "pred_real_rate": float((scores >= threshold).mean()),
        "mean_real_score": float(scores.mean()),
    }


def evaluate(model: str, groups: dict, targets: dict, output: Path) -> list[dict]:
    rows = []
    for method in ("sam", "svar"):
        key = method
        for seed in SEEDS:
            folder = output / model / method / f"seed_{seed}"
            predict = _predict_qa if method == "sam" else predict_svar
            source_test = groups["test"]
            matrix = np.stack([row[key] for row in source_test]).astype(np.float32)
            labels = np.asarray([row["label"] for row in source_test], dtype=np.int64)
            scores, threshold = predict(folder, matrix)
            report = _json(folder / "result.json")
            native = (report["metrics"]["test"]["auroc"] if method == "sam"
                      else report["test_metrics"]["real_positive"]["auc"])
            replay = classification_metrics(labels, scores, threshold)["auroc"]
            if abs(native - replay) > 2e-5:
                raise ValueError(f"{model}/{method}/seed_{seed}: checkpoint replay failed")
            rows.append(_metric_row(model, method, seed, "clevr_exist_9k", "test",
                                    labels, scores, threshold))
            for dataset, target in targets.items():
                all_scores, source_threshold = predict(folder, target[key])
                if source_threshold != threshold:
                    raise ValueError("Source threshold changed across target datasets")
                for cohort, selected in (("all", np.ones(len(all_scores), dtype=bool)),
                                         ("test", target["splits"] == "test")):
                    if not selected.any():
                        raise ValueError(f"{model}/{dataset}: empty {cohort} cohort")
                    rows.append(_metric_row(
                        model, method, seed, dataset, cohort,
                        target["labels"][selected], all_scores[selected], threshold,
                    ))
            print(f"[evaluate] {model}/{method}/seed_{seed}", flush=True)
    return rows


def _fmt(rows: list[dict], field: str) -> str:
    values = [float(row[field]) for row in rows if row[field] is not None]
    if len(values) != len(SEEDS):
        return "—"
    array = 100 * np.asarray(values, dtype=np.float64)
    return f"{array.mean():.2f} ± {array.std():.2f}"


def write_outputs(model: str, groups: dict, audit: dict, targets: dict,
                  rows: list[dict], output: Path) -> None:
    folder = output / model
    _save_json(folder / "cohort_audit.json", {
        "source": audit,
        "target": {dataset: {
            "summary_path": target["summary_path"],
            "all": dict(Counter(target["labels"].tolist())),
            "test": dict(Counter(target["labels"][target["splits"] == "test"].tolist())),
        } for dataset, target in targets.items()},
    })
    with (folder / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _save_json(folder / "per_seed_results.json", {"rows": rows})
    lines = [
        f"# {model}: CLEVR Yes-only → POPE popular / AMBER existence", "",
        "CLEVR 原始图片级 8:2 划分；仅用来源 train 训练，来源 test 用于原域核对。种子 43/44/45。SAM 为物体词前 risk + 答案前 F_log；SVAR 为物体词前第 5–18 层。来源训练均值、标准差（SAM）和阈值原样迁移；目标不训练、不调参。", "",
        "目标 `all` 包含全部 Yes-only 样本；`test` 只包含目标预设图片级测试集。两者都是 CLEVR 之外的图像。AMBER existence 的 GT 全为 No，AUC/AUPR 不可定义。均值 ± 总体标准差，单位 %。", "",
        "| 方法 | 测试集 | 范围 | N (Real/Hall) | AUC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Hall Recall | Accuracy | 预测 Real 比例 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("clevr_exist_9k", "pope", "amber_discriminative"):
        for cohort in (("test",) if target == "clevr_exist_9k" else ("all", "test")):
            for method in ("SAM", "SVAR"):
                selected = [row for row in rows if row["target"] == target and row["cohort"] == cohort and row["method"] == method]
                if len(selected) != len(SEEDS):
                    raise ValueError(f"Missing {model}/{target}/{cohort}/{method} seeds")
                n, real, hall = selected[0]["n"], selected[0]["real_n"], selected[0]["hall_n"]
                if any((row["n"], row["real_n"], row["hall_n"]) != (n, real, hall) for row in selected):
                    raise ValueError("Target cohort changed across seeds")
                metrics = " | ".join(_fmt(selected, field) for field in (
                    "auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
                    "hall_recall", "accuracy", "pred_real_rate"))
                lines.append(f"| {method} | {target} | {cohort} | {n} ({real}/{hall}) | {metrics} |")
    lines.append("")
    (folder / "complete_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[complete] {folder / 'complete_results.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("llava_1_5_7b", "qwen3_vl_8b"))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    groups, audit = load_clevr(args.model, args.source_root)
    print(f"[source] {args.model} {audit['counts']} widths={audit['widths']}", flush=True)
    targets = load_targets(args.model, args.target_root, audit["widths"])
    train(args.model, groups, audit, args.output_root)
    rows = evaluate(args.model, groups, targets, args.output_root)
    write_outputs(args.model, groups, audit, targets, rows, args.output_root)


if __name__ == "__main__":
    main()
