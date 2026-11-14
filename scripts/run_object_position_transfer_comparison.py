#!/usr/bin/env python3
"""Compare object-position SVAR and SAR+EMS on AMBER/CLEVR and POPE/CLEVR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from detection.qa_probe import QAProbe, choose_real_f1_threshold, classification_metrics, feature_vector
from features.baseline import baseline_config, svar_training_vector
from features.qa_baseline import qa_image_identity
from scripts.evaluate_coco_clevr_log_s_transfer import _predict_qa, _report
from scripts.run_amber_exist_clevr_svar_transfer import predict as predict_svar
from scripts.train_baselines import _train_svar
from scripts.train_qa_811_cached import _fit_method
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/qa_benchmarks"
OUT = BASE / "object_position_svar_vs_sar_ems_transfer_82_811"
CACHE = "LOGS-K32-TAU003-NATIVE6-FULL"
INNER = "LOGS-K32-TAU003-NATIVE6-FULL-811"
ANSWER = "prompt_last_token"
OBJECT = "question_object_pre_token"
FEATURE = f"logs_risk_f_log@{ANSWER}"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
AMBER = "amber_discriminative"
POPE = "pope"
CLEVR = "clevr_exist_9k"
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")
SCHEMES = {
    "correctness_amber_exist": (AMBER, CLEVR, "answer_correctness_all"),
    "yes_only_pope": (POPE, CLEVR, "object_hallucination_yes_only"),
}


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _valid_vector(record: dict, position: str) -> np.ndarray:
    vector = np.asarray(feature_vector(record, f"logs_risk_f_log@{position}"), dtype=np.float32).reshape(-1)
    if not vector.size or vector.size % 2 or not np.isfinite(vector).all():
        raise ValueError("empty, odd-width, or non-finite Log-S vector")
    return vector


def build_master(model: str, dataset: str) -> list[dict]:
    target = OUT / "prepared" / model / f"{dataset}.pkl"
    if target.exists():
        return _load_pickle(target)
    source = BASE / model / CACHE / dataset
    labels = {str(row["key"]): row for row in (json.loads(line) for line in (source / "labels.jsonl").open())}
    method = {}
    for record in _load_pickle(source / "features.pkl"):
        key = str(record["key"])
        if key not in labels:
            continue
        try:
            answer = _valid_vector(record, ANSWER)
            obj = _valid_vector(record, OBJECT) if dataset != AMBER else None
        except (KeyError, TypeError, ValueError):
            continue
        if obj is not None and obj.shape != answer.shape:
            continue
        method[key] = {"answer": answer, "object": obj,
                       "image_id": int(record["image_id"]),
                       "source_split": str(record["source_split"])}
    baseline_path = source / "baseline" / "answer_correctness_all" / "features.pkl"
    baseline = _load_pickle(baseline_path)
    rows = []
    for record in baseline:
        answer_record = record.get("positions", {}).get(ANSWER, record)
        key = str(answer_record["key"])
        current = method.get(key)
        if current is None:
            continue
        object_record = record.get("positions", {}).get(OBJECT) if dataset != AMBER else None
        if dataset != AMBER and object_record is None:
            continue
        label = labels[key]
        if (int(answer_record["label"]) != int(label["answer_correctness_all_label"])
                or int(answer_record["image_id"]) != current["image_id"]
                or str(answer_record["source_split"]) != current["source_split"]):
            raise ValueError(f"Metadata/label mismatch for {model}/{dataset}/{key}")
        try:
            svar_answer = svar_training_vector(answer_record["baselines"]["svar"], layer_start=5, layer_end=19)
            svar_object = (svar_training_vector(object_record["baselines"]["svar"], layer_start=5, layer_end=19)
                           if object_record is not None else None)
        except (KeyError, TypeError, ValueError):
            continue
        if (not np.isfinite(svar_answer).all() or
                (svar_object is not None and (svar_object.shape != svar_answer.shape or not np.isfinite(svar_object).all()))):
            continue
        answer = current["answer"]
        half = answer.size // 2
        mixed = np.concatenate((current["object"][:half], answer[half:])) if dataset != AMBER else answer
        rows.append({
            "key": key, "dataset": dataset,
            "image_id": current["image_id"], "source_split": current["source_split"],
            "label_correctness": int(label["answer_correctness_all_label"]),
            "label_yes_only": label["object_hallucination_yes_only_label"],
            "prediction": str(label["prediction"]).lower(),
            "gt_answer": str(label["gt_answer"]).lower(),
            "answer": answer, "mixed": mixed,
            "svar_answer": svar_answer, "svar_object": svar_object,
        })
    if len({row["key"] for row in rows}) != len(rows) or not rows:
        raise ValueError(f"Duplicate or empty matched cohort: {model}/{dataset}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Prepared {model}/{dataset}: matched={len(rows)}, method={len(method)}, labels={len(labels)}", flush=True)
    return rows


def _split_owner(model: str, dataset: str, split_mode: str) -> dict[str, str]:
    if split_mode == "811":
        path = BASE / INNER / "splits" / f"{dataset}.json"
        manifest = json.loads(path.read_text())
        if manifest["protocol"] != "strict_outer_image_level_811_from_existing_outer82":
            raise ValueError(f"Unexpected 8:1:1 split: {path}")
        return {str(identity): part for part in ("train", "val", "test") for identity in manifest["images"][part]}
    path = BASE / model / CACHE / dataset / "baseline" / "answer_correctness_all" / "qa_probe_splits.json"
    if not path.exists():
        path = path.with_name("probe_splits.prompt_last_token.json")
    manifest = json.loads(path.read_text())
    if manifest["label_protocol"] != "answer_correctness_all" or manifest["question_counts"]["val"] != 0:
        raise ValueError(f"Unexpected strict 8:2 split: {path}")
    return {str(key): part for part in ("train", "test") for key in manifest[part]}


def cohort(master: list[dict], *, model: str, dataset: str, scheme: str, split_mode: str) -> tuple[dict[str, list[dict]], dict]:
    if scheme == "correctness_amber_exist" and dataset == AMBER:
        selected = [row for row in master if row["source_split"] == "existence"]
    elif scheme == "yes_only_pope":
        selected = [row for row in master if row["prediction"] == "yes"]
    else:
        selected = master
    owner = _split_owner(model, dataset, split_mode)
    groups = {part: [] for part in ("train", "val", "test")}
    image_owner = {}
    for row in selected:
        identity = qa_image_identity(row)
        split = owner[identity if split_mode == "811" else row["key"]]
        previous = image_owner.setdefault(identity, split)
        if previous != split:
            raise ValueError(f"Image leakage: {model}/{dataset}/{identity}")
        label = row["label_correctness"] if scheme == "correctness_amber_exist" else row["label_yes_only"]
        if label not in (0, 1):
            raise ValueError(f"Unexpected label in {model}/{dataset}/{row['key']}")
        if scheme == "yes_only_pope" and label != int(row["gt_answer"] == "yes"):
            raise ValueError(f"GT/yes-only label mismatch: {row['key']}")
        groups[split].append({**row, "label": int(label), "probe_split": split})
    active = ("train", "val", "test") if split_mode == "811" else ("train", "test")
    counts = {part: {"real": sum(row["label"] == 1 for row in groups[part]),
                     "hall": sum(row["label"] == 0 for row in groups[part])} for part in active}
    if any(min(counts[part].values()) == 0 for part in active):
        raise ValueError(f"Single-class cohort: {model}/{dataset}/{scheme}/{split_mode}: {counts}")
    keys = sorted(row["key"] for part in active for row in groups[part])
    audit = {"model": model, "dataset": dataset, "scheme": scheme, "split_mode": split_mode,
             "label_protocol": SCHEMES[scheme][2], "position": "answer" if dataset == AMBER else "object",
             "n": len(keys), "question_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
             "split_class_counts": counts,
             "split_image_counts": {part: len({qa_image_identity(row) for row in groups[part]}) for part in active}}
    return groups, audit


def training_rows(groups: dict[str, list[dict]], method: str, *, protocol: str, dataset: str) -> dict[str, list[dict]]:
    result = {part: [] for part in groups}
    for part, rows in groups.items():
        for row in rows:
            common = {"key": row["key"], "dataset": dataset, "source_split": row["source_split"],
                      "image_id": row["image_id"], "label": row["label"], "probe_split": part,
                      "qa_label_protocol": protocol, "object_hallucination_yes_only_label": row["label"] if protocol == "object_hallucination_yes_only" else None}
            if method == "ours":
                result[part].append({**common, "positions": {ANSWER: {"dgst": {"logs_risk_f_log": row["mixed"]}}}})
            else:
                svar = row["svar_answer"] if dataset == AMBER else row["svar_object"]
                result[part].append({**common, "baseline_schema_version": "1.0", "response_token_idx": 0,
                                     "baselines": {"svar": {"vector": svar}}})
    return result


def fit_ours_82(rows: dict[str, list[dict]], cfg: dict, seed: int, output: Path, protocol: str) -> dict:
    X_train = np.stack([row["positions"][ANSWER]["dgst"]["logs_risk_f_log"] for row in rows["train"]]).astype(np.float32)
    y_train = np.asarray([row["label"] for row in rows["train"]], dtype=np.int64)
    X_test = np.stack([row["positions"][ANSWER]["dgst"]["logs_risk_f_log"] for row in rows["test"]]).astype(np.float32)
    y_test = np.asarray([row["label"] for row in rows["test"]], dtype=np.int64)
    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(X_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-12)
    train = ((X_train - mean) / std).astype(np.float32)
    test = ((X_test - mean) / std).astype(np.float32)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = QAProbe(X_train.shape[1], tuple(cfg["hidden_sizes"]), float(cfg["dropout"]), batch_norm=bool(cfg["batch_norm"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(cfg["lr_factor"]), patience=int(cfg["lr_patience"]))
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train), torch.from_numpy(y_train.astype(np.float32))),
                        batch_size=int(cfg["batch_size"]), shuffle=True,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    best_loss, best_state, stale, best_epoch, history = math.inf, None, 0, 0, []
    for epoch in range(int(cfg["max_epochs"])):
        model.train(); total, count = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite method training loss: seed {seed}")
            loss.backward(); optimizer.step()
            total += float(loss.item()) * len(y); count += len(y)
        train_loss = total / count
        improved = train_loss < best_loss
        if improved:
            best_loss, stale, best_epoch = train_loss, 0, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step(train_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "is_best": improved})
        if stale >= int(cfg["early_stopping_patience"]):
            break
    if best_state is None:
        raise ValueError("No method checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        train_score = torch.sigmoid(model(torch.from_numpy(train))).numpy()
        test_score = torch.sigmoid(model(torch.from_numpy(test))).numpy()
    threshold, _ = choose_real_f1_threshold(y_train, train_score)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "threshold": threshold}, output / "checkpoint.pt")
    report = {"feature_set": FEATURE, "label_protocol": protocol, "seed": seed,
              "split_protocol": "strict_outer_image_level_82", "checkpoint_selection": "minimum_train_loss",
              "threshold_selection": "train_real_f1", "threshold": threshold, "best_epoch": best_epoch,
              "epochs_completed": len(history), "training_config": {key: cfg[key] for key in
              ("hidden_sizes", "dropout", "batch_norm", "learning_rate", "weight_decay", "batch_size", "max_epochs", "lr_factor", "lr_patience", "early_stopping_patience")},
              "counts": {"train": len(y_train), "val": 0, "test": len(y_test)},
              "metrics": {"train": classification_metrics(y_train, train_score, threshold),
                          "test": classification_metrics(y_test, test_score, threshold)},
              "history": history}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def train(model: str, dataset: str, scheme: str, split_mode: str, method: str,
          rows: dict[str, list[dict]], audit: dict) -> None:
    cfg_file = "model_configs_qa_logs_k32_tau003_amber.yaml" if dataset == AMBER else "model_configs_qa_logs_k32_tau003.yaml"
    config = load_config(str(ROOT / "configs" / cfg_file))
    protocol = SCHEMES[scheme][2]
    folder = OUT / scheme / split_mode / model / dataset / method
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cohort_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    prepared = training_rows(rows, method, protocol=protocol, dataset=dataset)
    for seed in SEEDS:
        run = folder / f"seed_{seed}"
        report_path = run / "result.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("cohort_sha256") != audit["question_keys_sha256"] or report.get("label_protocol") != protocol:
                raise ValueError(f"Incompatible resumed result: {report_path}")
            print(f"Reuse {scheme}/{split_mode}/{model}/{dataset}/{method}/seed{seed}", flush=True)
            continue
        if method == "ours":
            if split_mode == "811":
                flat = [row for part in ("train", "val", "test") for row in prepared[part]]
                report = _fit_method(flat, model_name=model, dataset_name=dataset, feature_set=FEATURE,
                                     protocol=protocol, seed=seed, cfg=config["training"]["torch_probe"],
                                     device="cpu", output=run)
            else:
                report = fit_ours_82(prepared, config["training"]["torch_probe"], seed, run, protocol)
        else:
            run.mkdir(parents=True, exist_ok=True)
            base_cfg = {**baseline_config(config), "seed": seed, "_strict_82_no_validation": split_mode == "82"}
            split_rows = {**prepared, "val": prepared["train"]} if split_mode == "82" else prepared
            report = _train_svar(split_rows, run, base_cfg, "cpu", "real")
        report["cohort_sha256"] = audit["question_keys_sha256"]
        report["label_protocol"] = protocol
        report["method"] = method
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        auc = report["metrics"]["test"]["auroc"] if method == "ours" else report["test_metrics"]["real_positive"]["auc"]
        print(f"Done {scheme}/{split_mode}/{model}/{dataset}/{method}/seed{seed}: AUROC={auc:.4f}", flush=True)


def _matrix(rows: list[dict], method: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    block = "mixed" if method == "ours" else "svar_answer" if dataset == AMBER else "svar_object"
    matrix = np.stack([row[block] for row in rows]).astype(np.float32)
    label = np.asarray([row["label"] for row in rows], dtype=np.int64)
    return matrix, label


def evaluate(scheme: str, split_mode: str, model: str, left: str, right: str,
             cohorts: dict[str, tuple[dict[str, list[dict]], dict]]) -> list[dict]:
    targets = {(dataset, method): _matrix(cohorts[dataset][0]["test"], method, dataset)
               for dataset in (left, right) for method in ("ours", "svar")}
    results = []
    for method in ("ours", "svar"):
        for seed in SEEDS:
            for source in (left, right):
                folder = OUT / scheme / split_mode / model / source / method / f"seed_{seed}"
                for target in (left, right):
                    matrix, labels = targets[(target, method)]
                    score, threshold = (_predict_qa(folder, matrix) if method == "ours" else predict_svar(folder, matrix))
                    if source == target:
                        saved = json.loads((folder / "result.json").read_text())
                        native_auc = saved["metrics"]["test"]["auroc"] if method == "ours" else saved["test_metrics"]["real_positive"]["auc"]
                        native_f1 = saved["metrics"]["test"]["real"]["f1"] if method == "ours" else saved["test_metrics"]["real_positive"]["f1"]
                        actual = classification_metrics(labels, score, threshold)
                        if abs(actual["auroc"] - native_auc) > 2e-5 or abs(actual["real"]["f1"] - native_f1) > 2e-5:
                            raise ValueError(f"Native checkpoint replay failed: {folder}")
                    results.append({"scheme": scheme, "split_mode": split_mode, "method": method,
                                    **_report(model, seed, source, target,
                                              "answer" if target == AMBER else "object", labels, score, threshold)})
    return results


def summarize(rows: list[dict]) -> None:
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = [
        "# 物体位置 SVAR 与 SAR+EMS 跨数据集迁移对比", "",
        "两组互相独立的标签任务：① `answer_correctness_all`：AMBER existence ↔ CLEVR；② 只保留回答 Yes，GT=Yes 且回答 Yes 为 Real，GT=No 且回答 Yes 为 Hall：POPE ↔ CLEVR。后者用 POPE，因为 AMBER existence 的 GT 全为 No，其 Yes 子集只有 Hall，不能做双向二分类迁移。", "",
        "特征位置：AMBER 没有可靠的 question-object 位置，二者均用答案预测前的 prompt-last；POPE/CLEVR 中，SVAR 用 question-object-pre 位置，本文方法用 SAR(prompt-last)+EMS(question-object-pre)。SVAR 取层 5–18、448 维、248 宽原生单隐层 MLP；本文方法保持原 Log-S 单隐层配置。来源阈值随来源模型直接迁移，目标不重训或调阈值。", "",
        "8:1:1 采用来源验证集选权重及 Real-F1 阈值；8:2 采用原始固定图片划分、无验证集，并只在来源训练集选 Real-F1 阈值。所有数字为种子 43/44/45 的均值 ± 总体标准差，单位 %。", "",
    ]
    def fmt(group: list[dict], metric: str) -> str:
        values = 100 * np.asarray([row[metric] for row in group], dtype=np.float64)
        return f"{values.mean():.2f} ± {values.std():.2f}"
    display = {AMBER: "AMBER existence", POPE: "POPE Yes-only", CLEVR: "CLEVR"}
    for scheme, (left, right, _) in SCHEMES.items():
        lines.extend([f"## {scheme}", ""])
        for split_mode in ("82", "811"):
            lines.extend([f"### {split_mode}", "",
                          "| 模型 | 方法 | 训练 → 测试 | 测试 N (Real/Hall) | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
                          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
            for model in MODELS:
                for source, target in ((left, left), (right, right), (left, right), (right, left)):
                    for method in ("svar", "ours"):
                        selected = [row for row in rows if (row["scheme"], row["split_mode"], row["model"], row["source"], row["target"], row["method"]) ==
                                    (scheme, split_mode, model, source, target, method)]
                        if len(selected) != len(SEEDS):
                            raise ValueError(f"Missing results: {scheme}/{split_mode}/{model}/{source}/{target}/{method}")
                        count = {(row["n"], row["real_n"], row["hall_n"]) for row in selected}
                        if len(count) != 1:
                            raise ValueError("Changing cohort across seeds")
                        n, real, hall = next(iter(count))
                        lines.append(f"| {model} | {'SVAR' if method == 'svar' else 'SAR+EMS'} | {display[source]} → {display[target]} | {n} ({real}/{hall}) | " + " | ".join(fmt(selected, metric) for metric in METRICS) + " |")
            lines.append("")
    lines.extend(["## 注意", "", "两组协议的标签含义不同，不能混在一张排名表里。Yes-only 只评估回答 Yes 的问题；Real/Hall AUPR 应结合目标集正类占比。对完整 CLEVR 的迁移比按 GT 筛选的子集更接近无标签部署。逐种子和来源阈值见 `per_seed_results.csv`；样本审计见 `cohort_audits.json`。", ""])
    (OUT / "complete_results.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    master = {(model, dataset): build_master(model, dataset)
              for model in MODELS for dataset in (AMBER, POPE, CLEVR)}
    if args.prepare_only:
        return
    all_results, all_audits = [], []
    for scheme, (left, right, _) in SCHEMES.items():
        for split_mode in ("82", "811"):
            for model in MODELS:
                cohorts = {}
                for dataset in (left, right):
                    groups, audit = cohort(master[(model, dataset)], model=model, dataset=dataset,
                                           scheme=scheme, split_mode=split_mode)
                    cohorts[dataset] = groups, audit
                    all_audits.append(audit)
                    for method in ("svar", "ours"):
                        train(model, dataset, scheme, split_mode, method, groups, audit)
                all_results.extend(evaluate(scheme, split_mode, model, left, right, cohorts))
                print(f"Evaluated {scheme}/{split_mode}/{model}", flush=True)
    (OUT / "cohort_audits.json").write_text(json.dumps(all_audits, indent=2) + "\n")
    summarize(all_results)
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
