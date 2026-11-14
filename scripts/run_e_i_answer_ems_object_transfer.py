#!/usr/bin/env python3
"""Evaluate answer-position semantic mass/write and object-position EMS transfer."""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.qa_probe import QAProbe, classification_metrics
from scripts.evaluate_coco_clevr_log_s_transfer import _report
from scripts import run_object_position_transfer_comparison as previous


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/qa_benchmarks"
OLD = BASE / "object_position_svar_vs_sar_ems_transfer_82_811"
EXPERIMENT = "LOGS-K32-TAU003-EI-ANSWER"
OUT = BASE / "e_i_answer_ems_object_transfer_82_811"
TASKS = (
    ("correctness_amber_exist", "82", previous.AMBER, previous.CLEVR),
    ("yes_only_pope", "811", previous.POPE, previous.CLEVR),
)
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "hall_f1", "accuracy")


def _new_features(model: str, dataset: str) -> dict[str, dict]:
    path = BASE / model / EXPERIMENT / dataset / "features.pkl"
    with path.open("rb") as stream:
        rows = pickle.load(stream)
    return {str(row["key"]): row for row in rows}


def merged_master(model: str, dataset: str) -> tuple[list[dict], dict]:
    old_path = OLD / "prepared" / model / f"{dataset}.pkl"
    with old_path.open("rb") as stream:
        original = pickle.load(stream)
    extracted = _new_features(model, dataset)
    matched = []
    differences = []
    for row in original:
        if dataset == previous.POPE and row["prediction"] != "yes":
            continue
        if dataset == previous.AMBER and row["source_split"] != "existence":
            continue
        key = str(row["key"])
        new = extracted.get(key)
        if new is None:
            raise ValueError(f"Missing answer E/I features: {model}/{dataset}/{key}")
        block = new["positions"][previous.ANSWER]["dgst"]
        risk = np.asarray(block["logs_risk"], dtype=np.float32).reshape(-1)
        e_sum = np.asarray(block["e_sum"], dtype=np.float32).reshape(-1)
        i_sum = np.asarray(block["i_sum"], dtype=np.float32).reshape(-1)
        if not (risk.shape == e_sum.shape == i_sum.shape):
            raise ValueError(f"Different layer counts: {model}/{dataset}/{key}")
        if not all(np.isfinite(value).all() for value in (risk, e_sum, i_sum)):
            raise ValueError(f"Non-finite features: {model}/{dataset}/{key}")
        old_answer_risk = np.asarray(row["answer"][:risk.size], dtype=np.float32)
        differences.append(float(np.max(np.abs(risk - old_answer_risk))))
        if not np.allclose(risk, old_answer_risk, rtol=1e-5, atol=1e-4):
            raise ValueError(f"Old/new answer EMS mismatch: {model}/{dataset}/{key}")
        ems_object = np.asarray(row["mixed"][:risk.size], dtype=np.float32)
        matched.append({**row, "mixed": np.concatenate((ems_object, e_sum, i_sum)).astype(np.float32)})
    audit = {
        "model": model,
        "dataset": dataset,
        "matched_rows": len(matched),
        "feature_order": ["EMS_object_or_amber_answer", "E_answer_sum", "I_answer_sum"],
        "E_formula": "E_l = sum over visual tokens of p(answer_token | final_norm(hpre_visual_lm))",
        "I_formula": "I_l = sum over visual tokens of ||a_lm(answer_position)||_2",
        "max_abs_answer_ems_difference_against_original": max(differences),
        "amber_position_caveat": "AMBER has no exact object span; EMS uses answer position",
    }
    return matched, audit


def _format(rows: list[dict], field: str) -> str:
    values = 100 * np.asarray([row[field] for row in rows], dtype=np.float64)
    return f"{values.mean():.2f} ± {values.std():.2f}"


def _predict_logits(folder: Path, matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Use the same decision rule with raw logits to retain cross-domain ranks.

    On distant target domains sigmoid(float32) can round every score to 1.0,
    even when their logits differ substantially. AUROC/AUPR are rank metrics.
    """
    report = json.loads((folder / "result.json").read_text())
    saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    standardized = ((matrix - saved["mean"]) / saved["std"]).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError(f"Non-finite standardized input: {folder}")
    config = report["training_config"]
    network = QAProbe(matrix.shape[1], tuple(config["hidden_sizes"]),
                      float(config["dropout"]), batch_norm=bool(config["batch_norm"]))
    network.load_state_dict(saved["state_dict"])
    network.eval()
    with torch.inference_mode():
        logits = network(torch.from_numpy(standardized)).numpy().astype(np.float64)
    threshold = float(saved["threshold"])
    if not 0.0 < threshold < 1.0 or abs(threshold - float(report["threshold"])) > 1e-12:
        raise ValueError(f"Invalid saved threshold: {folder}")
    return logits, math.log(threshold / (1.0 - threshold))


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    previous.OUT = OUT
    masters = {}
    audits = []
    for model in previous.MODELS:
        for dataset in (previous.AMBER, previous.POPE, previous.CLEVR):
            masters[(model, dataset)], audit = merged_master(model, dataset)
            audits.append(audit)
    (OUT / "feature_alignment_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    rows = []
    cohort_audits = []
    original_cohorts = json.loads((OLD / "cohort_audits.json").read_text())
    for scheme, split_mode, left, right in TASKS:
        for model in previous.MODELS:
            cohorts = {}
            for dataset in (left, right):
                groups, cohort_audit = previous.cohort(
                    masters[(model, dataset)], model=model, dataset=dataset,
                    scheme=scheme, split_mode=split_mode,
                )
                references = [item for item in original_cohorts if all(
                    item[field] == cohort_audit[field]
                    for field in ("scheme", "split_mode", "model", "dataset")
                )]
                if len(references) != 1 or references[0]["question_keys_sha256"] != cohort_audit["question_keys_sha256"]:
                    raise ValueError(f"Target/source cohort changed: {scheme}/{split_mode}/{model}/{dataset}")
                cohort_audits.append(cohort_audit)
                cohorts[dataset] = groups
                previous.train(model, dataset, scheme, split_mode, "ours", groups, cohort_audit)
            for seed in previous.SEEDS:
                for source, target in ((left, left), (right, right), (right, left)):
                    folder = OUT / scheme / split_mode / model / source / "ours" / f"seed_{seed}"
                    matrix, labels = previous._matrix(cohorts[target]["test"], "ours", target)
                    scores, threshold = _predict_logits(folder, matrix)
                    if source == target:
                        saved = json.loads((folder / "result.json").read_text())
                        expected = saved["metrics"]["test"]["auroc"]
                        replay = classification_metrics(labels, scores, threshold)
                        actual = replay["auroc"]
                        # The saved training report uses sigmoid(float32),
                        # which can tie large positive logits at exactly 1.
                        if (abs(expected - actual) > 1e-3 or
                                abs(replay["real"]["f1"] - saved["metrics"]["test"]["real"]["f1"]) > 2e-5):
                            raise ValueError(f"Native checkpoint replay failed: {folder}")
                    result = _report(model, seed, source, target,
                                     "answer" if target == previous.AMBER else "object",
                                     labels, scores, threshold)
                    rows.append({"scheme": scheme, "split_mode": split_mode,
                                 "method": "EMS_object+E_answer+I_answer", **result})
            print(f"Finished {scheme}/{split_mode}/{model}", flush=True)
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "cohort_audits.json").write_text(json.dumps(cohort_audits, indent=2) + "\n")
    baseline = [item for item in csv.DictReader((OLD / "per_seed_results.csv").open())
                if item["method"] == "svar"]
    # Re-score the earlier SAR+EMS checkpoints from their raw logits as well.
    # The old sigmoid(float32) report ties some Qwen3 transfer scores at 1.0.
    for scheme, split_mode, target, source in TASKS:
        for model in previous.MODELS:
            with (OLD / "prepared" / model / f"{target}.pkl").open("rb") as handle:
                old_master = pickle.load(handle)
            old_groups, _ = previous.cohort(old_master, model=model, dataset=target,
                                             scheme=scheme, split_mode=split_mode)
            old_matrix, old_labels = previous._matrix(old_groups["test"], "ours", target)
            for seed in previous.SEEDS:
                for training_domain in (target, source):
                    folder = OLD / scheme / split_mode / model / training_domain / "ours" / f"seed_{seed}"
                    scores, threshold = _predict_logits(folder, old_matrix)
                    baseline.append({"scheme": scheme, "split_mode": split_mode,
                                     "method": "ours", **_report(
                                         model, seed, training_domain, target,
                                         "answer" if target == previous.AMBER else "object",
                                         old_labels, scores, threshold)})
    with (OUT / "comparator_raw_logit_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline[0]))
        writer.writeheader()
        writer.writerows(baseline)
    names = {previous.AMBER: "AMBER Existence", previous.POPE: "POPE all subsets"}
    model_names = {"llava_1_5_7b": "LLaVA-1.5-7B", "qwen3_vl_8b": "Qwen3-VL-8B"}
    lines = [
        "# Answer-position E/I + object-position EMS transfer",
        "",
        "Features: $E_l=\\sum_m p_{l,m}$ and $I_l=\\sum_m\\|a_{l,m}\\|_2$ at the answer-prediction position; EMS at the queried-object position. AMBER has no exact object span, so EMS uses the answer position there. Input order is EMS, E, I. Same 8:2 and 8:1:1 image splits, label protocols, training settings, seeds 43/44/45, source-fitted normalization and source thresholds as the previous SAM/SVAR comparison. No target adaptation. All values are mean ± population SD across seeds, in percent. Evaluation uses raw detector logits for rank metrics to avoid sigmoid(float32) saturation under domain shift; the saved source threshold is transformed by logit, so the classification decision is unchanged.",
        "",
        "| Target and split | Model | Method | In-domain AUROC | In-domain Real AUPR | CLEVR → target AUROC | CLEVR → target Real AUPR |",
        "|:--|:--|:--|--:|--:|--:|--:|",
    ]
    for scheme, split_mode, target, source in TASKS:
        for model in previous.MODELS:
            for method, source_rows in (("SVAR", baseline), ("SAR+EMS", baseline), ("EMS+E+I", rows)):
                method_key = "svar" if method == "SVAR" else "ours" if method == "SAR+EMS" else "EMS_object+E_answer+I_answer"
                subset = [row for row in source_rows if row["scheme"] == scheme and row["split_mode"] == split_mode
                          and row["model"] == model and row["method"] == method_key]
                native = [row for row in subset if row["source"] == target and row["target"] == target]
                transfer = [row for row in subset if row["source"] == source and row["target"] == target]
                if len(native) != 3 or len(transfer) != 3:
                    raise ValueError(f"Missing rows for {scheme}/{split_mode}/{model}/{method}")
                lines.append(f"| {names[target]} · {split_mode} | {model_names[model]} | {method} | "
                             f"{_format(native,'auroc')} | {_format(native,'real_aupr')} | "
                             f"{_format(transfer,'auroc')} | {_format(transfer,'real_aupr')} |")
    lines += [
        "",
        "Target test counts (Real/Hall): AMBER LLaVA 892/95, AMBER Qwen3 909/78; POPE LLaVA 369/33, POPE Qwen3 378/28. Cohort hashes match the prior transfer comparison exactly.",
        "",
        "Caution: AMBER answer-correctness is largely answer polarity because all Existence ground truths are No. POPE Yes-only has few Hall cases in the fixed test split. Qwen3 CLEVR → AMBER has near-complete rank inversion (AUROC 2.03%); its answer-position E magnitude shifts strongly across domains and labels. This is a descriptive diagnostic, not evidence that E alone caused the inversion. The previous SAR+EMS checkpoints were re-scored with raw logits for identical rank-metric handling; SVAR probabilities showed no saturation change. See `feature_alignment_audits.json`, `cohort_audits.json`, `comparator_raw_logit_results.csv`, and `per_seed_results.csv` for exact checks.", "",
    ]
    (OUT / "complete_results.md").write_text("\n".join(lines))
    compact = [
        "# Cross-dataset transfer (AUROC / Real AUPR)", "",
        "Values are percent means over seeds 43/44/45. Same target test questions are used for all methods. No target-data adaptation. See `complete_results.md` for standard deviations and protocol details.", "",
        "| Target · split | Model | Method | In-domain AUROC / Real AUPR | CLEVR → target AUROC / Real AUPR |",
        "|:--|:--|:--|--:|--:|",
    ]
    for scheme, split_mode, target, source in TASKS:
        for model in previous.MODELS:
            for method, source_rows in (("SVAR", baseline), ("SAR+EMS", baseline), ("EMS+E+I", rows)):
                key = "svar" if method == "SVAR" else "ours" if method == "SAR+EMS" else "EMS_object+E_answer+I_answer"
                selected = [item for item in source_rows if item["scheme"] == scheme and item["split_mode"] == split_mode
                            and item["model"] == model and item["method"] == key]
                parts = []
                for training_domain in (target, source):
                    values = [item for item in selected if item["source"] == training_domain and item["target"] == target]
                    if len(values) != len(previous.SEEDS):
                        raise ValueError(f"Incomplete compact row: {scheme}/{model}/{method}/{training_domain}")
                    parts.append(" / ".join(f"{100*np.mean([float(item[field]) for item in values]):.2f}"
                                            for field in ("auroc", "real_aupr")))
                compact.append(f"| {names[target]} · {split_mode} | {model_names[model]} | {method} | {parts[0]} | {parts[1]} |")
    compact += ["", "AMBER: answer-correctness labels; EMS also uses the answer position because no exact object token is available. POPE: predicted Yes only, with GT Yes as Real. Real AUPR uses Real as positive. For Qwen3 CLEVR → AMBER, the new feature combination reverses the target ranking (AUROC 2.03%).", ""]
    (OUT / "rebuttal_table_auc_real_aupr.md").write_text("\n".join(compact))
    print(OUT / "complete_results.md", flush=True)


if __name__ == "__main__":
    main()
