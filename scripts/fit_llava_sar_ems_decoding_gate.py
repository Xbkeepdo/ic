#!/usr/bin/env python3
"""Fit a small generation gate from audited COCO SAR and EMS features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    count = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True]) + 1
    correct = np.cumsum(labels[order])[count - 1]
    precision = correct / count
    recall = correct / labels.sum()
    f_half = 1.25 * precision * recall / (0.25 * precision + recall).clip(min=1e-12)
    selected = int(f_half.argmax())
    return {
        "threshold": float(sorted_scores[count[selected] - 1]),
        "precision": float(precision[selected]), "recall": float(recall[selected]),
        "predicted_hallucinations": int(count[selected]), "f_half": float(f_half[selected]),
    }


def fit_gate(cache: Path, source_split: Path, cohort: Path, output: Path) -> None:
    original = json.loads(source_split.read_text())
    pool = sorted({int(image_id) for group in original.values() for image_id in group})
    if len(pool) != 4000:
        raise ValueError("Expected 4000 COCO image IDs to restore the seed-43 image split")
    shuffled = np.random.default_rng(43).permutation(pool)
    manifest = json.loads(cohort.read_text())
    evaluation_ids = set(manifest["pilot"]) | set(manifest["confirmation"])
    if not evaluation_ids.issubset(set(shuffled[3600:])):
        raise ValueError("The generation cohorts overlap the restored training or validation split")

    layers = [12, 18, 22]
    with np.load(cache, allow_pickle=False) as records:
        image_ids = records["image_id"]
        if not np.isin(image_ids, pool).all():
            raise ValueError("Feature IDs outside the COCO4000 image pool")
        token_ids = records["target_token_id"]
        labels = (records["label"] == 0).astype(np.int8)
        risk = records["risk"][:, layers]
        sar = records["F_log"][:, layers]
        writes = records["I"][:, layers]
    train = np.isin(image_ids, shuffled[:3200])
    validation = np.isin(image_ids, shuffled[3200:3600])
    other = ~(train | validation)
    relative_sar = (sar - writes) / writes.clip(min=1e-12)

    candidates = {}
    for evidence_name, evidence in (("F_log", sar), ("relative_SAR", relative_sar)):
        values = np.concatenate((evidence, risk), axis=1).astype(np.float64)
        scaler = StandardScaler().fit(values[train])
        model = LogisticRegression(max_iter=1000, random_state=43)
        model.fit(scaler.transform(values[train]), labels[train])
        validation_score = model.predict_proba(scaler.transform(values[validation]))[:, 1]
        test_score = model.predict_proba(scaler.transform(values[other]))[:, 1]
        threshold = choose_threshold(labels[validation], validation_score)
        weight = model.coef_[0] / scaler.scale_
        candidates[evidence_name] = {
            "weights": weight.tolist(),
            "intercept": float(model.intercept_[0] - weight @ scaler.mean_),
            "validation": dict(threshold, auroc=float(roc_auc_score(labels[validation], validation_score)),
                               hall_aupr=float(average_precision_score(labels[validation], validation_score))),
            "test": {"auroc": float(roc_auc_score(labels[other], test_score)),
                     "hall_aupr": float(average_precision_score(labels[other], test_score))},
        }

    def validation_rank(name: str) -> tuple:
        result = candidates[name]["validation"]
        return result["f_half"], result["hall_aupr"]

    evidence_name = max(candidates, key=validation_rank)
    chosen = candidates[evidence_name]
    result = {
        "layers": layers, "evidence": evidence_name,
        "weights": chosen["weights"], "intercept": chosen["intercept"],
        "threshold": chosen["validation"]["threshold"],
        "token_ids": np.unique(token_ids[train]).tolist(),
        "feature_cache": str(cache), "split_seed": 43,
        "tokens_per_split": [int(mask.sum()) for mask in (train, validation, other)],
        "comparisons": {name: {"validation": trial["validation"], "test": trial["test"]}
                        for name, trial in candidates.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("evidence", "threshold", "tokens_per_split", "comparisons")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_gate(args.feature_cache, args.source_split, args.cohort, args.output)


if __name__ == "__main__":
    main()
