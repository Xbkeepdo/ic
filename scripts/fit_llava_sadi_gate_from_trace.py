#!/usr/bin/env python3
"""Cross-validate a SAR/EMS gate on cached SADI object-token traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


EVIDENCE_NAMES = ("F_log", "relative_SAR")


def read_trace(directory, expected_images, layers):
    images = set()
    counts = Counter()
    by_cohort = Counter()
    samples = []
    for path in sorted(directory.glob("sadi_sar_ems_trace.shard*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = int(row["image_id"])
                if image_id in images or row["layers"] != layers:
                    raise ValueError(f"Duplicate image or incompatible layers in {path}: {image_id}")
                images.add(image_id)
                by_cohort[row["cohort"]] += 1
                counts["unaligned_mentions"] += len(row["unaligned_mentions"])
                seen_categories = set()
                for position in sorted(row["object_positions"], key=lambda entry: entry["step"]):
                    counts["aligned_positions"] += 1
                    mentions = position["mentions"]
                    names = {mention["canonical"] for mention in mentions}
                    labels = {mention["label"] for mention in mentions}
                    if not mentions or not labels <= {0, 1}:
                        raise ValueError(f"Missing or invalid CHAIR label for image {image_id}")
                    fresh = names - seen_categories
                    seen_categories.update(names)
                    if not fresh:
                        counts["repeated_positions"] += 1
                        continue
                    if len(labels) != 1:
                        counts["ambiguous_positions"] += 1
                        continue
                    samples.append((image_id, int(0 in labels), position))
    if len(images) != expected_images or not samples:
        raise ValueError(f"Expected {expected_images} complete trace images, found {len(images)}")
    groups, labels, positions = zip(*samples, strict=True)
    matrices = {}
    risks = np.asarray([row["risk"] for row in positions], dtype=np.float64)
    for name in EVIDENCE_NAMES:
        evidence = np.asarray([row[name] for row in positions], dtype=np.float64)
        if evidence.shape != risks.shape or risks.shape != (len(samples), len(layers)):
            raise ValueError(f"Incompatible {name} or EMS layer count")
        matrices[name] = np.concatenate((evidence, risks), axis=1)
        if not np.isfinite(matrices[name]).all():
            raise ValueError(f"Non-finite {name} or EMS in trace")
    token_ids = np.asarray([row["target_token_id"] for row in positions], dtype=np.int64)
    nouns = np.asarray([row["winner_in_noun_gate"] for row in positions], dtype=bool)
    gaps = np.asarray([row["runner_up_logit_gap"] for row in positions], dtype=np.float64)
    if not np.isfinite(gaps).all() or (gaps < 0).any():
        raise ValueError("Winner-to-runner-up logit gaps must be finite and nonnegative")
    stats = {"images": len(images), "images_by_cohort": dict(by_cohort), **counts}
    return np.asarray(groups), np.asarray(labels), token_ids, nouns, gaps, matrices, stats


def grouped_splits(labels, groups, folds, seed):
    count = min(folds, *(np.unique(groups[labels == label]).size for label in (0, 1)))
    if count < 2:
        raise ValueError("Need at least two images carrying each label for grouped calibration")
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    if any(np.unique(labels[train]).size != 2 for train, _ in splits):
        raise ValueError("A grouped fold has only one training class")
    return splits


def coefficients(values, labels):
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(values), labels)
    weights = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - weights @ scaler.mean_)
    return weights, intercept


def predict(values, weights, intercept):
    return 0.5 + 0.5 * np.tanh((values @ weights + intercept) / 2)


def threshold_for(labels, scores, eligible):
    candidates = np.flatnonzero(eligible)
    if not labels[candidates].any():
        return float(np.nextafter(1.0, 0.0))
    ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
    descending = scores[ranked]
    ends = np.flatnonzero(np.r_[descending[1:] != descending[:-1], True])
    true = np.cumsum(labels[ranked])[ends]
    f_half = 1.25 * true / (0.25 * labels.sum() + ends + 1)
    cut = descending[ends[f_half.argmax()]]
    # The deployed gate gives a positive penalty only when score > threshold.
    return float(np.nextafter(cut, 0.0))


def ranking(labels, scores):
    if np.unique(labels).size < 2:
        return {"hall_auroc": None, "hall_aupr": None}
    return {
        "hall_auroc": float(roc_auc_score(labels, scores)),
        "hall_aupr": float(average_precision_score(labels, scores)),
    }


def operating_metrics(labels, scores, eligible, thresholds, gaps, max_gap):
    positive = labels == 1
    predicts_hall = eligible & (scores > thresholds)
    true = int((predicts_hall & positive).sum())
    false = int((predicts_hall & ~positive).sum())
    can_cross = predicts_hall & (max_gap * (scores - thresholds) / (1 - thresholds) > gaps)
    return {
        "selected": true + false,
        "true_positive": true,
        "false_positive": false,
        "hall_precision": true / (true + false) if true + false else None,
        "hall_recall": true / int(positive.sum()),
        "hall_f0_5": 1.25 * true / (0.25 * int(positive.sum()) + true + false),
        "possible_winner_replacements": int(can_cross.sum()),
        "possible_hall_winner_replacements": int((can_cross & positive).sum()),
    }


def cross_validate(values, labels, groups, eligible, gaps, max_gap, splits, seed):
    scores = np.full(len(labels), np.nan)
    cross_fitted_thresholds = np.full(len(labels), np.nan)
    fold_details = []
    for index, (train, held) in enumerate(splits):
        inner_labels, inner_groups = labels[train], groups[train]
        inner = grouped_splits(inner_labels, inner_groups, len(splits), seed + index + 1)
        inner_scores = np.full(len(train), np.nan)
        for fit, check in inner:
            inner_scores[check] = predict(values[train[check]], *coefficients(values[train[fit]], labels[train[fit]]))
        threshold = threshold_for(inner_labels, inner_scores, eligible[train])
        scores[held] = predict(values[held], *coefficients(values[train], labels[train]))
        cross_fitted_thresholds[held] = threshold
        fold_details.append({
            "heldout_images": int(np.unique(groups[held]).size),
            "heldout_positions": int(len(held)),
            "heldout_hallucinations": int(labels[held].sum()),
            "threshold": threshold,
        })
    if not (np.isfinite(scores).all() and np.isfinite(cross_fitted_thresholds).all()):
        raise RuntimeError("Missing cross-validated scores or thresholds")
    return {
        **ranking(labels, scores),
        "eligible_ranking": ranking(labels[eligible], scores[eligible]),
        "cross_fitted_operating_point": operating_metrics(
            labels, scores, eligible, cross_fitted_thresholds, gaps, max_gap,
        ),
        "deployment_threshold_from_oof": threshold_for(labels, scores, eligible),
        "folds": fold_details,
    }


def run(args):
    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    layers = prior["layers"]
    if (layers != [12, 18, 22] or prior["evidence"] not in EVIDENCE_NAMES
            or len(prior["weights"]) != 2 * len(layers) or not 0 <= prior["threshold"] < 1):
        raise ValueError("Expected the existing three-layer calibrated SAR/EMS gate JSON")
    groups, labels, tokens, nouns, gaps, values, stats = read_trace(args.trace_dir, args.expected_images, layers)
    splits = grouped_splits(labels, groups, args.folds, args.seed)
    vocabulary = sorted({int(token) for token in prior["token_ids"]})
    known = nouns & np.isin(tokens, vocabulary)
    eligible = known & (gaps < args.max_gap)
    old_scores = predict(values[prior["evidence"]], np.asarray(prior["weights"]), prior["intercept"])
    if not np.isfinite(old_scores).all():
        raise ValueError("Non-finite prior gate scores")
    old_metrics = {
        **ranking(labels, old_scores),
        "eligible_ranking": ranking(labels[eligible], old_scores[eligible]),
        "threshold": prior["threshold"],
        "operating_point": operating_metrics(labels, old_scores, eligible, prior["threshold"], gaps, args.max_gap),
    }
    comparisons = {name: cross_validate(values[name], labels, groups, eligible, gaps, args.max_gap, splits, args.seed)
                   for name in EVIDENCE_NAMES}
    weights, intercept = coefficients(values[args.evidence], labels)
    threshold = comparisons[args.evidence]["deployment_threshold_from_oof"]
    diagnostics = {
        **stats,
        "sample_rule": "First occurrence of a CHAIR canonical category per image. Shared positions use one label.",
        "positions": int(len(labels)),
        "hallucinations": int(labels.sum()),
        "real_positions": int((labels == 0).sum()),
        "max_gap": args.max_gap,
        "coverage": {
            name: {"positions": int(mask.sum()), "hallucinations": int(labels[mask].sum())}
            for name, mask in (("noun_gate", nouns), ("prior_vocabulary_and_noun_gate", known), ("eligible", eligible))
        },
        "runner_up_gap_quantiles_10_50_90": {
            name: np.quantile(gaps[labels == label], [0.1, 0.5, 0.9]).tolist()
            for name, label in (("real", 0), ("hallucinated", 1))
        },
        "prior_path": str(args.prior),
        "prior": old_metrics,
        "grouped_oof": comparisons,
    }
    result = {
        "layers": layers,
        "evidence": args.evidence,
        "weights": weights.tolist(),
        "intercept": intercept,
        "threshold": threshold,
        "token_ids": vocabulary,
        "split_seed": args.seed,
        "trace": str(args.trace_dir),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": args.evidence, "threshold": threshold, "output": str(args.output),
                      "diagnostics": diagnostics}, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", choices=EVIDENCE_NAMES, default="F_log")
    parser.add_argument("--expected-images", type=int, default=140)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-gap", type=float, default=2.0, help="Maximum penalty and initial logit gap gate")
    args = parser.parse_args()
    if args.folds < 3 or args.expected_images < 1 or not np.isfinite(args.max_gap) or args.max_gap <= 0:
        parser.error("Need at least three folds, positive image count, and a positive finite max gap")
    if args.output.resolve() == args.prior.resolve():
        parser.error("Write the new gate to a path different from the prior")
    run(args)


if __name__ == "__main__":
    main()
