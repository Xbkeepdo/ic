#!/usr/bin/env python3
"""Prepare POPE-popular and AMBER-existence cohorts for positioned QA extraction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.qa_benchmark import atomic_write_json, atomic_write_jsonl, load_jsonl, sha256_file


AMBER_OBJECT = re.compile(r"Is there a (.+?) in this image[?]", re.IGNORECASE)
EXPECTED_COUNTS = {"pope": 3000, "amber_discriminative": 4924}


def select_rows(dataset: str, source_rows: list[dict]) -> list[dict]:
    if dataset == "pope":
        rows = [dict(row) for row in source_rows if row["source_split"] == "popular"]
        for row in rows:
            start, end = int(row["object_char_start"]), int(row["object_char_end"])
            if row["question"][start:end] != row["object_surface"]:
                raise ValueError(f"Invalid POPE object span: {row['key']}")
    elif dataset == "amber_discriminative":
        rows = []
        for original in source_rows:
            if original["source_split"] != "existence":
                continue
            row = dict(original)
            match = AMBER_OBJECT.fullmatch(row["question"])
            if match is None:
                raise ValueError(f"Unparseable AMBER existence object: {row['key']}")
            row.update(
                object_word=match.group(1).lower(),
                object_surface=match.group(1),
                object_char_start=match.start(1),
                object_char_end=match.end(1),
                object_span_status="found",
                object_span_protocol="amber_official_existence_query_surface_v1",
            )
            rows.append(row)
    else:
        raise ValueError(dataset)
    if len(rows) != EXPECTED_COUNTS[dataset]:
        raise ValueError(f"Unexpected {dataset} cohort size: {len(rows)}")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate {dataset} question keys")
    return rows


def select_yes_only(rows: list[dict], labels_path: Path) -> list[dict]:
    labels = load_jsonl(labels_path)
    by_key = {row["key"]: row for row in labels}
    if len(by_key) != len(labels):
        raise ValueError(f"Duplicate label keys in {labels_path}")
    if not {row["key"] for row in rows}.issubset(by_key):
        raise ValueError(f"Incomplete labels in {labels_path}")
    selected = []
    for row in rows:
        label = by_key[row["key"]]
        yes_only = label["object_hallucination_yes_only_label"]
        if yes_only is None:
            continue
        if label["prediction"] != "yes" or yes_only != int(row["gt_answer"] == "yes"):
            raise ValueError(f"Inconsistent Yes-only label: {row['key']}")
        selected.append(row)
    if not selected:
        raise ValueError(f"No model-Yes questions in {labels_path}")
    return selected


def prepare_one(
    dataset: str,
    source_root: Path,
    output_root: Path,
    labels_path: Path | None,
) -> None:
    source = source_root / dataset
    questions_path = source / "questions.jsonl"
    source_rows = load_jsonl(questions_path)
    rows = select_rows(dataset, source_rows) if labels_path is None else source_rows
    if labels_path is not None:
        rows = select_yes_only(rows, labels_path)
    split_path = source / "image_splits.json"
    with split_path.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    split_lookup = {
        int(image_id): split
        for split in ("train", "val", "test")
        for image_id in splits[split]
    }
    for row in rows:
        if row["probe_split"] != split_lookup[int(row["image_id"])]:
            raise ValueError(f"Image split mismatch: {row['key']}")
        if row.get("object_span_status") != "found":
            raise ValueError(f"Missing exact object span: {row['key']}")
        start, end = int(row["object_char_start"]), int(row["object_char_end"])
        if row["question"][start:end] != row["object_surface"]:
            raise ValueError(f"Object span mismatch: {row['key']}")
    destination = output_root / dataset
    atomic_write_jsonl(destination / "questions.jsonl", rows)
    atomic_write_json(destination / "image_splits.json", splits)
    atomic_write_json(
        destination / "manifest.json",
        {
            "cohort": "pope_popular_or_amber_existence_yes_only_v1",
            "dataset": dataset,
            "label_protocol": "object_hallucination_yes_only" if labels_path else None,
            "source_questions_sha256": sha256_file(questions_path),
            "labels_sha256": sha256_file(labels_path) if labels_path else None,
            "questions_sha256": sha256_file(destination / "questions.jsonl"),
            "counts": dict(Counter(row["probe_split"] for row in rows)),
            "num_questions": len(rows),
            "num_images": len({row["image_id"] for row in rows}),
        },
    )
    print(f"[prepare] {dataset}: {len(rows)} questions, {len({row['image_id'] for row in rows})} images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("pope", "amber_discriminative", "both"), default="both")
    parser.add_argument("--yes-only-labels", type=Path)
    args = parser.parse_args()
    if args.yes_only_labels and args.dataset == "both":
        parser.error("--yes-only-labels requires one --dataset")
    datasets = EXPECTED_COUNTS if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        prepare_one(dataset, args.source_root, args.output_root, args.yes_only_labels)


if __name__ == "__main__":
    main()
