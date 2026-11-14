#!/usr/bin/env python3
"""Split the existing QA outer-test cohort into image-disjoint val/test sets.

The original train cohort and every question key remain unchanged. This script
only writes a new split manifest; it never edits the 8:2 source artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import pickle
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.qa_baseline import qa_image_identity  # noqa: E402


MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
DATASETS = ("pope", "clevr_exist_9k", "amber_discriminative")
SOURCE_NAME = "LOGS-K32-TAU003-NATIVE6-FULL"
TARGET_NAME = "LOGS-K32-TAU003-NATIVE6-FULL-811"


def _load_rows(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid QA feature cache: {path}")
    return rows


def _exact_clevr_half(groups: dict[str, list[str]], seed: int) -> set[str]:
    """Select whole official-val images totaling exactly 900 questions."""
    identities = sorted(groups)
    random.Random(seed).shuffle(identities)
    target = sum(len(groups[identity]) for identity in identities) // 2
    if target != 900:
        raise ValueError(f"Expected 1,800 CLEVR outer-test questions, got {2*target}")
    parents: dict[int, tuple[int, str] | None] = {0: None}
    for identity in identities:
        width = len(groups[identity])
        for subtotal in sorted(parents, reverse=True):
            candidate = subtotal + width
            if candidate <= target and candidate not in parents:
                parents[candidate] = (subtotal, identity)
        if target in parents:
            break
    if target not in parents:
        raise ValueError("Cannot divide CLEVR outer-test questions by image")
    selected: set[str] = set()
    subtotal = target
    while subtotal:
        previous, identity = parents[subtotal]  # type: ignore[misc]
        selected.add(identity)
        subtotal = previous
    return selected


def prepare(dataset: str, *, seed: int = 42) -> Path:
    source_root = ROOT / "outputs" / "qa_benchmarks"
    first = source_root / MODELS[0] / SOURCE_NAME / dataset
    rows = _load_rows(first / "features.pkl")
    by_identity: dict[str, list[str]] = defaultdict(list)
    old_owner: dict[str, str] = {}
    keys: set[str] = set()
    for row in rows:
        identity = qa_image_identity(row)
        split = str(row["probe_split"])
        if split not in {"train", "test"}:
            raise ValueError(f"Expected old 8:2 split, got {split!r}")
        if identity in old_owner and old_owner[identity] != split:
            raise ValueError(f"Old split leaks image {identity}")
        old_owner[identity] = split
        key = str(row["key"])
        if key in keys:
            raise ValueError(f"Duplicate QA question key: {key}")
        keys.add(key)
        by_identity[identity].append(key)
    outer_test = {identity: by_identity[identity] for identity, split in old_owner.items() if split == "test"}
    if dataset == "clevr_exist_9k":
        validation = _exact_clevr_half(outer_test, seed)
    else:
        identities = sorted(outer_test)
        random.Random(seed).shuffle(identities)
        validation = set(identities[: len(identities) // 2])
    assignment = {
        identity: ("train" if old_owner[identity] == "train" else "val" if identity in validation else "test")
        for identity in old_owner
    }
    question_counts = Counter(assignment[qa_image_identity(row)] for row in rows)
    image_counts = Counter(assignment.values())
    expected = {
        "pope": ({"train": 7200, "val": 900, "test": 900}, {"train": 400, "val": 50, "test": 50}),
        "clevr_exist_9k": ({"train": 7200, "val": 900, "test": 900}, None),
        "amber_discriminative": (None, {"train": 803, "val": 100, "test": 101}),
    }[dataset]
    for actual, wanted in zip((question_counts, image_counts), expected):
        if wanted is not None and dict(actual) != wanted:
            raise ValueError(f"Unexpected {dataset} 8:1:1 counts: {dict(actual)} != {wanted}")

    key_digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    for model in MODELS[1:]:
        other = _load_rows(source_root / model / SOURCE_NAME / dataset / "features.pkl")
        if hashlib.sha256("\n".join(sorted(str(row["key"]) for row in other)).encode()).hexdigest() != key_digest:
            raise ValueError(f"{model}/{dataset} question keys differ")
        for row in other:
            identity = qa_image_identity(row)
            if identity not in assignment or str(row["probe_split"]) != old_owner[identity]:
                raise ValueError(f"{model}/{dataset} image cohort or old split differs")

    payload = {
        "protocol": "strict_outer_image_level_811_from_existing_outer82",
        "seed": seed,
        "source_output": SOURCE_NAME,
        "source_question_keys_sha256": key_digest,
        "selection": "seeded shuffle of outer-test physical images; CLEVR exact 900-question subset sum",
        "question_counts": {split: question_counts[split] for split in ("train", "val", "test")},
        "image_counts": {split: image_counts[split] for split in ("train", "val", "test")},
        "images": {split: sorted(identity for identity, owner in assignment.items() if owner == split) for split in ("train", "val", "test")},
        "question_keys": {split: sorted(str(row["key"]) for row in rows if assignment[qa_image_identity(row)] == split) for split in ("train", "val", "test")},
    }
    output = source_root / TARGET_NAME / "splits" / f"{dataset}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"{dataset}: questions={dict(question_counts)}, images={dict(image_counts)} -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        prepare(dataset, seed=args.seed)


if __name__ == "__main__":
    main()
