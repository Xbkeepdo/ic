#!/usr/bin/env python3
"""Pack existing QA DHCP payloads into one mmap-friendly array per model.

This is a storage conversion only: every float16 attention value, sample key,
label, and existing train/test assignment is copied without recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import tempfile

import numpy as np


POSITION = "prompt_last_token"
PROTOCOL = "answer_correctness_all"
RUN = "LOGS-K32-TAU003-NATIVE6-FULL"
MODELS = (
    "llava_1_5_7b",
    "qwen2_5_vl_7b",
    "internvl_2_5_8b",
    "qwen3_vl_8b",
)
ARRAY_NAME = "dhcp_prompt_last_fp16.npy"
INDEX_NAME = "dhcp_prompt_last_index.json"


def _payload(row: dict) -> dict:
    position = (row.get("positions") or {}).get(POSITION)
    if position is None:
        if row.get("baseline_position_protocol") != POSITION:
            raise ValueError(f"Missing {POSITION} for {row.get('key')}")
        position = row
    return position["baselines"]["dhcp"]


def _attention(payload: dict, shard_root: Path) -> np.ndarray:
    if "attention" in payload:
        value = np.asarray(payload["attention"])
    else:
        ref = payload.get("shard_reference", payload)
        path = (shard_root / str(ref["shard"])).resolve()
        if shard_root != path and shard_root not in path.parents:
            raise ValueError(f"DHCP shard escapes root: {ref['shard']!r}")
        shard = np.load(path, mmap_mode="r", allow_pickle=False)
        index = int(ref["index"])
        if index < 0 or index >= len(shard):
            raise IndexError(f"Invalid DHCP shard index: {index}")
        value = np.asarray(shard[index])
        if tuple(value.shape) != tuple(ref["shape"]):
            raise ValueError(f"DHCP shape mismatch in {path}: {value.shape}")
    if value.dtype != np.float16 or not np.isfinite(value).all():
        raise ValueError("DHCP attention must be finite float16")
    return value


def pack_model(output_root: Path, model: str) -> dict:
    baseline_dir = output_root / model / RUN / "clevr_exist_9k" / "baseline" / PROTOCOL
    source = baseline_dir / "features.pkl"
    destination = baseline_dir / ARRAY_NAME
    index_path = baseline_dir / INDEX_NAME
    source_stat = source.stat()
    if destination.is_file() and index_path.is_file():
        saved = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            saved.get("source_feature_size") == source_stat.st_size
            and saved.get("source_feature_mtime_ns") == source_stat.st_mtime_ns
            and saved.get("model") == model
            and saved.get("position") == POSITION
            and saved.get("protocol") == PROTOCOL
        ):
            array = np.load(destination, mmap_mode="r", allow_pickle=False)
            if list(array.shape) == saved.get("array_shape") and array.dtype == np.float16:
                print(f"[pack_qa_dhcp] reuse {model}: {len(array)} rows", flush=True)
                return saved

    with source.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty QA feature list: {source}")
    if len(rows) != 9000:
        raise ValueError(f"Expected 9000 CLEVR rows, got {len(rows)}")
    shard_root = (baseline_dir / "dhcp" / "shards").resolve()
    first = _attention(_payload(rows[0]), shard_root)
    shape = tuple(int(x) for x in first.shape)
    if len(shape) != 4 or shape[-2:] != (12, 12):
        raise ValueError(f"Unexpected DHCP item shape {shape}")
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{ARRAY_NAME}.", suffix=".tmp", dir=baseline_dir
    )
    os.close(fd)
    temporary = Path(temp_name)
    items = []
    try:
        packed = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(len(rows), *shape)
        )
        for offset, row in enumerate(rows):
            key = str(row["key"])
            split = str(row["probe_split"])
            label = int(row["label"])
            if split not in {"train", "test"} or label not in {0, 1}:
                raise ValueError(f"Invalid split or label for {key}")
            value = _attention(_payload(row), shard_root)
            if value.shape != shape:
                raise ValueError(f"Inconsistent DHCP shape at {key}: {value.shape}")
            packed[offset] = value
            items.append({"key": key, "split": split, "label": label})
            if (offset + 1) % 1000 == 0:
                print(f"[pack_qa_dhcp] {model}: {offset + 1}/{len(rows)}", flush=True)
        packed.flush()
        del packed
        if source.stat().st_mtime_ns != source_stat.st_mtime_ns:
            raise RuntimeError(f"Source changed during packing: {source}")
        metadata = {
            "schema_version": "qa-dhcp-dense-v1",
            "model": model,
            "dataset": "clevr_exist_9k",
            "position": POSITION,
            "protocol": PROTOCOL,
            "dtype": "float16",
            "array_shape": [len(rows), *shape],
            "source_feature_size": source_stat.st_size,
            "source_feature_mtime_ns": source_stat.st_mtime_ns,
            "items": items,
        }
        os.replace(temporary, destination)
        index_temp = index_path.with_name(f".{index_path.name}.tmp")
        index_temp.write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(index_temp, index_path)
        print(f"[pack_qa_dhcp] complete {model}: {destination}", flush=True)
        return metadata
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qa_benchmarks"))
    args = parser.parse_args()
    pack_model(args.output_root, args.model)


if __name__ == "__main__":
    main()
