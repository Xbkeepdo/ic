#!/usr/bin/env python3
"""Train native DHCP on a packed CLEVR Prompt-last cache (seeds 43/44/45)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.baselines import (  # noqa: E402
    DHCPMLP,
    evaluate_detection_scores,
    raw_labels_to_hallucination_targets,
    select_detection_threshold,
)
from scripts.pack_qa_dhcp import ARRAY_NAME, INDEX_NAME, MODELS, PROTOCOL, RUN  # noqa: E402
from scripts.train_baselines import aggregate_baseline_outputs, _build_threshold_reports  # noqa: E402


SEEDS = (43, 44, 45)
EPOCHS = 30
BATCH_SIZE = 1024
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3


def _load_cache(baseline_dir: Path, model: str) -> tuple[np.memmap, dict]:
    index = json.loads((baseline_dir / INDEX_NAME).read_text(encoding="utf-8"))
    array = np.load(baseline_dir / ARRAY_NAME, mmap_mode="r", allow_pickle=False)
    expected = {
        "schema_version": "qa-dhcp-dense-v1",
        "model": model,
        "dataset": "clevr_exist_9k",
        "position": "prompt_last_token",
        "protocol": PROTOCOL,
        "dtype": "float16",
    }
    for field, value in expected.items():
        if index.get(field) != value:
            raise ValueError(f"DHCP cache {field}: {index.get(field)!r} != {value!r}")
    if array.dtype != np.float16 or list(array.shape) != index.get("array_shape"):
        raise ValueError("DHCP array shape or dtype differs from cache index")
    if len(array) != 9000 or len(index["items"]) != len(array):
        raise ValueError("Incomplete DHCP cache")
    source = baseline_dir / "features.pkl"
    stat = source.stat()
    if (
        stat.st_size != index["source_feature_size"]
        or stat.st_mtime_ns != index["source_feature_mtime_ns"]
    ):
        raise ValueError("Source QA features changed after DHCP cache packing")
    keys = [item["key"] for item in index["items"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate DHCP sample keys")
    return array, index


def _scores(
    detector: DHCPMLP,
    features: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    detector.eval()
    chunks = []
    with torch.no_grad():
        for selected in indices.split(batch_size):
            values = features.index_select(0, selected).float()
            probability = torch.softmax(detector(values), dim=-1)[:, 1]
            chunks.append(probability.cpu().numpy())
    return np.concatenate(chunks)


def _load_gpu_array(array: np.memmap, device: torch.device) -> torch.Tensor:
    # GPU resident float16 input eliminates random disk reads on each epoch.
    # The native DHCP MLP still receives the same float32 values per batch.
    features = torch.empty(array.shape, dtype=torch.float16, device=device)
    for start in range(0, len(array), 256):
        stop = min(start + 256, len(array))
        features[start:stop] = torch.from_numpy(
            np.array(array[start:stop], copy=True)
        ).to(device)
    return features


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train_model(output_root: Path, model_name: str, device_name: str) -> dict:
    baseline_dir = output_root / model_name / RUN / "clevr_exist_9k" / "baseline" / PROTOCOL
    array, index = _load_cache(baseline_dir, model_name)
    split_info = json.loads(
        (baseline_dir / "probe_splits.prompt_last_token.json").read_text(encoding="utf-8")
    )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DHCP cached training requires a CUDA GPU")
    train_np = np.array([i for i, row in enumerate(index["items"]) if row["split"] == "train"], dtype=np.int64)
    test_np = np.array([i for i, row in enumerate(index["items"]) if row["split"] == "test"], dtype=np.int64)
    if len(train_np) != 7200 or len(test_np) != 1800:
        raise ValueError(f"Unexpected CLEVR split: {len(train_np)}/{len(test_np)}")
    if split_info["question_counts"] != {"train": 7200, "val": 0, "test": 1800}:
        raise ValueError("Saved CLEVR split counts differ from DHCP cache")
    for split, indices in (("train", train_np), ("test", test_np)):
        packed_keys = {index["items"][int(i)]["key"] for i in indices}
        if packed_keys != set(split_info[split]):
            raise ValueError(f"DHCP {split} keys differ from saved probe split")
    labels = np.array([row["label"] for row in index["items"]], dtype=np.int64)
    if set(labels[train_np]) != {0, 1} or set(labels[test_np]) != {0, 1}:
        raise ValueError("Train and test must contain both labels")
    train_raw, test_raw = labels[train_np], labels[test_np]
    train_targets = raw_labels_to_hallucination_targets(train_raw)
    tensor = _load_gpu_array(array, device)
    train_indices = torch.as_tensor(train_np, device=device)
    test_indices = torch.as_tensor(test_np, device=device)
    all_targets = torch.as_tensor(raw_labels_to_hallucination_targets(labels), device=device)
    input_shape = tuple(int(v) for v in array.shape[1:])
    input_dim = int(np.prod(input_shape))
    result_dir = baseline_dir / "results" / "dhcp_cached_prompt_last_token"
    checkpoint_dir = baseline_dir / "checkpoints" / "dhcp_cached_prompt_last_token"
    outputs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = DHCPMLP(input_dim, hidden_dim=HIDDEN_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        counts = np.bincount(train_targets, minlength=2)
        weights = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights[train_targets], dtype=torch.double),
            num_samples=len(train_targets),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        history = []
        for epoch in range(EPOCHS):
            model.train()
            sampled = list(iter(sampler))
            losses = []
            for start in range(0, len(sampled), BATCH_SIZE):
                local = torch.as_tensor(sampled[start:start + BATCH_SIZE], device=device)
                selected = train_indices.index_select(0, local)
                values = tensor.index_select(0, selected).float()
                targets = all_targets.index_select(0, selected)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(values), targets)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            train_loss = float(np.mean(losses))
            history.append({"epoch": epoch + 1, "train_loss": train_loss})
            print(
                f"[DHCP] {model_name} seed={seed} epoch={epoch + 1}/{EPOCHS} "
                f"train_loss={train_loss:.6f}", flush=True
            )
        train_scores = _scores(model, tensor, train_indices, BATCH_SIZE)
        test_scores = _scores(model, tensor, test_indices, BATCH_SIZE)
        threshold = select_detection_threshold(
            train_raw, train_scores, positive_class="real"
        )
        reports = _build_threshold_reports(
            train_labels=train_raw,
            train_scores=train_scores,
            test_labels=test_raw,
            test_scores=test_scores,
            selected_threshold=threshold,
            positive_class="real",
        )
        seed_checkpoint_dir = checkpoint_dir / f"seed{seed}"
        seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_checkpoint_dir / "dhcp.pt"
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_shape": input_shape,
            "hidden_dim": HIDDEN_DIM,
            "threshold": threshold,
            "threshold_score_class": "real",
        }, checkpoint)
        method = {
            "paper_config": {
                "protocol": "object_prediction_fixed_grid_mass_preserving",
                "target_grid": [12, 12],
                "hidden_dim": HIDDEN_DIM,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "weighted_sampler": True,
                "positive_class": "real",
            },
            "input_shape": list(input_shape),
            "input_dim": input_dim,
            "threshold": threshold,
            "threshold_score_class": "real",
            "train_metrics": evaluate_detection_scores(train_raw, train_scores, threshold, positive_class="real"),
            "test_metrics": evaluate_detection_scores(test_raw, test_scores, threshold, positive_class="real"),
            "threshold_reports": reports,
            "history": history,
            "checkpoint": str(checkpoint),
            "selection_protocol": "fixed_last_epoch_train_f1_threshold",
            "storage_protocol": "qa-dhcp-dense-v1-gpu-resident",
        }
        output = {
            "model": model_name,
            "seed": seed,
            "trainer": "native_paper",
            "configured_methods": ["dhcp"],
            "feature_path": str(baseline_dir / ARRAY_NAME),
            "split_path": str(baseline_dir / INDEX_NAME),
            "stored_label_semantics": {"0": "hallucination", "1": "real"},
            "detector_target_semantics": {"0": "real", "1": "hallucination"},
            "headline_positive_class": "real",
            "label_protocol": "qa_answer_correctness_all_prompt_last_token_0hall_1real_question_probe_split_physical_image_disjoint",
            "counts": {"train": len(train_np), "val": 0, "test": len(test_np)},
            "image_split_counts": split_info["image_counts"],
            "methods": {"dhcp": method},
            "split_protocol": "strict_82_no_validation",
            "checkpoint_selection": "last_epoch",
            "threshold_selection": "train_f1",
            "threshold_reporting": ["fixed_0.5", "train_f1"],
        }
        path = result_dir / f"seed{seed}" / "dhcp.json"
        _save_json(path, output)
        outputs.append(output)
    summary = aggregate_baseline_outputs(outputs, positive_class="real")
    _save_json(result_dir / "dhcp_3seed.json", summary)
    print(f"[DHCP] complete {model_name}: {result_dir / 'dhcp_3seed.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qa_benchmarks"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train_model(args.output_root, args.model, args.device)


if __name__ == "__main__":
    main()
