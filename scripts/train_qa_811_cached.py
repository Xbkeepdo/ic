#!/usr/bin/env python3
"""Train the formal QA method and native baselines on cached 8:1:1 splits."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler  # noqa: E402

from detection.baselines import (  # noqa: E402
    DHCPMLP,
    evaluate_detection_scores,
    raw_labels_to_hallucination_targets,
    select_detection_threshold,
)
from detection.qa_probe import (  # noqa: E402
    QAProbe,
    build_matrix,
    choose_real_f1_threshold,
    classification_metrics,
)
from features.baseline import (  # noqa: E402
    DHCPShardReader,
    baseline_config,
    get_baseline_payload,
    validate_baseline_record,
)
from features.qa_baseline import (  # noqa: E402
    baseline_record_for_position,
    qa_image_identity,
)
from scripts.train_baselines import (  # noqa: E402
    _build_threshold_reports,
    _evaluate_instruct_lens,
    _evaluate_projectaway,
    _train_metatoken,
    _train_svar,
    _train_token_grounding,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import save_json  # noqa: E402


SOURCE_NAME = "LOGS-K32-TAU003-NATIVE6-FULL"
TARGET_NAME = "LOGS-K32-TAU003-NATIVE6-FULL-811"
MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
DATASETS = ("pope", "clevr_exist_9k", "amber_discriminative")
PROTOCOLS = ("answer_correctness_all", "object_hallucination_yes_only")
SEEDS = (43, 44, 45)
METHODS = ("metatoken", "svar", "dhcp", "projectaway", "token_grounding", "instruct_lens")
SPLITS = ("train", "val", "test")


def _source(model: str, dataset: str) -> Path:
    return ROOT / "outputs" / "qa_benchmarks" / model / SOURCE_NAME / dataset


def _target(model: str, dataset: str) -> Path:
    return ROOT / "outputs" / "qa_benchmarks" / model / TARGET_NAME / dataset


def _assignment(dataset: str) -> dict[str, str]:
    path = ROOT / "outputs" / "qa_benchmarks" / TARGET_NAME / "splits" / f"{dataset}.json"
    payload = json.loads(path.read_text())
    if payload["protocol"] != "strict_outer_image_level_811_from_existing_outer82":
        raise ValueError(f"Unexpected split protocol: {path}")
    assignment = {identity: split for split in SPLITS for identity in payload["images"][split]}
    if len(assignment) != sum(payload["image_counts"].values()):
        raise ValueError(f"Overlapping images in {path}")
    return assignment


def _rows(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid QA cache: {path}")
    return records


def _device(value: str) -> str:
    return "cuda:0" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


def _fit_method(rows: list[dict], *, model_name: str, dataset_name: str,
                feature_set: str, protocol: str, seed: int,
                cfg: dict, device: str, output: Path) -> dict:
    matrices = {}
    for split in SPLITS:
        selected = [row for row in rows if row["probe_split"] == split]
        matrices[split] = build_matrix(selected, feature_set, protocol)
    widths = {matrices[split][0].shape[1] for split in SPLITS}
    if len(widths) != 1:
        raise ValueError(f"Method feature widths differ: {widths}")
    for split in SPLITS:
        X, y, _ = matrices[split]
        if len(X) == 0 or len(set(y.tolist())) != 2:
            raise ValueError(f"{feature_set}/{protocol}/{split} is empty or single-class")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    X_train, y_train, _ = matrices["train"]
    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(X_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-12)
    scaled = {
        split: ((matrices[split][0] - mean) / std).astype(np.float32)
        for split in SPLITS
    }
    model = QAProbe(
        X_train.shape[1], tuple(cfg["hidden_sizes"]), float(cfg["dropout"]),
        batch_norm=bool(cfg["batch_norm"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(cfg["lr_factor"]),
        patience=int(cfg["lr_patience"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled["train"]), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    val_x = torch.from_numpy(scaled["val"])
    val_y = torch.from_numpy(matrices["val"][1].astype(np.float32))
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    history = []
    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        total_loss, total_rows = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite QA method training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_rows += len(y)
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device)).cpu()
            val_loss = float(criterion(val_logits, val_y).item())
        if not math.isfinite(val_loss):
            raise RuntimeError("Non-finite QA method validation loss")
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / total_rows,
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "is_best": improved,
        })
        if stale >= int(cfg["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("QA method did not select a checkpoint")
    model.load_state_dict(best_state)

    def predict(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()

    probabilities = {split: predict(scaled[split]) for split in SPLITS}
    threshold, validation_f1 = choose_real_f1_threshold(
        matrices["val"][1], probabilities["val"]
    )
    report = {
        "model": model_name,
        "dataset": dataset_name,
        "position": feature_set.rsplit("@", 1)[1],
        "feature_set": feature_set,
        "label_protocol": protocol,
        "seed": seed,
        "split_protocol": "strict_outer_image_level_811",
        "checkpoint_selection": "minimum_validation_loss",
        "threshold_selection": "validation_real_f1",
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_completed": len(history),
        "validation_f1_at_selected_threshold": validation_f1,
        "threshold": threshold,
        "counts": {split: len(matrices[split][1]) for split in SPLITS},
        "class_counts": {split: np.bincount(matrices[split][1], minlength=2).tolist() for split in SPLITS},
        "metrics": {split: classification_metrics(matrices[split][1], probabilities[split], threshold) for split in SPLITS},
        "fixed_0.5_metrics": {split: classification_metrics(matrices[split][1], probabilities[split], 0.5) for split in SPLITS},
        "training_config": {key: cfg[key] for key in ("hidden_sizes", "dropout", "batch_norm", "learning_rate", "weight_decay", "batch_size", "max_epochs", "lr_factor", "lr_patience", "early_stopping_patience")},
        "history": history,
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "threshold": threshold}, output / "checkpoint.pt")
    save_json(report, str(output / "result.json"))
    return report


def _train_dhcp_gpu(groups: dict[str, list[dict]], baseline_dir: Path,
                    checkpoint_dir: Path, cfg: dict, device: str) -> dict:
    """Train the native DHCP head with its extracted input resident on GPU."""
    dhcp_cfg = dict(cfg.get("dhcp") or {})
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    records = [row for split in SPLITS for row in groups[split]]
    offsets = {}
    start = 0
    for split in SPLITS:
        offsets[split] = torch.arange(start, start + len(groups[split]), device=device)
        start += len(groups[split])
    reader = None
    first = get_baseline_payload(records[0], "dhcp")
    if "attention" not in first:
        reader = DHCPShardReader(baseline_dir / "dhcp" / "shards")

    def attention(row: dict) -> np.ndarray:
        payload = get_baseline_payload(row, "dhcp")
        if "attention" in payload:
            return np.asarray(payload["attention"], dtype=np.float16)
        assert reader is not None
        return np.asarray(reader.load(payload.get("shard_reference", payload)), dtype=np.float16)

    shape = attention(records[0]).shape
    if not shape or math.prod(shape) <= 0:
        raise ValueError("Empty DHCP input shape")
    tensor = torch.empty((len(records), *shape), dtype=torch.float16, device=device)
    for index, row in enumerate(records):
        values = attention(row)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"DHCP input shape/value mismatch at row {index}")
        tensor[index].copy_(torch.from_numpy(np.array(values, copy=True)).to(device))
    raw = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    hall = raw_labels_to_hallucination_targets(raw)
    hall_gpu = torch.as_tensor(hall, dtype=torch.long, device=device)
    train_n = len(groups["train"])
    train_targets = hall[:train_n]
    counts = np.bincount(train_targets, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights[train_targets], dtype=torch.double),
        num_samples=train_n, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    hidden_dim = int(dhcp_cfg.get("hidden_dim", dhcp_cfg.get("hidden_size", 128)))
    model = DHCPMLP(math.prod(shape), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(dhcp_cfg.get("learning_rate", 1e-3)))
    criterion = nn.CrossEntropyLoss()
    batch_size = int(dhcp_cfg.get("batch_size", 1024))
    maximum_epochs = int(dhcp_cfg.get("epochs", dhcp_cfg.get("max_epochs", 30)))
    patience = int(dhcp_cfg.get("early_stopping_patience", 5))

    def evaluate(indices: torch.Tensor) -> tuple[float, np.ndarray]:
        model.eval()
        losses, chunks = [], []
        with torch.no_grad():
            for selected in indices.split(batch_size):
                logits = model(tensor.index_select(0, selected).float())
                targets = hall_gpu.index_select(0, selected)
                loss = criterion(logits, targets)
                losses.append(float(loss.item()))
                chunks.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return float(np.mean(losses)), np.concatenate(chunks)

    best_loss, best_epoch, best_state, stale, history = math.inf, 0, None, 0, []
    for epoch in range(maximum_epochs):
        model.train()
        sampled = list(iter(sampler))
        train_losses = []
        for start in range(0, len(sampled), batch_size):
            local = torch.as_tensor(sampled[start:start + batch_size], device=device)
            values = tensor.index_select(0, local).float()
            targets = hall_gpu.index_select(0, local)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, _ = evaluate(offsets["val"])
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                        "val_loss": val_loss, "is_best": improved})
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No DHCP validation checkpoint selected")
    model.load_state_dict(best_state)
    _, val_scores = evaluate(offsets["val"])
    _, test_scores = evaluate(offsets["test"])
    val_raw = raw[train_n:train_n + len(groups["val"])]
    test_raw = raw[train_n + len(groups["val"]):]
    threshold = select_detection_threshold(val_raw, val_scores, positive_class="real")
    checkpoint = checkpoint_dir / "dhcp.pt"
    torch.save({"state_dict": best_state, "input_shape": shape,
                "hidden_dim": hidden_dim, "threshold": threshold}, checkpoint)
    return {
        "paper_config": {
            "protocol": "object_prediction_fixed_grid_mass_preserving",
            "target_grid": [12, 12], "hidden_dim": hidden_dim,
            "learning_rate": float(dhcp_cfg.get("learning_rate", 1e-3)),
            "batch_size": batch_size, "epochs": maximum_epochs,
            "early_stopping_patience": patience,
            "weighted_sampler": True, "positive_class": "real",
        },
        "input_shape": list(shape), "input_dim": math.prod(shape),
        "threshold": threshold, "threshold_score_class": "real",
        "val_metrics": evaluate_detection_scores(val_raw, val_scores, threshold, positive_class="real"),
        "test_metrics": evaluate_detection_scores(test_raw, test_scores, threshold, positive_class="real"),
        "threshold_reports": _build_threshold_reports(
            train_labels=val_raw, train_scores=val_scores,
            test_labels=test_raw, test_scores=test_scores,
            selected_threshold=threshold, positive_class="real",
        ),
        "history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "checkpoint": str(checkpoint), "selection_protocol": "validation_selected",
    }


def _fit_baselines(records: list[dict], *, model_name: str, dataset_name: str,
                   position_name: str, protocol: str, seed: int,
                   baseline_cfg: dict, source_baseline_dir: Path,
                   device: str, output: Path) -> dict:
    groups = {split: [row for row in records if row["probe_split"] == split] for split in SPLITS}
    for split in SPLITS:
        labels = {int(row["label"]) for row in groups[split]}
        if labels != {0, 1}:
            raise ValueError(f"Baseline {protocol}/{split} is empty or single-class: {labels}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = {**baseline_cfg, "seed": seed, "_strict_82_no_validation": False}
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "position": position_name,
        "label_protocol": protocol,
        "seed": seed,
        "split_protocol": "strict_outer_image_level_811",
        "checkpoint_selection": "validation_loss_where_applicable",
        "threshold_selection": "validation_real_f1",
        "counts": {split: len(groups[split]) for split in SPLITS},
        "methods": {},
    }
    saved = output / "result.json"
    if saved.exists():
        previous = json.loads(saved.read_text())
        for field in ("model", "dataset", "position", "label_protocol", "seed", "split_protocol"):
            if previous.get(field) != result[field]:
                raise ValueError(f"Cannot resume incompatible baseline result: {saved}, field={field}")
        result = previous
    for method in METHODS:
        if method in result["methods"]:
            print(f"REUSE BASELINE {model_name}/{dataset_name}/{position_name}/{protocol}/seed{seed}/{method}", flush=True)
            continue
        path = output / method
        path.mkdir(exist_ok=True)
        print(f"BASELINE {model_name}/{dataset_name}/{position_name}/{protocol}/seed{seed}/{method}", flush=True)
        if method == "metatoken":
            values = _train_metatoken(groups, path, cfg, "real")
        elif method == "svar":
            values = _train_svar(groups, path, cfg, device, "real")
        elif method == "dhcp":
            values = _train_dhcp_gpu(groups, source_baseline_dir, path, cfg, device)
        elif method == "projectaway":
            values = _evaluate_projectaway(groups, "real")
        elif method == "token_grounding":
            values = _train_token_grounding(groups, path, cfg, device, "real")
        else:
            values = _evaluate_instruct_lens(groups, "real")
        result["methods"][method] = values
        save_json(result, str(output / "result.json"))
    return result


def run(model: str, dataset: str, *, position: str, protocol: str,
        kinds: str, device: str, seeds: tuple[int, ...]) -> None:
    assignment = _assignment(dataset)
    source, target = _source(model, dataset), _target(model, dataset)
    config_path = ROOT / "configs" / (
        "model_configs_qa_logs_k32_tau003_amber.yaml"
        if dataset == "amber_discriminative" else "model_configs_qa_logs_k32_tau003.yaml"
    )
    config = load_config(str(config_path))
    if kinds in {"all", "method"}:
        rows = _rows(source / "features.pkl")
        rows = [{**row, "probe_split": assignment[qa_image_identity(row)]} for row in rows]
        feature_set = f"logs_risk_f_log@{position}"
        for seed in seeds:
            output = target / "method" / protocol / position / f"seed_{seed}"
            if (output / "result.json").exists():
                print(f"REUSE {output / 'result.json'}", flush=True)
                continue
            print(f"METHOD {model}/{dataset}/{position}/{protocol}/seed{seed}", flush=True)
            _fit_method(rows, model_name=model, dataset_name=dataset,
                        feature_set=feature_set, protocol=protocol,
                        seed=seed, cfg=config["training"]["torch_probe"],
                        device=device, output=output)
    if kinds in {"all", "baseline"}:
        baseline_dir = source / "baseline" / protocol
        records = _rows(baseline_dir / "features.pkl")
        positioned = []
        for record in records:
            value = baseline_record_for_position(record, position)
            if value is None:
                continue
            assigned = assignment[qa_image_identity(value)]
            value = {**value, "probe_split": assigned}
            validate_baseline_record(value, required=METHODS)
            positioned.append(value)
        if not positioned:
            raise ValueError(f"No baseline rows for {position}")
        for seed in seeds:
            output = target / "baseline" / protocol / position / f"seed_{seed}"
            saved = output / "result.json"
            if saved.exists() and set(json.loads(saved.read_text())["methods"]) == set(METHODS):
                print(f"REUSE {saved}", flush=True)
                continue
            _fit_baselines(positioned, model_name=model, dataset_name=dataset,
                           position_name=position, protocol=protocol, seed=seed,
                           baseline_cfg=baseline_config(config),
                           source_baseline_dir=baseline_dir,
                           device=device, output=output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--position", choices=("prompt_last_token", "question_object_pre_token", "all"), default="all")
    parser.add_argument("--protocol", choices=PROTOCOLS + ("all",), default="all")
    parser.add_argument("--kinds", choices=("all", "method", "baseline"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    if args.dataset == "amber_discriminative" and args.position == "question_object_pre_token":
        parser.error("AMBER has only prompt_last_token")
    positions = (
        ("prompt_last_token",) if args.dataset == "amber_discriminative"
        else ("prompt_last_token", "question_object_pre_token")
    ) if args.position == "all" else (args.position,)
    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    for position in positions:
        for protocol in protocols:
            run(args.model, args.dataset, position=position, protocol=protocol,
                kinds=args.kinds, device=_device(args.device), seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
