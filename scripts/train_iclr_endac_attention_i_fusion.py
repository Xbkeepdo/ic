#!/usr/bin/env python3
"""Compare ICLR risk with aligned ENDAC attention or visual I features."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GF_ROOT = ROOT.parent / "gfchair"
sys.path.insert(0, str(ROOT))

from scripts.train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    _json_ready,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config  # noqa: E402
from utils.io_utils import load_pkl  # noqa: E402


MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
    "qwen3_vl_8b",
)
RISK_KEYS = {
    "raw_semantic": "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_per_layer",
    "raw_target_penalty": (
        "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_plus_target_penalty_per_layer"
    ),
    "softmax_semantic": "dgst_t_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer",
    "softmax_target_penalty": (
        "dgst_t_hpre_softmax_prob_gauss_risk_sqrt_hpre_plus_target_penalty_per_layer"
    ),
}
FUSIONS = ("risk", "risk_attention", "risk_I")
SEEDS = (43, 44, 45)


def _key(row: dict) -> tuple[int, int, int]:
    return (
        int(row["image_id"]),
        int(row["response_token_idx"]),
        int(row["target_token_id"]),
    )


def load_aligned(
    model: str,
    risk_run_name: str = "COCO4000-ICLR",
    require_identical_generations: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    gf_dir = GF_ROOT / "outputs/coco4000_512_endac_attention_i_fusion_811" / model
    gf_source = GF_ROOT / "outputs/coco4000_512_endac_811" / model / "features.pkl"
    if Path(risk_run_name).name != risk_run_name or risk_run_name in {".", ".."}:
        raise ValueError("risk_run_name must be one directory name")
    td_dir = ROOT / "outputs" / model / risk_run_name
    gf = torch.load(gf_dir / "features.pt", map_location="cpu", weights_only=False, mmap=True)
    source = load_pkl(str(gf_source))
    td = load_pkl(str(td_dir / "features.pkl"))
    gf_generations = json.loads((gf_source.parent / "generations.json").read_text())
    td_generations = json.loads((td_dir / "generations.json").read_text())
    same_full_generation_images = sum(
        gf_generations.get(image_id) == row
        for image_id, row in td_generations.items()
    )
    if require_identical_generations and (
        same_full_generation_images != len(gf_generations)
        or len(gf_generations) != len(td_generations)
    ):
        raise ValueError(f"{model}: generation files differ between GF and risk")
    if len(source) != len(gf["sample_id"]):
        raise ValueError(f"{model}: GF cache/source length mismatch")
    if [row["sample_id"] for row in source] != list(gf["sample_id"]):
        raise ValueError(f"{model}: GF cache/source order mismatch")
    if len({_key(row) for row in source}) != len(source):
        raise ValueError(f"{model}: duplicate GF mention keys")
    td_by_key = {_key(row): row for row in td}
    if len(td_by_key) != len(td):
        raise ValueError(f"{model}: duplicate ICLR mention keys")

    positions = []
    matched = []
    matched_key_rows = 0
    rejected_prefix_rows = 0
    for index, row in enumerate(source):
        other = td_by_key.get(_key(row))
        if other is None:
            continue
        matched_key_rows += 1
        if int(row["label"]) != int(other["label"]):
            raise ValueError(f"{model}: label mismatch at {_key(row)}")
        if str(row["canonical_object"]) != str(other["token_str"]):
            raise ValueError(f"{model}: object mismatch at {_key(row)}")
        if int(gf["image_id"][index]) != int(row["image_id"]):
            raise ValueError(f"{model}: GF image mismatch at index {index}")
        if int(gf["labels"][index]) != int(row["label"]):
            raise ValueError(f"{model}: GF label mismatch at index {index}")
        image_id, token_idx, target_id = _key(row)
        gf_tokens = gf_generations[str(image_id)]["response_token_ids"]
        td_tokens = td_generations[str(image_id)]["response_token_ids"]
        if (token_idx >= len(gf_tokens) or token_idx >= len(td_tokens)
                or int(gf_tokens[token_idx]) != target_id
                or int(td_tokens[token_idx]) != target_id):
            raise ValueError(f"{model}: generation target mismatch at {_key(row)}")
        if gf_tokens[:token_idx + 1] != td_tokens[:token_idx + 1]:
            rejected_prefix_rows += 1
            continue
        positions.append(index)
        matched.append(other)

    selected = np.asarray(positions, dtype=np.int64)
    if len(selected) == 0:
        raise ValueError(f"{model}: no exact mention matches")
    image_ids = np.asarray(gf["image_id"], dtype=np.int64)[selected]
    labels = np.asarray(gf["labels"], dtype=np.int64)[selected]
    attention = np.asarray(gf["all_visual"]["raw"], dtype=np.float32)[selected, :, 1]
    attention_norm = np.asarray(gf["all_visual"]["norm"], dtype=np.float32)[selected, :, 1]
    if not np.array_equal(attention, attention_norm):
        raise ValueError(f"{model}: raw/norm all-visual attention differs")
    visual_i = np.asarray(gf["I"], dtype=np.float32)[selected]
    risks = {
        name: np.stack(
            [np.asarray(row[key], dtype=np.float32) for row in matched]
        )
        for name, key in RISK_KEYS.items()
    }
    if any(value.shape != attention.shape for value in (*risks.values(), visual_i)):
        raise ValueError(f"{model}: layer dimensions differ between sources")
    if any(not np.isfinite(value).all() for value in (*risks.values(), attention, visual_i)):
        raise ValueError(f"{model}: nonfinite input features")

    splits = json.loads((td_dir / "image_splits.json").read_text())
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}
    if train_ids & test_ids:
        raise ValueError(f"{model}: overlapping image splits")
    train_mask = np.isin(image_ids, list(train_ids))
    test_mask = np.isin(image_ids, list(test_ids))
    if not np.all(train_mask | test_mask):
        raise ValueError(f"{model}: matched samples outside ICLR split")
    if set(np.unique(labels[train_mask])) != {0, 1} or set(np.unique(labels[test_mask])) != {0, 1}:
        raise ValueError(f"{model}: train/test needs both classes")
    data = dict(
        image_id=image_ids,
        label=labels,
        attention=attention,
        I=visual_i,
        train_mask=train_mask,
        test_mask=test_mask,
        **risks,
    )
    audit = dict(
        model=model,
        join_key="(image_id, response_token_idx, target_token_id, identical response-token prefix through target)",
        labels_and_canonical_objects_checked=True,
        gf_rows=len(source),
        iclr_rows=len(td),
        risk_run_name=risk_run_name,
        identical_full_generation_images=same_full_generation_images,
        matched_key_rows=matched_key_rows,
        rejected_prefix_rows=rejected_prefix_rows,
        matched_rows=len(selected),
        unmatched_gf_rows=len(source) - len(selected),
        unmatched_iclr_rows=len(td) - len(selected),
        matched_train_rows=int(train_mask.sum()),
        matched_test_rows=int(test_mask.sum()),
        matched_train_images=len(set(image_ids[train_mask].tolist())),
        matched_test_images=len(set(image_ids[test_mask].tolist())),
        layers=int(attention.shape[1]),
        split="ICLR strict image-level 3200/800; GF old 3200/400/400 ignored",
        gf_attention="all_visual/raw/attention_only; norm is exactly identical",
        gf_I="raw visual sum of attention-write norms per layer",
        gf_source=str(gf_dir / "features.pt"),
        iclr_source=str(td_dir / "features.pkl"),
        risk_keys=RISK_KEYS,
    )
    del td, source, matched, gf, gf_generations, td_generations
    gc.collect()
    return data, audit


def _matrix(data: dict[str, np.ndarray], risk: str, fusion: str) -> np.ndarray:
    parts = [data[risk]]
    if fusion == "risk_attention":
        parts.append(data["attention"])
    elif fusion == "risk_I":
        parts.append(data["I"])
    elif fusion != "risk":
        raise ValueError(fusion)
    return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]


def _probe_config(root: dict, seed: int) -> TorchProbeConfig:
    training = root["training"]
    cfg = training["torch_probe"]
    return TorchProbeConfig(
        hidden_sizes=tuple(int(x) for x in cfg["hidden_sizes"]),
        dropout=float(cfg["dropout"]),
        drop_last=bool(cfg["drop_last"]),
        batch_size=int(cfg["batch_size"]),
        num_epochs=int(cfg["max_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        lr_factor=float(cfg["lr_factor"]),
        lr_patience=int(cfg["lr_patience"]),
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        seed=seed,
        positive_class=str(training["positive_class"]),
        split_protocol=str(training["split_protocol"]),
        threshold_selection=str(training["threshold_selection"]),
        fixed_threshold=float(cfg["fixed_threshold"]),
        threshold_reporting=tuple(cfg["threshold_reporting"]),
        checkpoint_selection=str(cfg["checkpoint_selection"]),
    )


def _summary(model: str, out: Path, audit: dict) -> None:
    rows = []
    raw = {}
    for risk in RISK_KEYS:
        for fusion in FUSIONS:
            cases = []
            for seed in SEEDS:
                path = out / model / risk / fusion / f"seed{seed}" / "metrics.json"
                metrics = json.loads(path.read_text())
                cases.append(metrics)
            raw[(risk, fusion)] = cases
            for threshold in ("fixed_0.5", "train_f1"):
                reports = [case["threshold_reports"][threshold]["test_metrics"] for case in cases]
                row = dict(model=model, risk=risk, fusion=fusion, threshold=threshold)
                for name, values in {
                    "real_f1": [report["real_positive"]["f1"] for report in reports],
                    "real_auc": [report["real_positive"]["auc"] for report in reports],
                    "hall_aupr": [report["hallucination_positive"]["aupr"] for report in reports],
                }.items():
                    row[f"{name}_mean"] = float(np.mean(values))
                    row[f"{name}_std"] = float(np.std(values))
                base = raw.get((risk, "risk"))
                if base is not None:
                    baseline = [case["threshold_reports"][threshold]["test_metrics"] for case in base]
                    row["delta_real_auc"] = float(np.mean([
                        report["real_positive"]["auc"] - ref["real_positive"]["auc"]
                        for report, ref in zip(reports, baseline)
                    ]))
                    row["delta_hall_aupr"] = float(np.mean([
                        report["hallucination_positive"]["aupr"] - ref["hallucination_positive"]["aupr"]
                        for report, ref in zip(reports, baseline)
                    ]))
                rows.append(row)
    csv_path = out / model / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {model}: ICLR risk + ENDAC attention / I",
        "",
        f"Exact matched mentions: {audit['matched_rows']} (train {audit['matched_train_rows']}, test {audit['matched_test_rows']}).",
        f"Key matches rejected for differing generation prefix: {audit['rejected_prefix_rows']}.",
        "The original GF 3200/400/400 split is ignored; all heads use ICLR's image-level 3200/800 split.",
        "Each risk-only baseline is retrained on the same matched rows, seeds 43–45, using the ICLR Torch MLP.",
        "Attention = ENDAC all_visual/raw/attention_only; all_visual/norm/attention_only is identical. I is raw visual write-norm sum.",
        "Deltas are paired against risk-only for the same risk branch and seed. Values below are exploratory test results.",
        "",
        "| Threshold | Risk | Added feature | Real F1 | Real AUC | Δ AUC | Hall AUPR | Δ Hall AUPR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fmt = lambda key: f"{row[key + '_mean']:.3f} ± {row[key + '_std']:.3f}"
        lines.append(
            f"| {row['threshold']} | {row['risk']} | {row['fusion']} | "
            f"{fmt('real_f1')} | {fmt('real_auc')} | {row['delta_real_auc']:+.3f} | "
            f"{fmt('hall_aupr')} | {row['delta_hall_aupr']:+.3f} |"
        )
    (out / model / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--risk-run-name", default="COCO4000-ICLR")
    parser.add_argument("--require-identical-generations", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/iclr_endac_attention_i_fusion_prefix_aligned",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    data, audit = load_aligned(
        args.model,
        args.risk_run_name,
        args.require_identical_generations,
    )
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        return
    out = args.output_root.resolve()
    model_out = out / args.model
    model_out.mkdir(parents=True, exist_ok=True)
    audit_path = model_out / "alignment.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError(f"Existing alignment differs: {audit_path}")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    device = torch.device(args.device)
    for risk in RISK_KEYS:
        for fusion in FUSIONS:
            features = _matrix(data, risk, fusion)
            for seed in SEEDS:
                case_out = model_out / risk / fusion / f"seed{seed}"
                metrics_path = case_out / "metrics.json"
                if metrics_path.exists():
                    print("SKIP", args.model, risk, fusion, seed, flush=True)
                    continue
                cfg = _probe_config(config, seed)
                metrics = train_and_evaluate_probe(
                    X_train=features[data["train_mask"]],
                    y_train=data["label"][data["train_mask"]],
                    X_val=np.empty((0, features.shape[1]), dtype=np.float32),
                    y_val=np.empty((0,), dtype=np.int32),
                    X_test=features[data["test_mask"]],
                    y_test=data["label"][data["test_mask"]],
                    config=cfg,
                    device=device,
                    output_dir=str(case_out),
                )
                metrics_path.write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
                print("DONE", args.model, risk, fusion, seed, flush=True)
    _summary(args.model, out, audit)
    print("SUMMARY", model_out / "summary.md", flush=True)


if __name__ == "__main__":
    main()
