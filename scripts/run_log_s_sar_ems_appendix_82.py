#!/usr/bin/env python3
"""Train aligned SAR-source and EMS-target ablations on the strict COCO 8:2 split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts.train_log_s_hpre_k_tau_ablation import _probe
from scripts.train_torch_probe_feature_sets import _json_ready, train_and_evaluate_probe
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/log_s_sar_ems_appendix_82"
MODELS = ("llava_1_5_7b", "qwen2_5_vl_7b", "internvl_2_5_8b", "qwen3_vl_8b")
DISPLAY = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "qwen3_vl_8b": "Qwen3-VL-8B",
}
SEEDS = (43, 44, 45)
CASES = (
    "sar_only",
    "ems_only",
    "sar_ems",
    "i_ems",
    "target_i_s",
    "target_i_residual",
    "target_softmax_a_s",
    "target_softmax_i_residual",
)
CASE_NAMES = {
    "sar_only": "SAR only",
    "ems_only": "EMS only",
    "sar_ems": "SAR + EMS; T=norm(A s)",
    "i_ems": "I + EMS; T=norm(A s)",
    "target_i_s": "SAR + EMS; T=norm(I s)",
    "target_i_residual": "SAR + EMS; T=norm(I (1+s))",
    "target_softmax_a_s": "SAR + EMS; T=softmax(A s)",
    "target_softmax_i_residual": "SAR + EMS; T=softmax(I (1+s))",
}
TEX_NAMES = {
    "sar_only": "SAR only",
    "ems_only": "EMS only",
    "sar_ems": "SAR + EMS / $\\operatorname{norm}_m(A_m s_m)$",
    "i_ems": "$I$ + EMS / $\\operatorname{norm}_m(A_m s_m)$",
    "target_i_s": "$\\operatorname{norm}_m(I_m s_m)$",
    "target_i_residual": "$\\operatorname{norm}_m(I_m(1+s_m))$",
    "target_softmax_a_s": "$\\operatorname{softmax}_m(A_m s_m)$",
    "target_softmax_i_residual": "$\\operatorname{softmax}_m(I_m(1+s_m))$",
}
METRICS = ("auroc", "real_aupr", "hall_aupr", "real_f1", "real_precision", "real_recall", "accuracy")
KEYS = ("label", "image_id", "response_token_idx", "target_token_id")


def _read(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in fields}


def _check_rows(model: str, source: str, base: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> None:
    for key in KEYS:
        if not np.array_equal(base[key], other[key]):
            raise ValueError(f"{model}: {source} differs from reference at {key}")


def _check_matrix(model: str, name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{model}: invalid {name} matrix, shape={matrix.shape}, expected={shape}")
    return matrix


def load_cases(model: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    base = _read(
        ROOT / "outputs/log_s_hpre_k_tau_ablation" / model / "compact_features.npz",
        KEYS + ("f_log", "log_s_hpre_k032_tau003", "train_mask", "strict_test_mask"),
    )
    sar = np.asarray(base["f_log"], dtype=np.float32)
    ems = np.asarray(base["log_s_hpre_k032_tau003"], dtype=np.float32)
    shape = sar.shape
    if ems.shape != shape or len(shape) != 2:
        raise ValueError(f"{model}: SAR/EMS layer mismatch")
    train = np.asarray(base["train_mask"], dtype=bool)
    test = np.asarray(base["strict_test_mask"], dtype=bool)
    if np.any(train & test) or not np.all(train | test):
        raise ValueError(f"{model}: invalid strict 8:2 masks")
    train_images = set(base["image_id"][train].tolist())
    test_images = set(base["image_id"][test].tolist())
    if train_images & test_images:
        raise ValueError(f"{model}: train/test image overlap")

    mechanism = _read(
        ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz",
        KEYS + ("sar", "ems", "i_ems"),
    )
    _check_rows(model, "mechanism", base, mechanism)
    if not np.allclose(mechanism["sar"], sar, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{model}: mechanism SAR differs")
    if not np.allclose(mechanism["ems"], ems, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{model}: mechanism EMS differs")
    i_ems = np.asarray(mechanism["i_ems"], dtype=np.float32)
    if i_ems.shape != (shape[0], 2 * shape[1]):
        raise ValueError(f"{model}: invalid I+EMS shape")
    if not np.allclose(i_ems[:, shape[1]:], ems, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{model}: I+EMS risk half differs")

    norm_path = ROOT / "outputs" / f"{model}_log_s_hpre_amnorm_target_82_ablation" / "compact_features.npz"
    norm = _read(norm_path, KEYS + ("attention_target_risk", "amnorm_target_risk", "train_mask", "test_mask"))
    _check_rows(model, "amnorm target", base, norm)
    if not np.array_equal(norm["train_mask"], train) or not np.array_equal(norm["test_mask"], test):
        raise ValueError(f"{model}: amnorm target uses a different image split")
    if not np.allclose(norm["attention_target_risk"], ems, rtol=2e-4, atol=1e-4):
        raise ValueError(f"{model}: amnorm reference risk differs")

    flog = _read(
        ROOT / "outputs/flog_target_k_tau_ablation" / model / "compact_features.npz",
        KEYS + ("f_log", "flog_target_k032_tau003", "train_mask", "strict_test_mask"),
    )
    soft = _read(
        ROOT / "outputs/softmax_target_k_tau_ablation" / model / "compact_features.npz",
        KEYS + ("f_log", "attn_softmax_k032_tau003", "norm_softmax_k032_tau003", "train_mask", "strict_test_mask"),
    )
    for source, data in (("F-target", flog), ("softmax target", soft)):
        _check_rows(model, source, base, data)
        if not np.array_equal(data["train_mask"], train) or not np.array_equal(data["strict_test_mask"], test):
            raise ValueError(f"{model}: {source} uses a different image split")
        if not np.allclose(data["f_log"], sar, rtol=2e-4, atol=1e-4):
            raise ValueError(f"{model}: {source} SAR differs")

    risks = {
        "target_i_s": norm["amnorm_target_risk"],
        "target_i_residual": flog["flog_target_k032_tau003"],
        "target_softmax_a_s": soft["attn_softmax_k032_tau003"],
        "target_softmax_i_residual": soft["norm_softmax_k032_tau003"],
    }
    cases = {
        "sar_only": _check_matrix(model, "SAR", sar, shape),
        "ems_only": _check_matrix(model, "EMS", ems, shape),
        "sar_ems": np.concatenate((ems, sar), axis=1),
        "i_ems": np.concatenate((ems, i_ems[:, :shape[1]]), axis=1),
        **{
            name: np.concatenate((_check_matrix(model, name, risk, shape), sar), axis=1)
            for name, risk in risks.items()
        },
    }
    if tuple(cases) != CASES:
        raise AssertionError("Case order drifted")
    audit = {
        "model": model, "rows": int(shape[0]), "layers": int(shape[1]),
        "train_tokens": int(train.sum()), "test_tokens": int(test.sum()),
        "train_images_with_mentions": len(train_images), "test_images_with_mentions": len(test_images),
        "risk_reference_max_abs_diff": float(np.max(np.abs(norm["attention_target_risk"] - ems))),
        "fusion_feature_order": "EMS then SAR (or EMS then I), matching the prior strict-8:2 reference",
        "sources": {
            "reference": str(ROOT / "outputs/log_s_hpre_k_tau_ablation" / model / "compact_features.npz"),
            "I": str(ROOT / "outputs/log_s_mechanism_ablation_811" / model / "compact_features.npz"),
            "norm_I_s": str(norm_path),
            "norm_I_residual": str(ROOT / "outputs/flog_target_k_tau_ablation" / model / "compact_features.npz"),
            "softmax_targets": str(ROOT / "outputs/softmax_target_k_tau_ablation" / model / "compact_features.npz"),
        },
    }
    return cases, base, audit


def _flatten(result: dict) -> dict[str, float]:
    real, hall = result["real_positive"], result["hallucination_positive"]
    return {
        "auroc": float(real["auc"]),
        "real_aupr": float(real["aupr"]),
        "hall_aupr": float(hall["aupr"]),
        "real_f1": float(real["f1"]),
        "real_precision": float(real["precision"]),
        "real_recall": float(real["recall"]),
        "accuracy": float(result["accuracy"]),
    }


def train(model: str, device_name: str) -> None:
    torch.set_num_threads(1)
    cases, data, audit = load_cases(model)
    model_out = OUT / model
    model_out.mkdir(parents=True, exist_ok=True)
    (model_out / "alignment_audit.json").write_text(json.dumps(audit, indent=2))
    root_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    train_mask = np.asarray(data["train_mask"], dtype=bool)
    test_mask = np.asarray(data["strict_test_mask"], dtype=bool)
    labels = np.asarray(data["label"], dtype=np.int64)
    for case, matrix in cases.items():
        for seed in SEEDS:
            folder = model_out / case / f"seed{seed}"
            metrics_path = folder / "metrics.json"
            if metrics_path.exists():
                print(f"Skip {model}/{case}/seed{seed}", flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            config = _probe(root_config, seed)
            print(f"Train {model}/{case}/seed{seed} on {device_name}", flush=True)
            result = train_and_evaluate_probe(
                X_train=matrix[train_mask], y_train=labels[train_mask],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=matrix[test_mask], y_test=labels[test_mask],
                config=config, device=torch.device(device_name), output_dir=str(folder),
            )
            metrics_path.write_text(json.dumps(_json_ready(result), indent=2))
            (folder / "feature_case.json").write_text(json.dumps({
                "model": model, "case": case, "display": CASE_NAMES[case],
                "seed": seed, "input_dim": int(matrix.shape[1]),
                "split": "strict_82_no_validation", "config": asdict(config),
            }, indent=2))
    print(f"Complete {model}", flush=True)


def summarize() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        model_out = OUT / model
        if not (model_out / "alignment_audit.json").exists():
            raise FileNotFoundError(model_out / "alignment_audit.json")
        for case in CASES:
            for seed in SEEDS:
                path = model_out / case / f"seed{seed}" / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                values = _flatten(json.loads(path.read_text()))
                rows.append({"model": model, "case": case, "seed": seed, **values, "source": str(path)})
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for model in MODELS:
        for case in CASES:
            matching = [r for r in rows if r["model"] == model and r["case"] == case]
            grouped[(model, case)] = {
                metric: (float(np.mean([r[metric] for r in matching])), float(np.std([r[metric] for r in matching])))
                for metric in METRICS
            }
    lines = [
        "# SAR semantic and EMS target ablations — COCO ENDAC strict 8:2", "",
        "Four models, 3200 train / 800 test images, identical aligned object-token keys; K=32, source τ=0.03, seeds 43/44/45.",
        "All eight cases use the same one-hidden-layer (width 128) probe: train-only z-score, no BatchNorm, dropout 0.3, batch 256 with drop_last, Adam LR 1e-3 and weight decay 1e-5. The checkpoint minimizes training loss; Real-F1 threshold is selected on training scores.",
        "AUROC and AUPR do not depend on this threshold. Values are test mean ± population standard deviation in percent. Real/Hall AUPR state their positive class explicitly. Existing feature caches were audited for exact mention-key and split alignment; all probes here were trained afresh with one trainer.", "",
    ]
    for title, ordered in (
        ("SAR semantic contribution", CASES[:4]),
        ("EMS target construction with SAR fixed", ("sar_ems",) + CASES[4:]),
    ):
        lines.extend([f"## {title}", "", "| Model | Feature / Target | AUROC | Real AUPR | Hall AUPR | Real F1 |", "|---|---|---:|---:|---:|---:|"])
        for model in MODELS:
            for case in ordered:
                values = grouped[(model, case)]
                formatted = [f"{100*values[m][0]:.2f} ± {100*values[m][1]:.2f}" for m in ("auroc", "real_aupr", "hall_aupr", "real_f1")]
                lines.append(f"| {DISPLAY[model]} | {CASE_NAMES[case]} | " + " | ".join(formatted) + " |")
        lines.append("")
    lines.extend([
        "The SAR contrast is SAR+EMS versus I+EMS; EMS is identical in those rows. The target contrast holds SAR, source, cost, K and τ fixed, changing only T.",
        "The semantic-only and I source curves use all 4000 images and are descriptive; these classification rows use only the held-out 800-image test split.", "",
    ])
    (OUT / "complete_results.md").write_text("\n".join(lines))

    for filename, caption, ordered in (
        ("sar_semantic_ablation_82.tex", "SAR semantic contribution", CASES[:4]),
        ("ems_target_constructions_82.tex", "Alternative target distributions for EMS", ("sar_ems",) + CASES[4:]),
    ):
        tex = [
            "% Requires booktabs and graphicx; each cell is AUROC / Hall-positive AUPR (percent, mean over three seeds).",
            "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
            "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
            "Input / target & LLaVA-1.5 & Qwen2.5-VL & InternVL2.5 & Qwen3-VL \\\\ ", "\\midrule",
        ]
        for case in ordered:
            cells = []
            for model in MODELS:
                metric = grouped[(model, case)]
                cells.append(f"{100*metric['auroc'][0]:.2f} / {100*metric['hall_aupr'][0]:.2f}")
            label = TEX_NAMES[case]
            if filename == "sar_semantic_ablation_82.tex" and case == "sar_ems":
                label = "SAR + EMS"
            elif filename == "sar_semantic_ablation_82.tex" and case == "i_ems":
                label = "$I$ + EMS"
            tex.append(f"{label} & " + " & ".join(cells) + " \\\\ ")
        tex += [
            "\\bottomrule", "\\end{tabular}", "}",
            f"\\caption{{{caption} on the fixed 3200/800 COCO image split. Each cell is test AUROC / hallucination-positive AUPR (\\%, mean over seeds 43/44/45). K=32 and source temperature $\\tau=0.03$; all probes share one training setup.}}",
            f"\\label{{tab:{filename.removesuffix('.tex').replace('_', '-')}}}", "\\end{table*}", "",
        ]
        (OUT / filename).write_text("\n".join(tex))
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.model:
        if args.audit_only:
            _cases, _data, audit = load_cases(args.model)
            print(json.dumps(audit, indent=2))
        else:
            train(args.model, args.device)
    else:
        parser.error("Provide --model or --summarize")


if __name__ == "__main__":
    main()
