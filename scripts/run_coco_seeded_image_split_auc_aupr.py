#!/usr/bin/env python3
"""Re-evaluate original COCO Ours and native baselines with seed-specific 8:2 splits.

Each of seeds 43/44/45 both shuffles the same 4000 image IDs and initializes
the detector associated with that method. Ours reuses the pre-optimization
K=32/tau=.03 cached risk and F_log features; no LVLM extraction is repeated.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from scripts import train_baselines
from scripts.train_log_s_hpre_k_tau_ablation import _fit
from features.baseline import baseline_config
from utils.config_utils import load_config
from utils.io_utils import load_pkl


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/coco4000_seeded_image_split_auc_aupr"
SOURCE_RUN = "COCO4000-512-ENDAC-BASELINES-V2"
MODELS = ("llava_1_5_7b", "qwen3_vl_8b")
SEEDS = (43, 44, 45)
METHODS = (
    "ours", "metatoken", "svar", "projectaway",
    "token_grounding", "instruct_lens", "dhcp",
)
DISPLAY = {
    "ours": "Ours (SAR+EMS)",
    "metatoken": "MetaToken (GB)",
    "svar": "SVAR",
    "dhcp": "DHCP",
    "projectaway": "ProjectAway",
    "token_grounding": "TokenGrounding (MLP)",
    "instruct_lens": "InsLen",
}


# The native DHCP dataset decodes one item from its shard on every epoch and
# on the train-as-monitor pass. This run keeps exactly those decoded float32
# tensors in RAM after the first read. The sampler, model, and optimizer stay
# in the native training implementation.
_NATIVE_DHCP_DATASET = train_baselines.DHCPRecordDataset
_DHCP_VALUE_CACHE: dict[tuple[str, int, int, int], tuple[torch.Tensor, int]] = {}


class CachedDHCPRecordDataset(_NATIVE_DHCP_DATASET):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        key = (str(self.reader.root), *row_key(self.records[index]))
        value = _DHCP_VALUE_CACHE.get(key)
        if value is None:
            value = super().__getitem__(index)
            _DHCP_VALUE_CACHE[key] = value
        return value


def source_root(model: str) -> Path:
    return ROOT / "outputs" / model / SOURCE_RUN


def old_feature_path(model: str) -> Path:
    return ROOT / "outputs/log_s_hpre_k_tau_ablation" / model / "compact_features.npz"


def row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["image_id"]), int(row["response_token_idx"]), int(row["target_token_id"]))


def image_pool() -> np.ndarray:
    pools = []
    for model in MODELS:
        saved = json.loads((source_root(model) / "image_splits.json").read_text())
        values = np.asarray(sorted(set(map(int, saved["train"] + saved["test"]))), dtype=np.int64)
        if len(values) != 4000:
            raise ValueError(f"{model}: expected 4000 unique images")
        pools.append(values)
    if not np.array_equal(pools[0], pools[1]):
        raise ValueError("LLaVA and Qwen3 image pools differ")
    return pools[0]


def seeded_split(seed: int, pool: np.ndarray) -> dict[str, list[int]]:
    shuffled = np.random.default_rng(seed).permutation(pool)
    return {
        "train": np.sort(shuffled[:3200]).astype(int).tolist(),
        "val": [],
        "test": np.sort(shuffled[3200:]).astype(int).tolist(),
    }


def save_splits(pool: np.ndarray) -> dict[int, dict[str, list[int]]]:
    folder = OUT / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for seed in SEEDS:
        split = seeded_split(seed, pool)
        path = folder / f"seed{seed}.json"
        if path.exists():
            if json.loads(path.read_text()) != split:
                raise ValueError(f"Existing split differs: {path}")
        else:
            path.write_text(json.dumps(split, indent=2) + "\n")
        result[seed] = split
    return result


def score_metrics(result: dict[str, Any], method: str) -> dict[str, float]:
    if method == "ours":
        test = result["threshold_reports"]["train_f1"]["test_metrics"]
    else:
        entry = result["methods"][method]
        if method == "metatoken":
            entry = entry["gb"]
        elif method == "token_grounding":
            entry = entry["mlp"]
        test = entry["test_metrics"]
    return {
        "auroc": float(test["real_positive"]["auc"]),
        "real_aupr": float(test["real_positive"]["aupr"]),
        "hall_aupr": float(test["hallucination_positive"]["aupr"]),
    }


def run_model(model: str, device: str, splits: dict[int, dict[str, list[int]]]) -> None:
    torch.set_num_threads(1)
    model_root = OUT / model
    model_root.mkdir(parents=True, exist_ok=True)
    baseline_source = source_root(model) / "baseline"
    baseline_path = baseline_source / "features.pkl"
    records = load_pkl(str(baseline_path))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid baseline cache: {baseline_path}")
    with np.load(old_feature_path(model), allow_pickle=False) as saved:
        ours = {key: np.asarray(saved[key]) for key in (
            "image_id", "response_token_idx", "target_token_id", "label",
            "log_s_hpre_k032_tau003", "f_log",
        )}
    if ours["log_s_hpre_k032_tau003"].shape != ours["f_log"].shape:
        raise ValueError(f"{model}: SAR/EMS dimensions differ")
    baseline_labels = {row_key(row): int(row["label"]) for row in records}
    ours_labels = {
        (int(image), int(response), int(target)): int(label)
        for image, response, target, label in zip(
            ours["image_id"], ours["response_token_idx"],
            ours["target_token_id"], ours["label"],
        )
    }
    if len(baseline_labels) != len(records) or baseline_labels != ours_labels:
        raise ValueError(f"{model}: baseline and Ours mention keys/labels differ")
    config = load_config(str(ROOT / "configs/model_configs_coco4000_512_endac_baselines_v2.yaml"))
    baseline_cfg = copy.deepcopy(baseline_config(config))
    baseline_cfg["metatoken"]["classifiers"] = ["gb"]
    baseline_cfg["token_grounding"]["classifiers"] = ["mlp"]
    ours_config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
    baseline_dir = model_root / "baseline"
    shards_link = baseline_dir / "dhcp" / "shards"
    shards_link.parent.mkdir(parents=True, exist_ok=True)
    if not shards_link.exists():
        shards_link.symlink_to((baseline_source / "dhcp" / "shards").resolve(), target_is_directory=True)
    if shards_link.resolve() != (baseline_source / "dhcp" / "shards").resolve():
        local = Path("/tmp/coco_seeded_dhcp_cache") / f"{model}_shards"
        if shards_link.resolve() != local.resolve():
            raise ValueError("DHCP shard link points to unexpected source")
    train_baselines.DHCPRecordDataset = CachedDHCPRecordDataset
    matrix = np.concatenate((ours["log_s_hpre_k032_tau003"], ours["f_log"]), axis=1)
    print(f"[{model}] aligned {len(records)} object tokens", flush=True)

    for seed in SEEDS:
        split = splits[seed]
        train_images = set(split["train"])
        test_images = set(split["test"])
        train_mask = np.isin(ours["image_id"], list(train_images))
        test_mask = np.isin(ours["image_id"], list(test_images))
        if not np.all(train_mask | test_mask) or np.any(train_mask & test_mask):
            raise ValueError(f"{model}/seed{seed}: incomplete/overlapping Ours split")
        split_records = {
            "train": [row for row in records if int(row["image_id"]) in train_images],
            "val": [],
            "test": [row for row in records if int(row["image_id"]) in test_images],
        }
        if (len(split_records["train"]), len(split_records["test"])) != (int(train_mask.sum()), int(test_mask.sum())):
            raise ValueError(f"{model}/seed{seed}: token split counts differ")
        counts = {
            "seed": seed, "train_images": 3200, "test_images": 800,
            "train_tokens": int(train_mask.sum()), "test_tokens": int(test_mask.sum()),
            "train_hall": int(np.sum(ours["label"][train_mask] == 0)),
            "test_hall": int(np.sum(ours["label"][test_mask] == 0)),
        }
        (model_root / f"seed{seed}_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        for method in METHODS:
            if method == "ours":
                folder = model_root / f"seed{seed}" / "ours"
                _fit(
                    matrix=matrix, labels=ours["label"],
                    fit_mask=train_mask, eval_mask=test_mask,
                    seed=seed, root_config=ours_config,
                    device=torch.device(device), folder=folder,
                )
            else:
                run_name = f"seed{seed}_{method}"
                result_path = baseline_dir / "results" / run_name / f"{model}_baselines.json"
                if result_path.exists():
                    saved = json.loads(result_path.read_text())
                    if method in saved.get("methods", {}):
                        print(f"[{model}] skip seed{seed} {method}", flush=True)
                        continue
                seed_cfg = copy.deepcopy(baseline_cfg)
                seed_cfg["seed"] = seed
                seed_cfg["_strict_82_no_validation"] = True
                train_baselines._train_one_seed(
                    model=model, seed=seed, methods=(method,),
                    split_records=split_records,
                    image_split_counts={"train": 3200, "val": 0, "test": 800},
                    feature_path=baseline_path,
                    split_path=OUT / "splits" / f"seed{seed}.json",
                    baseline_dir=baseline_dir,
                    baseline_cfg=seed_cfg, device=device,
                    run_name=run_name, result_stem=f"{model}_baselines",
                    positive_class="real",
                )
            print(f"[{model}] completed seed{seed} {method}", flush=True)


def summarize() -> None:
    lines = [
        "# COCO4000：图片划分和检测器随机性共用种子", "",
        "LLaVA-1.5-7B 与 Qwen3-VL-8B；每个 seed 43/44/45 用同一 4000 图池重新生成图片级 3200/800 训练/测试划分，并用于相应随机检测器初始化。两模型同种子共享图片划分。Ours 复用优化前的 K=32、τ=0.03 SAR+EMS 特征；其他方法复用原生 baseline 特征，各自按既有原生检测器训练。MetaToken 取 GB，TokenGrounding 取 MLP。ProjectAway/InsLen 无随机初始化，但其测试图片随种子改变。AUROC、Real AUPR 和 Hall AUPR 都由测试集连续分数计算，无阈值选择。", "",
        "| 模型 | 方法 | AUROC (%) | Real AUPR (%) | Hall AUPR (%) |",
        "|---|---|---:|---:|---:|",
    ]
    all_rows = []
    for model in MODELS:
        model_rows = []
        for method in METHODS:
            values = []
            for seed in SEEDS:
                if method == "ours":
                    path = OUT / model / f"seed{seed}" / "ours" / "metrics.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    saved = json.loads(path.read_text())
                else:
                    path = OUT / model / "baseline" / "results" / f"seed{seed}_{method}" / f"{model}_baselines.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    saved = json.loads(path.read_text())
                score = score_metrics(saved, method)
                values.append({"model": model, "method": method, "seed": seed, **score})
            model_rows.extend(values)
            summary = {key: (mean(row[key] for row in values), pstdev(row[key] for row in values)) for key in ("auroc", "real_aupr", "hall_aupr")}
            fmt = lambda item: f"{100 * summary[item][0]:.2f} ± {100 * summary[item][1]:.2f}"
            lines.append(f"| {model} | {DISPLAY[method]} | {fmt('auroc')} | {fmt('real_aupr')} | {fmt('hall_aupr')} |")
        all_rows.extend(model_rows)
    lines += ["", "每种子的图片列表在 `splits/seed*.json`；逐种子完整检测结果保存在各模型目录。表内标准差为三个 seed 的总体标准差。", ""]
    (OUT / "complete_results.md").write_text("\n".join(lines))
    (OUT / "per_seed.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    print(OUT / "complete_results.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
        return
    if args.model is None:
        parser.error("--model is required unless --summarize is used")
    run_model(args.model, args.device, save_splits(image_pool()))


if __name__ == "__main__":
    main()
