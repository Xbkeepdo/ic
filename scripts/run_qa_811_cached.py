#!/usr/bin/env python3
"""Run the cached 8:1:1 native QA baselines for all formal cohorts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2_5_vl_7b", "llava_1_5_7b", "qwen3_vl_8b", "internvl_2_5_8b")
DATASETS = ("pope", "clevr_exist_9k", "amber_discriminative")
LOGS = ROOT / "outputs" / "qa_benchmarks" / "LOGS-K32-TAU003-NATIVE6-FULL-811" / "logs"


def complete(model: str, dataset: str) -> bool:
    root = ROOT / "outputs" / "qa_benchmarks" / model / "LOGS-K32-TAU003-NATIVE6-FULL-811" / dataset / "baseline"
    positions = ("prompt_last_token",) if dataset == "amber_discriminative" else (
        "prompt_last_token", "question_object_pre_token"
    )
    for protocol in ("answer_correctness_all", "object_hallucination_yes_only"):
        for position in positions:
            for seed in (43, 44, 45):
                path = root / protocol / position / f"seed_{seed}" / "result.json"
                if not path.is_file() or len(json.loads(path.read_text()).get("methods", {})) != 6:
                    return False
    return True


def run(index: int, model: str, dataset: str) -> tuple[str, str, int]:
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{model}_{dataset}.log"
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(index % 2),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
    command = [sys.executable, str(ROOT / "scripts" / "train_qa_811_cached.py"),
               "--model", model, "--dataset", dataset, "--kinds", "baseline"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\nRUN {' '.join(command)}\n")
        handle.flush()
        status = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    return model, dataset, status


def main() -> None:
    jobs = [(model, dataset) for model in MODELS for dataset in DATASETS
            if not complete(model, dataset)]
    print(f"Remaining model/dataset jobs: {len(jobs)}", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, index, model, dataset)
                   for index, (model, dataset) in enumerate(jobs)]
        for future in as_completed(futures):
            model, dataset, status = future.result()
            print(f"{model}/{dataset}: exit={status}", flush=True)
            if status:
                failed.append((model, dataset, status))
    if failed:
        raise SystemExit(f"Failed jobs: {failed}")


if __name__ == "__main__":
    main()
