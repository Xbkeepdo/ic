#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
GF_ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
OUT="$ROOT/outputs/flog_target_k_tau_ablation"
GF_OUT="$GF_ROOT/outputs/coco4000_512_flog_target_k_tau_ablation_811"
PY=/opt/conda/private/envs/vicr/bin/python

mkdir -p "$OUT"
cd "$ROOT"

wait_for_extraction() {
  local model="$1"
  local gpu_owner="$2"
  while [[ ! -f "$GF_OUT/$model/summary.json" || ! -f "$GF_OUT/$gpu_owner/summary.json" ]]; do
    sleep 30
  done
}

(
  wait_for_extraction qwen2_5_vl_7b qwen3_vl_8b
  "$PY" scripts/train_flog_target_k_tau_ablation.py \
    --model qwen2_5_vl_7b --device cuda:0 --stage validation \
    > "$OUT/qwen2_5_vl_7b.validation.log" 2>&1
) &
pid0=$!
(
  wait_for_extraction llava_1_5_7b internvl_2_5_8b
  "$PY" scripts/train_flog_target_k_tau_ablation.py \
    --model llava_1_5_7b --device cuda:1 --stage validation \
    > "$OUT/llava_1_5_7b.validation.log" 2>&1
) &
pid1=$!
wait "$pid0"
wait "$pid1"

echo "Local Qwen2.5 and LLaVA validation completed."
