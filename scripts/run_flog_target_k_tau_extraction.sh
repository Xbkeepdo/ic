#!/usr/bin/env bash
set -euo pipefail

GF_ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
GF_OUT="$GF_ROOT/outputs/coco4000_512_flog_target_k_tau_ablation_811"
PY=/opt/conda/private/envs/vicr/bin/python

mkdir -p "$GF_OUT"
cd "$GF_ROOT"

run_model() {
  local model="$1"
  local device="$2"
  "$PY" scripts/extract_flog_target_k_tau_ablation_811.py \
    --model "$model" --device "$device" --output-root "$GF_OUT" \
    > "$GF_OUT/$model.log" 2>&1
}

(
  run_model qwen3_vl_8b cuda:0
  run_model qwen2_5_vl_7b cuda:0
) &
pid0=$!
(
  run_model internvl_2_5_8b cuda:1
  run_model llava_1_5_7b cuda:1
) &
pid1=$!
wait "$pid0"
wait "$pid1"

echo "All four F_log target extractions completed."
