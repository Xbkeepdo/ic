#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
OUT="$ROOT/outputs/flog_target_k_tau_ablation"
PY=/opt/conda/private/envs/vicr/bin/python

cd "$ROOT"
while [[ ! -f "$OUT/shared_selection.json" ]]; do
  sleep 30
done

"$PY" scripts/train_flog_target_k_tau_ablation.py \
  --model qwen2_5_vl_7b --device cuda:0 --stage test-grid \
  > "$OUT/qwen2_5_vl_7b.test_grid.log" 2>&1 &
pid0=$!
"$PY" scripts/train_flog_target_k_tau_ablation.py \
  --model llava_1_5_7b --device cuda:1 --stage test-grid \
  > "$OUT/llava_1_5_7b.test_grid.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

echo "Local Qwen2.5 and LLaVA test grids completed."
