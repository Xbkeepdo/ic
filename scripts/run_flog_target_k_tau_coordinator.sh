#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
OUT="$ROOT/outputs/flog_target_k_tau_ablation"
PY=/opt/conda/private/envs/vicr/bin/python
MODELS=(llava_1_5_7b qwen2_5_vl_7b internvl_2_5_8b qwen3_vl_8b)

cd "$ROOT"
while true; do
  ready=1
  for model in "${MODELS[@]}"; do
    if [[ ! -f "$OUT/$model/validation_summary.csv" ]]; then
      ready=0
    fi
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep 30
done

"$PY" scripts/train_flog_target_k_tau_ablation.py --select-shared

while true; do
  ready=1
  for model in "${MODELS[@]}"; do
    if [[ ! -f "$OUT/$model/test_grid_summary.csv" ]]; then
      ready=0
    fi
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep 30
done

"$PY" scripts/train_flog_target_k_tau_ablation.py --summarize-grid
for model in "${MODELS[@]}"; do
  "$PY" scripts/train_flog_target_k_tau_ablation.py \
    --model "$model" --stage final --device cpu
done
"$PY" scripts/train_flog_target_k_tau_ablation.py --summarize-final
"$PY" scripts/update_flog_target_k_tau_progress.py
echo "F_log target ablation completed: $OUT/complete_results.md"
