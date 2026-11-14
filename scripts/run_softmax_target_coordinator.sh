#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
OUT="$ROOT/outputs/softmax_target_k_tau_ablation"
PY=/opt/conda/private/envs/vicr/bin/python
MODELS=(llava_1_5_7b qwen2_5_vl_7b internvl_2_5_8b qwen3_vl_8b)
cd "$ROOT"

for _ in {1..720}; do
  ready=1
  for model in "${MODELS[@]}"; do
    [[ -f "$OUT/$model/validation_summary.csv" ]] || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 30
done
[[ "$ready" -eq 1 ]]
"$PY" scripts/train_softmax_target_k_tau_ablation.py --select-shared

for _ in {1..720}; do
  ready=1
  for model in "${MODELS[@]}"; do
    [[ -f "$OUT/$model/test_grid_summary.csv" ]] || ready=0
  done
  [[ "$ready" -eq 1 ]] && break
  sleep 30
done
[[ "$ready" -eq 1 ]]
"$PY" scripts/train_softmax_target_k_tau_ablation.py --summarize-grid
for model in "${MODELS[@]}"; do
  "$PY" scripts/train_softmax_target_k_tau_ablation.py \
    --model "$model" --stage final --device cpu
done
"$PY" scripts/train_softmax_target_k_tau_ablation.py --summarize-final
echo "Completed: $OUT/summary.md and $OUT/complete_results.md"
