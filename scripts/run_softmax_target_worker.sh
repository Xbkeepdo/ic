#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
DEVICE="$2"
ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
GF_OUT=/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair/outputs/coco4000_512_softmax_target_k_tau_ablation_811
OUT="$ROOT/outputs/softmax_target_k_tau_ablation"
PY=/opt/conda/private/envs/vicr/bin/python
cd "$ROOT"

for _ in {1..720}; do
  [[ -f "$GF_OUT/$MODEL/summary.json" ]] && break
  sleep 30
done
[[ -f "$GF_OUT/$MODEL/summary.json" ]]

"$PY" scripts/train_softmax_target_k_tau_ablation.py \
  --model "$MODEL" --device "$DEVICE" --stage validation \
  > "$OUT/$MODEL.validation.log" 2>&1

for _ in {1..720}; do
  [[ -f "$OUT/shared_selection.json" ]] && break
  sleep 30
done
[[ -f "$OUT/shared_selection.json" ]]

"$PY" scripts/train_softmax_target_k_tau_ablation.py \
  --model "$MODEL" --device "$DEVICE" --stage test-grid \
  > "$OUT/$MODEL.test_grid.log" 2>&1
