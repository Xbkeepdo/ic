#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
GF_ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
OUT="$ROOT/outputs/log_s_hpre_k_tau_ablation"
GF_OUT="$GF_ROOT/outputs/coco4000_512_log_s_hpre_k_tau_ablation_811"
PY=/opt/conda/private/envs/vicr/bin/python

mkdir -p "$OUT"

while true; do
  complete=1
  for model in qwen2_5_vl_7b llava_1_5_7b qwen3_vl_8b internvl_2_5_8b; do
    if [[ ! -f "$GF_OUT/$model/summary.json" ]]; then
      complete=0
    fi
  done
  if [[ "$complete" -eq 1 ]]; then
    break
  fi
  sleep 30
done

cd "$ROOT"
(
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model qwen2_5_vl_7b --device cuda:0 --stage validation
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model qwen3_vl_8b --device cuda:0 --stage validation
) > "$OUT/gpu0.validation.log" 2>&1 &
p0=$!
(
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model llava_1_5_7b --device cuda:1 --stage validation
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model internvl_2_5_8b --device cuda:1 --stage validation
) > "$OUT/gpu1.validation.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"

"$PY" scripts/train_log_s_hpre_k_tau_ablation.py --select-shared

(
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model qwen2_5_vl_7b --device cuda:0 --stage final
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model qwen3_vl_8b --device cuda:0 --stage final
) > "$OUT/gpu0.final.log" 2>&1 &
p0=$!
(
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model llava_1_5_7b --device cuda:1 --stage final
  "$PY" scripts/train_log_s_hpre_k_tau_ablation.py --model internvl_2_5_8b --device cuda:1 --stage final
) > "$OUT/gpu1.final.log" 2>&1 &
p1=$!
wait "$p0"
wait "$p1"

"$PY" scripts/train_log_s_hpre_k_tau_ablation.py --summarize-final
"$PY" scripts/update_log_s_hpre_k_tau_ablation_progress.py --once
