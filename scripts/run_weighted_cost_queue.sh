#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
gpu="${1:?GPU index required}"
log_dir=outputs/weighted_cost_projection/logs
mkdir -p "$log_dir"

case "$gpu" in
  0) models=(llava_1_5_7b internvl_2_5_8b) ;;
  1) models=(qwen2_5_vl_7b qwen3_vl_8b) ;;
  *) echo "Queue only supports local GPU 0 or 1" >&2; exit 2 ;;
esac

for model in "${models[@]}"; do
  for mode in all; do
    stem="$model-$mode"
    if bash scripts/run_weighted_cost_extraction.sh "$model" "$mode" "$gpu" \
      >"$log_dir/$stem.log" 2>&1; then
      printf 'complete\n' >"$log_dir/$stem.status"
    else
      printf 'failed\n' >"$log_dir/$stem.status"
    fi
  done
done
