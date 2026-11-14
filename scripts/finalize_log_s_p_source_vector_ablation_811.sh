#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
extract_root="$project_root/../gfchair/outputs/coco4000_512_log_s_source_vectors_811"
result_root="$project_root/outputs/log_s_p_source_vector_ablation_811"
python_bin=/opt/conda/private/envs/vicr/bin/python
mkdir -p "$result_root"

for ((attempt=0; attempt<360; attempt++)); do
  if [[ -f "$extract_root/llava_1_5_7b/summary.json" && \
        -f "$extract_root/qwen3_vl_8b/summary.json" ]]; then
    break
  fi
  sleep 30
done
if [[ ! -f "$extract_root/llava_1_5_7b/summary.json" || \
      ! -f "$extract_root/qwen3_vl_8b/summary.json" ]]; then
  echo 'Source-vector extraction did not complete within three hours.' >&2
  exit 1
fi

cd "$project_root"
"$python_bin" -m scripts.run_log_s_p_source_vector_ablation_811 \
  --model llava_1_5_7b --case post_ffn_residual --case attention_output --device cpu \
  > "$result_root/train_llava.log" 2>&1
"$python_bin" -m scripts.run_log_s_p_source_vector_ablation_811 \
  --model qwen3_vl_8b --case post_ffn_residual --case attention_output --device cpu \
  > "$result_root/train_qwen3.log" 2>&1
"$python_bin" -m scripts.run_log_s_p_source_vector_ablation_811 --summarize \
  > "$result_root/summarize.log" 2>&1
echo 'Source-vector ablation completed.'
