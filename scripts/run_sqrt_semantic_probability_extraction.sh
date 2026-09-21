#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

model="${1:?model required}"
gpu="${2:?GPU index required}"
source_dir="outputs/$model/COCO4000-512-WCOST-ALL"
output_dir="outputs/$model/COCO4000-512-SQRT-SEMANTIC"
config="configs/model_configs_coco4000_512_sqrt_semantic.yaml"
python_bin="${PYTHON_BIN:-/opt/conda/private/envs/vicr/bin/python}"

mkdir -p "$output_dir"
for name in generations.json labeling.json image_splits.json chair_summary.json coco_ground_truth.jsonl; do
  if [[ -f "$source_dir/$name" ]]; then
    cp -f "$source_dir/$name" "$output_dir/$name"
  fi
done

export CUDA_VISIBLE_DEVICES="$gpu"
export NLTK_DATA="${NLTK_DATA:-/home/apulis-dev/userdata/nltk_data}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$python_bin" scripts/extract_features.py \
  --model "$model" \
  --config "$config" \
  --output-dir "$output_dir" \
  --device cuda:0 \
  --feature-devices cuda:0 \
  --extraction-mode method_only \
  --dgst-branches hpre_softmax_prob_direct \
  --resume
