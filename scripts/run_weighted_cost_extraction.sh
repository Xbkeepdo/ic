#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

model="${1:?model required}"
source_mode="${2:?source mode required: all}"
gpu="${3:?GPU index required}"
case "$source_mode" in
  all) config=configs/model_configs_coco4000_512_weighted_cost_all.yaml ;;
  raw_attention_all) config=configs/model_configs_coco4000_512_weighted_cost_raw_attention.yaml ;;
  softmax) config=configs/model_configs_coco4000_512_weighted_cost.yaml ;;
  pam|cam) config="configs/model_configs_coco4000_512_weighted_cost_${source_mode}.yaml" ;;
  *) echo "Unknown source mode: $source_mode" >&2; exit 2 ;;
esac
case "$model" in
  llava_1_5_7b|internvl_2_5_8b|qwen2_5_vl_7b)
    source_dir="outputs/$model/COCO4000-512-ICLR-RISK" ;;
  qwen3_vl_8b)
    source_dir="outputs/$model/COCO4000-ICLR" ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac

if [[ "$source_mode" == raw_attention_all ]]; then
  output_dir="outputs/$model/COCO4000-512-WCOST-RAWATTN"
else
  output_dir="outputs/$model/COCO4000-512-WCOST-${source_mode^^}"
fi
python_bin="${PYTHON_BIN:-/opt/conda/private/envs/vicr/bin/python}"
export CUDA_VISIBLE_DEVICES="$gpu"
export NLTK_DATA="${NLTK_DATA:-/home/apulis-dev/userdata/nltk_data}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$python_bin" scripts/generate_and_label.py \
  --model "$model" --config "$config" --output-dir "$output_dir" \
  --chair-cache outputs/chair_cache/coco_val2014_chair.pkl \
  --device cuda:0 --reuse-generations-from "$source_dir" --resume
"$python_bin" scripts/extract_features.py \
  --model "$model" --config "$config" --output-dir "$output_dir" \
  --device cuda:0 --feature-devices cuda:0 --resume
"$python_bin" scripts/train_weighted_cost_projection.py \
  --run-dir "$output_dir" --device cuda:0
