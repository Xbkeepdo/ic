#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL EXTRACTION_CONFIG" >&2
  exit 2
fi

MODEL="$1"
EXTRACTION_CONFIG="$2"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="outputs/${MODEL}/COCO4000-512-ENDAC-BASELINES-V2"
TRAIN_CONFIG="configs/model_configs_coco4000_512_endac_baselines_v2.yaml"
FEATURE_DEVICES_VALUE="${FEATURE_DEVICES:-cuda:0 cuda:1}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" scripts/extract_baselines.py \
  --model "$MODEL" \
  --config "$EXTRACTION_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --feature-devices $FEATURE_DEVICES_VALUE \
  --resume

"$PYTHON_BIN" scripts/merge_endac_baseline_payloads.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/train_baselines.py \
  --model "$MODEL" \
  --config "$TRAIN_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0
