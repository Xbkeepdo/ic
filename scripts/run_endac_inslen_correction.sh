#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MODEL" >&2
  exit 2
fi

MODEL="$1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="configs/model_configs_coco4000_512_endac_inslen_finalnorm.yaml"
SOURCE_DIR="outputs/${MODEL}/COCO4000-512-ENDAC-BASELINES-V2"
CORRECTION_DIR="outputs/${MODEL}/COCO4000-512-ENDAC-INSLEN-FINALNORM"
FEATURE_DEVICES_VALUE="${FEATURE_DEVICES:-cuda:0 cuda:1}"

mkdir -p "$CORRECTION_DIR"
for name in generations.json labeling.json image_splits.json; do
  if [[ ! -e "$CORRECTION_DIR/$name" ]]; then
    ln -s "$(realpath "$SOURCE_DIR/$name")" "$CORRECTION_DIR/$name"
  fi
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" scripts/extract_baselines.py \
  --model "$MODEL" \
  --config "$CONFIG" \
  --output-dir "$CORRECTION_DIR" \
  --device cuda:0 \
  --feature-devices $FEATURE_DEVICES_VALUE \
  --resume

"$PYTHON_BIN" scripts/merge_endac_baseline_payloads.py \
  --model "$MODEL" \
  --output-dir "$SOURCE_DIR" \
  --instruct-lens-correction "$CORRECTION_DIR/baseline/features.pkl"

"$PYTHON_BIN" scripts/train_baselines.py \
  --model "$MODEL" \
  --config configs/model_configs_coco4000_512_endac_baselines_v2.yaml \
  --output-dir "$SOURCE_DIR" \
  --device cuda:0
