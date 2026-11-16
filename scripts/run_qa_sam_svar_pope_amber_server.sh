#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:?usage: $0 <llava_1_5_7b|qwen3_vl_8b> <physical_gpu_index>}"
GPU="${2:?physical GPU index required}"
case "$MODEL" in
  llava_1_5_7b|qwen3_vl_8b) ;;
  *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1) ;;
  *) echo "GPU must be 0 or 1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON=/root/rivermind-data/envs/ic-feature-extract/bin/python
CONFIG=configs/model_configs_qa_sam_svar_pope_amber_server.yaml
SOURCE_ROOT=/root/rivermind-data/dataset/qa_sam_svar_base
SELECTED_ROOT=/root/rivermind-data/dataset/qa_sam_svar_yesonly/"$MODEL"
OUTPUT_ROOT=/root/rivermind-data/ic-ac/outputs/qa_benchmarks
OUTPUT=POPE-POPULAR-AMBER-EXIST-SAM-SVAR-YESONLY
STATUS_DIR="$OUTPUT_ROOT/_sam_svar_pope_amber_status"
mkdir -p "$STATUS_DIR" "$SELECTED_ROOT"
STATUS_FILE="$STATUS_DIR/$MODEL.status"

on_exit() {
  code=$?
  if (( code == 0 )); then
    printf 'completed %s\n' "$(date -Is)" > "$STATUS_FILE"
  else
    printf 'failed exit=%s %s\n' "$code" "$(date -Is)" > "$STATUS_FILE"
  fi
}
trap on_exit EXIT
printf 'running %s\n' "$(date -Is)" > "$STATUS_FILE"

for DATASET in pope amber_discriminative; do
  echo "[$(date -Is)] $MODEL $DATASET generation"
  "$PYTHON" scripts/qa_pipeline.py \
    --model "$MODEL" --dataset "$DATASET" --config "$CONFIG" \
    --prepared-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --output "$OUTPUT" \
    --stage generate --device cuda:0 --generation-devices cuda:0 \
    --feature-devices cuda:0 --checkpoint-every 25 --resume

  echo "[$(date -Is)] $MODEL $DATASET labels"
  "$PYTHON" scripts/qa_pipeline.py \
    --model "$MODEL" --dataset "$DATASET" --config "$CONFIG" \
    --prepared-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --output "$OUTPUT" \
    --stage label --device cuda:0 --generation-devices cuda:0 \
    --feature-devices cuda:0 --resume

  LABELS="$OUTPUT_ROOT/$MODEL/$OUTPUT/$DATASET/labels.jsonl"
  echo "[$(date -Is)] $MODEL $DATASET Yes-only selection"
  "$PYTHON" scripts/prepare_qa_sam_svar_subsets.py \
    --source-root "$SOURCE_ROOT" --output-root "$SELECTED_ROOT" \
    --dataset "$DATASET" --yes-only-labels "$LABELS"

  echo "[$(date -Is)] $MODEL $DATASET SAM and SVAR extraction"
  "$PYTHON" scripts/qa_pipeline.py \
    --model "$MODEL" --dataset "$DATASET" --config "$CONFIG" \
    --prepared-root "$SELECTED_ROOT" --output-root "$OUTPUT_ROOT" --output "$OUTPUT" \
    --stage extract --device cuda:0 --generation-devices cuda:0 \
    --feature-devices cuda:0 --feature-sync-every 16 --resume

  "$PYTHON" scripts/export_qa_sam_svar_cross_position.py \
    --questions "$SELECTED_ROOT/$DATASET/questions.jsonl" \
    --run-dir "$OUTPUT_ROOT/$MODEL/$OUTPUT/$DATASET"

  echo "[$(date -Is)] $MODEL $DATASET complete"
done
