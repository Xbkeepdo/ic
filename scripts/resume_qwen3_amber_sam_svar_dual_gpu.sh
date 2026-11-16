#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON=/root/rivermind-data/envs/ic-feature-extract/bin/python
CONFIG=configs/model_configs_qa_sam_svar_pope_amber_server.yaml
MODEL=qwen3_vl_8b
DATASET=amber_discriminative
SOURCE_ROOT=/root/rivermind-data/dataset/qa_sam_svar_base
SELECTED_ROOT=/root/rivermind-data/dataset/qa_sam_svar_yesonly/$MODEL
OUTPUT_ROOT=/root/rivermind-data/ic-ac/outputs/qa_benchmarks
OUTPUT=POPE-POPULAR-AMBER-EXIST-SAM-SVAR-YESONLY
RUN_DIR=$OUTPUT_ROOT/$MODEL/$OUTPUT/$DATASET
STATUS_DIR=$OUTPUT_ROOT/_sam_svar_pope_amber_status
STATUS_FILE=$STATUS_DIR/$MODEL.status

on_exit() {
  code=$?
  if (( code == 0 )); then
    printf 'completed %s\n' "$(date -Is)" > "$STATUS_FILE"
  else
    printf 'failed exit=%s %s\n' "$code" "$(date -Is)" > "$STATUS_FILE"
  fi
}
trap on_exit EXIT
printf 'running dual_gpu %s\n' "$(date -Is)" > "$STATUS_FILE"

"$PYTHON" scripts/qa_pipeline.py \
  --model "$MODEL" --dataset "$DATASET" --config "$CONFIG" \
  --prepared-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --output "$OUTPUT" \
  --stage generate --device cuda:1 --generation-devices cuda:1 \
  --feature-devices cuda:0 cuda:1 --checkpoint-every 25 --resume

"$PYTHON" scripts/qa_pipeline.py \
  --model "$MODEL" --dataset "$DATASET" --config "$CONFIG" \
  --prepared-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --output "$OUTPUT" \
  --stage label --device cuda:1 --generation-devices cuda:1 \
  --feature-devices cuda:0 cuda:1 --resume

"$PYTHON" scripts/prepare_qa_sam_svar_subsets.py \
  --source-root "$SOURCE_ROOT" --output-root "$SELECTED_ROOT" \
  --dataset "$DATASET" --yes-only-labels "$RUN_DIR/labels.jsonl"

"$PYTHON" scripts/qa_pipeline.py \
  --model "$MODEL" --dataset "$DATASET" --config "$CONFIG" \
  --prepared-root "$SELECTED_ROOT" --output-root "$OUTPUT_ROOT" --output "$OUTPUT" \
  --stage extract --device cuda:1 --generation-devices cuda:1 \
  --feature-devices cuda:0 cuda:1 --feature-sync-every 16 --resume

"$PYTHON" scripts/export_qa_sam_svar_cross_position.py \
  --questions "$SELECTED_ROOT/$DATASET/questions.jsonl" --run-dir "$RUN_DIR"
echo "[$(date -Is)] dual-GPU Qwen3 AMBER extraction complete"
