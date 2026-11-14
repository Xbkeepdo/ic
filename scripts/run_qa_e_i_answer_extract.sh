#!/usr/bin/env bash
set -euo pipefail

model="$1"
device="$2"
root="/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector"
prepared="$root/outputs/qa_benchmarks/LOGS-K32-TAU003-EI-ANSWER/prepared/$model"
cd "$root"

for dataset in clevr_exist_9k pope amber_discriminative; do
    /opt/conda/private/envs/vicr/bin/python scripts/qa_pipeline.py \
        --model "$model" \
        --dataset "$dataset" \
        --config configs/model_configs_qa_logs_e_i_answer.yaml \
        --prepared-root "$prepared" \
        --output LOGS-K32-TAU003-EI-ANSWER \
        --stage extract \
        --device "$device"
done
