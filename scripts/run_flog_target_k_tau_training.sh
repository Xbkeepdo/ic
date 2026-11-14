#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector
GF_ROOT=/home/apulis-dev/userdata/CODEX/test-cocochair/gfchair
OUT="$ROOT/outputs/flog_target_k_tau_ablation"
GF_OUT="$GF_ROOT/outputs/coco4000_512_flog_target_k_tau_ablation_811"
PY=/opt/conda/private/envs/vicr/bin/python

mkdir -p "$OUT"
cd "$ROOT"

set_status() {
  cat > "$OUT/progress.md" <<EOF
# F_log target K/tau 消融进度

- 阶段：$1
- 模型：LLaVA-1.5-7B、Qwen2.5-VL-7B、InternVL2.5-8B、Qwen3-VL-8B
- 网格：K ∈ {32, 64}，tau ∈ {0.03, 0.05, 0.07}
- 训练：Risk-only 和 Risk + F_log，各 3 seeds
- 提取进度：见 gfchair 对应模型的 progress.json；训练进度：见本目录各模型的 progress.json 和 test_grid_progress.json
EOF
}

set_status "等待四模型特征提取完成"
while true; do
  ready=1
  for model in qwen2_5_vl_7b llava_1_5_7b qwen3_vl_8b internvl_2_5_8b; do
    if [[ ! -f "$GF_OUT/$model/summary.json" ]]; then
      ready=0
    fi
  done
  if [[ "$ready" -eq 1 ]]; then
    break
  fi
  sleep 30
done

set_status "内部验证：6 组 × 2 种输入 × 3 seeds"
(
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model qwen3_vl_8b --device cuda:0 --stage validation
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model qwen2_5_vl_7b --device cuda:0 --stage validation
) > "$OUT/gpu0.validation.log" 2>&1 &
pid0=$!
(
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model internvl_2_5_8b --device cuda:1 --stage validation
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model llava_1_5_7b --device cuda:1 --stage validation
) > "$OUT/gpu1.validation.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PY" scripts/train_flog_target_k_tau_ablation.py --select-shared
set_status "固定测试集网格：6 组 × 2 种输入 × 3 seeds"
(
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model qwen3_vl_8b --device cuda:0 --stage test-grid
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model qwen2_5_vl_7b --device cuda:0 --stage test-grid
) > "$OUT/gpu0.test_grid.log" 2>&1 &
pid0=$!
(
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model internvl_2_5_8b --device cuda:1 --stage test-grid
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model llava_1_5_7b --device cuda:1 --stage test-grid
) > "$OUT/gpu1.test_grid.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

set_status "汇总结果"
"$PY" scripts/train_flog_target_k_tau_ablation.py --summarize-grid
for model in qwen2_5_vl_7b llava_1_5_7b qwen3_vl_8b internvl_2_5_8b; do
  "$PY" scripts/train_flog_target_k_tau_ablation.py --model "$model" --stage final --device cpu
done
"$PY" scripts/train_flog_target_k_tau_ablation.py --summarize-final
set_status "完成；见 complete_results.md、summary.md、shared_selection.md"
