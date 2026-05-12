#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL specific environment variables
export IMAGE_MAX_TOKEN_NUM=1280
export IMAGE_MIN_TOKEN_NUM=256
export FPS_MAX_FRAMES=16

export MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
export MODEL_NAME_SHORT="qwen3_vl_8b_instruct"
# Put the best checkpoint path with the highest validation score here
export CHECKPOINT_DIR="output/ChartCF_qwen3_vl_8b_instruct"

bash scripts/eval/run_eval_charxiv.sh

bash scripts/eval/run_eval_chartqa.sh

bash scripts/eval/run_eval_chartbench.sh

bash scripts/eval/run_eval_chartx.sh

bash scripts/eval/run_eval_ecdbench.sh
