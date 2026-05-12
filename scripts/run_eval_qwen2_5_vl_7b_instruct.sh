#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
export MODEL_NAME_SHORT="qwen2_5_vl_7b_instruct"
# Put the best checkpoint path with the highest validation score here
export CHECKPOINT_DIR="output/ChartCF_qwen2_5_vl_7b_instruct"

bash scripts/eval/run_eval_charxiv.sh

bash scripts/eval/run_eval_chartqa.sh

bash scripts/eval/run_eval_chartbench.sh

bash scripts/eval/run_eval_chartx.sh

bash scripts/eval/run_eval_ecdbench.sh
