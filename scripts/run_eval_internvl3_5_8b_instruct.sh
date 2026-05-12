#!/usr/bin/env bash
set -euo pipefail

export MODEL_NAME="OpenGVLab/InternVL3_5-8B-Instruct"
export MODEL_NAME_SHORT="internvl3_5_8b_instruct"
# Put the best checkpoint path with the highest validation score here
export CHECKPOINT_DIR="output/ChartCF_internvl3_5_8b_instruct"

bash scripts/eval/run_eval_charxiv.sh

bash scripts/eval/run_eval_chartqa.sh

bash scripts/eval/run_eval_chartbench.sh

bash scripts/eval/run_eval_chartx.sh

bash scripts/eval/run_eval_ecdbench.sh
