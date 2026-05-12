#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME_SHORT="${MODEL_NAME_SHORT:-qwen2_5_vl_7b_instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: CHECKPOINT_DIR is not set. Please set it to the path of the best checkpoint."
    exit 1
fi

BATCH_SIZE=16
MAX_NEW_TOKENS=512
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
GPT_EVAL_WORKERS=16
GRADING_MODEL="gpt-4o"

ECDBENCH_DATA_PATH="data/public_benchmarks/ECDBench/ECD_Bench_All.json"
ECDBENCH_IMAGE_DIR="data/public_benchmarks/ECDBench/rendered_images"
SWIFT_DATA_DIR="data/public_benchmarks/ECDBench/swift_data"
SWIFT_DATASET="${SWIFT_DATA_DIR}/test.json"
SAVE_NAME="${CHECKPOINT_DIR}/ECDBench"
SWIFT_OUTPUT="${SAVE_NAME}/swift_output_test.jsonl"
FINAL_OUTPUT="${SAVE_NAME}/gen-${MODEL_NAME_SHORT}-test.json"

mkdir -p "$SAVE_NAME" "$SWIFT_DATA_DIR"

python src/chartcf/evaluation/ECDBench/prepare_swift_dataset.py \
    --input_file "$ECDBENCH_DATA_PATH" \
    --image_dir "$ECDBENCH_IMAGE_DIR" \
    --output_dir "$SWIFT_DATA_DIR"

MASTER_PORT=29511 \
NPROC_PER_NODE=$NUM_DEVICES \
swift infer \
    --use_hf true \
    --max_pixels $((1280 * 28 * 28)) \
    --model "$MODEL_NAME" \
    --adapters "$CHECKPOINT_DIR" \
    --val_dataset "$SWIFT_DATASET" \
    --infer_backend pt \
    --max_batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --stream false \
    --result_path "$SWIFT_OUTPUT"

python src/chartcf/evaluation/ECDBench/convert_swift_results.py \
    --swift_output "$SWIFT_OUTPUT" \
    --dataset "$SWIFT_DATASET" \
    --output "$FINAL_OUTPUT"

python src/chartcf/evaluation/ECDBench/evaluate.py \
    --infer_data_path "$FINAL_OUTPUT" \
    --output_file "${SAVE_NAME}/scores-${MODEL_NAME_SHORT}-test.json" \
    --eval_model_name "$GRADING_MODEL" \
    --max_concurrent "$GPT_EVAL_WORKERS" \
    --log_file "${SAVE_NAME}/stats-${MODEL_NAME_SHORT}-test.log"
