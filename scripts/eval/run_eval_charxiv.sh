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

echo "Using model: $MODEL_NAME"
echo "Using checkpoint: $CHECKPOINT_DIR"

SPLIT="val"
BATCH_SIZE=64
MAX_NEW_TOKENS=512
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

CHARXIV_DATA_DIR="data/public_benchmarks/CharXiv/data"
CHARXIV_IMAGE_DIR="data/public_benchmarks/CharXiv/images"
SWIFT_DATA_DIR="data/public_benchmarks/CharXiv/swift_data"
GPT_EVAL_WORKERS=16
GRADING_MODEL="gpt-4o"

SAVE_NAME="${CHECKPOINT_DIR}/CharXiv"

mkdir -p "$SAVE_NAME" "$SWIFT_DATA_DIR"

for MODE in descriptive reasoning; do
    python src/chartcf/evaluation/CharXiv/prepare_swift_dataset.py \
        --split "$SPLIT" \
        --mode "$MODE" \
        --data_dir "$CHARXIV_DATA_DIR" \
        --image_dir "$CHARXIV_IMAGE_DIR" \
        --output_dir "$SWIFT_DATA_DIR"

    SWIFT_DATASET="${SWIFT_DATA_DIR}/${MODE}_${SPLIT}.json"
    SWIFT_OUTPUT="${SAVE_NAME}/swift_output_${MODE}_${SPLIT}.jsonl"
    FINAL_OUTPUT="${SAVE_NAME}/gen-${MODEL_NAME_SHORT}-${MODE}_${SPLIT}.json"

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

    python src/chartcf/evaluation/CharXiv/convert_swift_results.py \
        --swift_output "$SWIFT_OUTPUT" \
        --dataset "$SWIFT_DATASET" \
        --output "$FINAL_OUTPUT" \
        --mode "$MODE"

    python src/chartcf/evaluation/CharXiv/eval_utils/evaluate_parallel.py \
        --model_name "$MODEL_NAME_SHORT" \
        --split "$SPLIT" \
        --mode "$MODE" \
        --output_name "$SAVE_NAME" \
        --num_workers "$GPT_EVAL_WORKERS" \
        --grading_model "$GRADING_MODEL" \
        --data_dir "$CHARXIV_DATA_DIR"
done

python src/chartcf/evaluation/CharXiv/eval_utils/get_stats.py \
    --model_name "$MODEL_NAME_SHORT" \
    --split "$SPLIT" \
    --output_name "$SAVE_NAME" \
    --data_dir "$CHARXIV_DATA_DIR"
