export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_NAME="OpenGVLab/InternVL3_5-8B-Instruct"

GLOBAL_BATCH_SIZE=64
BATCH_PER_DEVICE=2
NUM_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
SEED=0

DATA_PATH="data/ECD/syn_data/dpo_data_keep40.json"
OUTPUT_DIR="output/internvl3_5_8b_instruct/dpo_data_keep40"

mkdir -p $OUTPUT_DIR

NPROC_PER_NODE=$NUM_DEVICES \
swift rlhf \
    --rlhf_type dpo \
    --model_type internvl3_5 \
    --model $MODEL_NAME \
    --use_hf true \
    --dataset $DATA_PATH \
    --split_dataset_ratio 0.05 \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --use_dora false \
    --use_liger_kernel true \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 1e-4 \
    --aligner_lr 0.0 \
    --vit_lr 0.0 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --rpo_alpha 1.0 \
    --lr_scheduler_type cosine \
    --max_pixels $((1280 * 28 * 28)) \
    --logging_steps 1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 2 \
    --metric_for_best_model loss \
    --attn_impl flash_attention_2 \
    --gradient_checkpointing true \
    --dataloader_num_workers 4 \
    --max_length 8192 \
    --truncation_strategy delete \
    --logging_dir ${OUTPUT_DIR}/logs \
    --save_only_model true \
    --seed $SEED \
    --data_seed $SEED \
    --strict true \
    --create_checkpoint_symlink true