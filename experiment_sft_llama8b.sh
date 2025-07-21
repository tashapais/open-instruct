#!/bin/bash

# SFT Training Script for Llama 3.1 8B on GSM8K
# Experiment: Comparing SFT vs RLVR learning

export WANDB_PROJECT="sft-vs-rlvr-comparison"
export WANDB_RUN_GROUP="llama-8b-gsm8k"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training hyperparameters (4-GPU Budget Version)
MODEL_NAME="meta-llama/Llama-3.1-8B"
DATASET="ai2-adapt-dev/rlvr_gsm8k_zs"
EXP_NAME="sft_llama8b_gsm8k_4gpu_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./output/${EXP_NAME}"

echo "Starting SFT training experiment: ${EXP_NAME}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run SFT training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --exp_name ${EXP_NAME} \
    --model_name_or_path ${MODEL_NAME} \
    --tokenizer_name ${MODEL_NAME} \
    --use_slow_tokenizer \
    --use_flash_attn \
    --dataset_mixer_list ${DATASET} 1.0 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir ${OUTPUT_DIR} \
    --with_tracking \
    --wandb_project_name ${WANDB_PROJECT} \
    --report_to wandb \
    --logging_steps 10 \
    --seed 42 \
    --gradient_checkpointing \
    --checkpointing_steps 500 \
    --keep_last_n_checkpoints 2

echo "SFT training completed. Model saved to: ${OUTPUT_DIR}"
echo "Check W&B for training metrics: https://wandb.ai/tashapais/${WANDB_PROJECT}" 