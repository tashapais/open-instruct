#!/bin/bash

# RLVR (GRPO) Training Script for Llama 3.1 8B on GSM8K
# Experiment: Comparing SFT vs RLVR learning

export WANDB_PROJECT="sft-vs-rlvr-comparison"
export WANDB_RUN_GROUP="llama-8b-gsm8k"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training hyperparameters (4-GPU Budget Version)
MODEL_NAME="meta-llama/Llama-3.1-8B"
DATASET="ai2-adapt-dev/rlvr_gsm8k_zs"
EXP_NAME="rlvr_llama8b_gsm8k_4gpu_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./output/${EXP_NAME}"

echo "Starting RLVR training experiment: ${EXP_NAME}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run RLVR training with GRPO
python open_instruct/grpo_fast.py \
    --exp_name ${EXP_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_mixer_list ${DATASET} 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ${DATASET} 32 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 1024 \
    --response_length 1024 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 32 \
    --num_samples_per_prompt_rollout 4 \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --temperature 0.7 \
    --ground_truths_key ground_truth \
    --chat_template_name r1_simple_chat_postpend_think \
    --learning_rate 3e-7 \
    --total_episodes 10000 \
    --deepspeed_stage 2 \
    --num_epochs 1 \
    --num_learners_per_node 3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 1 \
    --beta 0.01 \
    --seed 42 \
    --num_evals 50 \
    --save_freq 500 \
    --vllm_sync_backend nccl \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --with_tracking \
    --wandb_project_name ${WANDB_PROJECT}

echo "RLVR training completed. Model saved to: ${OUTPUT_DIR}"
echo "Check W&B for training metrics: https://wandb.ai/tashapais/${WANDB_PROJECT}" 