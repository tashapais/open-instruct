#!/bin/bash

# Evaluation Script for SFT vs RLVR Comparison
# Evaluates both models on GSM8K and MATH benchmarks

echo "===== SFT vs RLVR Model Comparison Evaluation ====="

# Set paths to your trained models (4-GPU Budget Version)
SFT_MODEL_PATH="./output/sft_llama8b_gsm8k_4gpu_*"
RLVR_MODEL_PATH="./output/rlvr_llama8b_gsm8k_4gpu_*"

# Find the latest model directories
SFT_MODEL=$(ls -td ${SFT_MODEL_PATH} | head -1)
RLVR_MODEL=$(ls -td ${RLVR_MODEL_PATH} | head -1)

echo "SFT Model: ${SFT_MODEL}"
echo "RLVR Model: ${RLVR_MODEL}"

# Create evaluation results directory
EVAL_DIR="./evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${EVAL_DIR}

echo "Evaluation results will be saved to: ${EVAL_DIR}"

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    local model_type=$2
    local output_file="${EVAL_DIR}/${model_type}_results.json"
    
    echo "Evaluating ${model_type} model..."
    
    # GSM8K Evaluation
    python eval/gsm/run_eval.py \
        --data_path eval/gsm/test.jsonl \
        --model_name_or_path ${model_path} \
        --tokenizer_name ${model_path} \
        --eval_batch_size 4 \
        --load_in_8bit \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        --save_path ${output_file}
        
    echo "${model_type} evaluation completed. Results saved to ${output_file}"
}

# Evaluate both models
if [ -d "${SFT_MODEL}" ]; then
    run_evaluation ${SFT_MODEL} "sft"
else
    echo "SFT model not found at ${SFT_MODEL_PATH}"
fi

if [ -d "${RLVR_MODEL}" ]; then
    run_evaluation ${RLVR_MODEL} "rlvr"
else
    echo "RLVR model not found at ${RLVR_MODEL_PATH}"
fi

# Compare results
echo ""
echo "===== COMPARISON SUMMARY ====="
echo "Check the following files for detailed results:"
echo "- SFT Results: ${EVAL_DIR}/sft_results.json"
echo "- RLVR Results: ${EVAL_DIR}/rlvr_results.json"
echo ""
echo "Key metrics to compare:"
echo "1. Accuracy on GSM8K problems"
echo "2. Reasoning quality in generated solutions"
echo "3. Training efficiency (check W&B logs)"
echo "4. Convergence behavior"
echo ""
echo "W&B Dashboard: https://wandb.ai/tashapais/sft-vs-rlvr-comparison" 