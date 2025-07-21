# SFT vs RLVR Training Comparison Experiment

This experiment compares **Supervised Fine-Tuning (SFT)** vs **Reinforcement Learning with Verifiable Rewards (RLVR)** on the same base model and dataset to quantify how RL learns differently from standard supervised learning.

## Experiment Design

### Base Model
- **Llama 3.1 8B** (`meta-llama/Llama-3.1-8B`)
- Chosen for optimal balance of performance vs compute requirements
- Well-documented in open-instruct with proven configurations

### Dataset
- **GSM8K** (`ai2-adapt-dev/rlvr_gsm8k_zs`)
- Mathematical reasoning problems ideal for comparing learning approaches
- RLVR particularly shines on reasoning tasks where verifiable rewards help

### Experimental Setup
- **Same seed (42)** for reproducible comparison
- **Same base model** to isolate training methodology effects  
- **Same dataset** to ensure fair comparison
- **W&B tracking** for comprehensive monitoring and comparison

## Hardware Requirements

### Budget Configuration (4 GPUs)
- **4 GPUs** (A40/A100 recommended for budget efficiency)
- **40-80GB+ VRAM total** (sufficient for Llama 3.1 8B)
- **Fast storage** for dataset caching
- **Estimated cost**: ~$150-250 total

### Prime Intellect Setup
When creating your cluster on Prime Intellect:
1. Select **4 GPUs** (A40/A100 for budget efficiency)
2. Choose **Ubuntu 22.04** with CUDA 12+
3. Select **Cheapest** location and security as needed
4. Clone this repo: `git clone https://github.com/tashapais/open-instruct`

## Pre-Training Setup

### 1. Environment Setup
```bash
# Install dependencies
pip install --upgrade pip "setuptools<70.0.0" wheel 
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install packaging
pip install flash-attn==2.7.2.post2 flashinfer-python>=0.2.7.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt
```

### 2. Weights & Biases Setup
```bash
# Install and login to W&B
pip install wandb
wandb login  # Enter your API key from https://wandb.ai/authorize
```

### 3. Verify GPU Setup
```bash
nvidia-smi  # Should show 8 GPUs
```

## Running the Experiment

### Phase 1: SFT Training
```bash
./experiment_sft_llama8b.sh
```

**Expected Runtime:** ~12-16 hours on 4x A100 GPUs

**What it does:**
- Trains Llama 3.1 8B using standard supervised fine-tuning
- Uses cross-entropy loss on ground truth answers
- Learns direct input→output mapping

### Phase 2: RLVR Training  
```bash
./experiment_rlvr_llama8b.sh
```

**Expected Runtime:** ~16-24 hours on 4x A100 GPUs

**What it does:**
- Trains using GRPO (Generalized Reinforcement Learning from Policy Optimization)
- Uses verifiable rewards based on answer correctness
- Learns through policy gradient optimization and exploration

### Phase 3: Evaluation
```bash
./evaluate_comparison.sh
```

**What it does:**
- Evaluates both models on GSM8K test set
- Compares accuracy and solution quality
- Generates comparison report

## Monitoring Training

### Weights & Biases Dashboard
Navigate to: `https://wandb.ai/tashapais/sft-vs-rlvr-comparison`

**Key Metrics to Watch:**

#### SFT Training:
- **Training Loss**: Should decrease smoothly
- **Learning Rate**: Linear warmup then decay
- **GPU Utilization**: Should be consistently high
- **Tokens/sec**: Throughput metric

#### RLVR Training:
- **Policy Loss**: May be more volatile due to RL nature
- **Advantage**: Should show learning signal
- **Reward**: Should generally increase
- **KL Divergence**: Monitors policy deviation
- **Value Loss**: Critic learning progress

## Expected Differences

### Learning Dynamics
- **SFT**: Smooth, predictable loss curves
- **RLVR**: More volatile, exploration-driven learning

### Solution Quality
- **SFT**: Learns to mimic training examples
- **RLVR**: Learns to maximize reward, may find novel solutions

### Reasoning Patterns
- **SFT**: Reproduces reasoning patterns from training data
- **RLVR**: May develop more robust reasoning through trial-and-error

### Performance
- **SFT**: Typically faster convergence
- **RLVR**: Often better final performance on reasoning tasks

## Analysis Framework

After both training runs complete, analyze:

### 1. Quantitative Metrics
```python
# Compare final accuracies
sft_accuracy = load_results("evaluation_results_*/sft_results.json")
rlvr_accuracy = load_results("evaluation_results_*/rlvr_results.json")
print(f"SFT Accuracy: {sft_accuracy:.2%}")
print(f"RLVR Accuracy: {rlvr_accuracy:.2%}")
print(f"RLVR Improvement: {(rlvr_accuracy - sft_accuracy):.2%}")
```

### 2. Learning Efficiency
- **Training time**: RLVR typically takes longer
- **Sample efficiency**: Steps to reach target performance
- **Compute efficiency**: FLOPs per performance point

### 3. Qualitative Analysis
- **Solution diversity**: RLVR may show more varied approaches
- **Error patterns**: Different failure modes between methods
- **Reasoning robustness**: Test on out-of-distribution problems

## Troubleshooting

### Common Issues

#### OOM (Out of Memory)
```bash
# Reduce batch size in scripts
--per_device_train_batch_size 1  # Already optimized
--gradient_accumulation_steps 1  # Reduce if needed
```

#### Slow Training
```bash
# Enable optimizations
--gradient_checkpointing  # Already enabled
--use_flash_attn  # Already enabled
```

#### WANDB Issues
```bash
# Re-login if tracking fails
wandb login --relogin
export WANDB_API_KEY="your_key_here"
```

### Hardware-Specific Adjustments

#### Scaling to 8 GPUs (if budget allows)
```bash
# Modify CUDA_VISIBLE_DEVICES in scripts
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Update num_processes in accelerate launch
--num_processes 8
# Reduce gradient accumulation
--gradient_accumulation_steps 2
# For RLVR: increase num_learners_per_node to 6
# For RLVR: increase vllm_num_engines to 2
```

#### Different GPU Memory
```bash
# For <24GB VRAM per GPU:
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
--vllm_gpu_memory_utilization 0.7  # Reduce for RLVR
```

## Expected Research Insights

This experiment will help quantify:

1. **Learning Pattern Differences**: How RL exploration differs from supervised imitation
2. **Sample Efficiency**: Whether RL needs more/fewer examples to reach performance
3. **Generalization**: Which approach better handles unseen problem types
4. **Solution Diversity**: Whether RL discovers multiple solution paths
5. **Training Stability**: Convergence behavior comparison

## Next Steps

After completing this experiment, consider:

1. **Different Datasets**: Try on other reasoning tasks (MATH, Code)
2. **Architecture Variations**: Compare on different model sizes
3. **Hybrid Approaches**: SFT followed by RLVR fine-tuning
4. **Reward Engineering**: Experiment with different reward functions

## Citation

If you use this experimental setup, please cite:
```bibtex
@article{tulu3,
  title={TÜLU 3: Pushing Frontiers in Open Language Model Post-Training},
  author={...},
  journal={arXiv preprint arXiv:2411.15124},
  year={2024}
}
```

## Support

- **Open-Instruct Issues**: https://github.com/allenai/open-instruct/issues
- **Prime Intellect Support**: Check their documentation
- **W&B Help**: https://docs.wandb.ai/ 