#!/usr/bin/env python3
"""
Weight Difference Analysis: SFT vs RLVR
Analyzes how weights change differently between supervised and reinforcement learning approaches.

This script compares the weight changes between:
1. Base → SFT: How supervised learning modifies weights
2. SFT → RLVR: How reinforcement learning further modifies weights  
3. Base → RLVR: Total effect of the combined pipeline

Models analyzed:
- Base: meta-llama/Llama-3.1-8B
- SFT: allenai/Llama-3.1-Tulu-3-8B-SFT  
- RLVR: allenai/Llama-3.1-Tulu-3-8B
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM
import pandas as pd
from collections import defaultdict
import argparse
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_weights(model_name, device='cpu'):
    """Load model and return state dict"""
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    return model.state_dict()

def compute_weight_statistics(weights_dict, layer_pattern=None):
    """Compute statistics for weight tensors"""
    stats = {}
    
    for name, tensor in weights_dict.items():
        if layer_pattern and layer_pattern not in name:
            continue
            
        if tensor.dtype == torch.float32:
            flat_weights = tensor.flatten().cpu().numpy()
            stats[name] = {
                'mean': np.mean(flat_weights),
                'std': np.std(flat_weights),
                'norm': np.linalg.norm(flat_weights),
                'min': np.min(flat_weights),
                'max': np.max(flat_weights),
                'shape': tensor.shape,
                'size': tensor.numel()
            }
    
    return stats

def compute_weight_differences(base_weights, target_weights, normalize=True):
    """Compute element-wise differences between weight dictionaries"""
    differences = {}
    
    for name in base_weights.keys():
        if name in target_weights:
            base_tensor = base_weights[name]
            target_tensor = target_weights[name]
            
            if base_tensor.shape == target_tensor.shape:
                diff = target_tensor - base_tensor
                
                if normalize:
                    # Normalize by base weight magnitude to get relative change
                    base_norm = torch.norm(base_tensor)
                    if base_norm > 1e-8:  # Avoid division by zero
                        diff = diff / base_norm
                
                differences[name] = diff
    
    return differences

def analyze_layer_changes(differences_dict, layer_types=['attention', 'mlp', 'embed', 'norm']):
    """Analyze changes by layer type"""
    layer_stats = defaultdict(list)
    
    for name, diff_tensor in differences_dict.items():
        flat_diff = diff_tensor.flatten().cpu().numpy()
        norm_change = np.linalg.norm(flat_diff)
        mean_abs_change = np.mean(np.abs(flat_diff))
        
        # Categorize by layer type
        layer_type = 'other'
        for ltype in layer_types:
            if ltype in name.lower() or any(keyword in name.lower() for keyword in {
                'attention': ['attn', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'],
                'mlp': ['mlp', 'gate_proj', 'up_proj', 'down_proj', 'fc'],
                'embed': ['embed', 'wte', 'wpe'],
                'norm': ['norm', 'ln']
            }.get(ltype, [ltype])):
                layer_type = ltype
                break
        
        layer_stats[layer_type].append({
            'name': name,
            'norm_change': norm_change,
            'mean_abs_change': mean_abs_change,
            'std_change': np.std(flat_diff)
        })
    
    return layer_stats

def create_comparison_plots(sft_diffs, rlvr_diffs, output_dir='./weight_analysis_plots'):
    """Create visualization plots comparing SFT vs RLVR changes"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analyze layer changes for both approaches
    sft_layer_stats = analyze_layer_changes(sft_diffs)
    rlvr_layer_stats = analyze_layer_changes(rlvr_diffs)
    
    # 1. Layer-wise comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SFT vs RLVR: Layer-wise Weight Changes', fontsize=16)
    
    # Prepare data for plotting
    layer_types = ['attention', 'mlp', 'embed', 'norm']
    sft_norms = []
    rlvr_norms = []
    
    for layer_type in layer_types:
        sft_layer_norms = [stat['norm_change'] for stat in sft_layer_stats.get(layer_type, [])]
        rlvr_layer_norms = [stat['norm_change'] for stat in rlvr_layer_stats.get(layer_type, [])]
        
        sft_norms.append(np.mean(sft_layer_norms) if sft_layer_norms else 0)
        rlvr_norms.append(np.mean(rlvr_layer_norms) if rlvr_layer_norms else 0)
    
    # Bar plot comparison
    x = np.arange(len(layer_types))
    width = 0.35
    
    axes[0,0].bar(x - width/2, sft_norms, width, label='SFT (Base→SFT)', alpha=0.8)
    axes[0,0].bar(x + width/2, rlvr_norms, width, label='RLVR (SFT→RLVR)', alpha=0.8)
    axes[0,0].set_xlabel('Layer Type')
    axes[0,0].set_ylabel('Average Norm Change')
    axes[0,0].set_title('Weight Change Magnitude by Layer Type')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(layer_types)
    axes[0,0].legend()
    
    # 2. Distribution of changes
    all_sft_changes = np.concatenate([diff.flatten().cpu().numpy() for diff in sft_diffs.values()])
    all_rlvr_changes = np.concatenate([diff.flatten().cpu().numpy() for diff in rlvr_diffs.values()])
    
    axes[0,1].hist(all_sft_changes, bins=100, alpha=0.7, label='SFT Changes', density=True)
    axes[0,1].hist(all_rlvr_changes, bins=100, alpha=0.7, label='RLVR Changes', density=True)
    axes[0,1].set_xlabel('Weight Change (Normalized)')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Distribution of Weight Changes')
    axes[0,1].legend()
    axes[0,1].set_xlim(-0.1, 0.1)  # Focus on main distribution
    
    # 3. Correlation analysis
    common_layers = set(sft_diffs.keys()) & set(rlvr_diffs.keys())
    sft_norms_common = []
    rlvr_norms_common = []
    
    for layer in common_layers:
        sft_norms_common.append(torch.norm(sft_diffs[layer]).item())
        rlvr_norms_common.append(torch.norm(rlvr_diffs[layer]).item())
    
    axes[1,0].scatter(sft_norms_common, rlvr_norms_common, alpha=0.6)
    axes[1,0].set_xlabel('SFT Change Magnitude')
    axes[1,0].set_ylabel('RLVR Change Magnitude')
    axes[1,0].set_title('SFT vs RLVR Change Correlation')
    
    # Add diagonal line
    max_val = max(max(sft_norms_common), max(rlvr_norms_common))
    axes[1,0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    # 4. Top changing layers
    # Get top 10 most changed layers for each approach
    sft_top_changes = sorted([(name, torch.norm(diff).item()) for name, diff in sft_diffs.items()], 
                           key=lambda x: x[1], reverse=True)[:10]
    rlvr_top_changes = sorted([(name, torch.norm(diff).item()) for name, diff in rlvr_diffs.items()], 
                            key=lambda x: x[1], reverse=True)[:10]
    
    # Create a combined view
    all_top_layers = list(set([x[0] for x in sft_top_changes] + [x[0] for x in rlvr_top_changes]))
    
    sft_dict = dict(sft_top_changes)
    rlvr_dict = dict(rlvr_top_changes)
    
    combined_data = []
    for layer in all_top_layers[:15]:  # Top 15 for visibility
        combined_data.append({
            'layer': layer.split('.')[-1],  # Just the layer name
            'SFT': sft_dict.get(layer, 0),
            'RLVR': rlvr_dict.get(layer, 0)
        })
    
    df = pd.DataFrame(combined_data)
    df_melted = df.melt(id_vars=['layer'], var_name='Method', value_name='Change')
    
    sns.barplot(data=df_melted, x='layer', y='Change', hue='Method', ax=axes[1,1])
    axes[1,1].set_title('Top Changing Layers')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sft_vs_rlvr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir / 'sft_vs_rlvr_comparison.png'}")
    
    return sft_layer_stats, rlvr_layer_stats

def generate_analysis_report(sft_stats, rlvr_stats, sft_layer_stats, rlvr_layer_stats, output_dir='./weight_analysis_plots'):
    """Generate a detailed analysis report"""
    output_dir = Path(output_dir)
    report_path = output_dir / 'weight_analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# SFT vs RLVR Weight Analysis Report\n\n")
        
        f.write("## Executive Summary\n")
        f.write("This report analyzes the weight differences between supervised fine-tuning (SFT) ")
        f.write("and reinforcement learning with verifiable rewards (RLVR) on Llama 3.1 8B.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Compare overall change magnitudes
        sft_total_change = np.mean([stat['norm'] for stat in sft_stats.values()])
        rlvr_total_change = np.mean([stat['norm'] for stat in rlvr_stats.values()])
        
        f.write(f"- **SFT Average Weight Change**: {sft_total_change:.6f}\n")
        f.write(f"- **RLVR Average Weight Change**: {rlvr_total_change:.6f}\n")
        f.write(f"- **Ratio (RLVR/SFT)**: {rlvr_total_change/sft_total_change:.2f}\n\n")
        
        # Layer-wise analysis
        f.write("## Layer-wise Analysis\n\n")
        
        for layer_type in ['attention', 'mlp', 'embed', 'norm']:
            sft_layer_norms = [stat['norm_change'] for stat in sft_layer_stats.get(layer_type, [])]
            rlvr_layer_norms = [stat['norm_change'] for stat in rlvr_layer_stats.get(layer_type, [])]
            
            if sft_layer_norms and rlvr_layer_norms:
                f.write(f"### {layer_type.capitalize()} Layers\n")
                f.write(f"- SFT avg change: {np.mean(sft_layer_norms):.6f}\n")
                f.write(f"- RLVR avg change: {np.mean(rlvr_layer_norms):.6f}\n")
                f.write(f"- Layers affected: {len(sft_layer_norms)}\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("1. **Learning Patterns**: Compare the magnitude and distribution patterns.\n")
        f.write("2. **Layer Sensitivity**: Which layer types are most affected by each approach?\n")
        f.write("3. **Precision vs Exploration**: SFT focuses on precision, RLVR on exploration.\n")
        f.write("4. **Correlation**: How do the changes relate between methods?\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("- Analyze specific high-change layers in detail\n")
        f.write("- Test on specific mathematical reasoning problems\n")
        f.write("- Compare activation patterns on the same inputs\n")
    
    print(f"Generated detailed report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze weight differences between SFT and RLVR models')
    parser.add_argument('--device', default='cpu', help='Device to load models on')
    parser.add_argument('--output_dir', default='./weight_analysis_plots', help='Output directory for plots and reports')
    parser.add_argument('--layer_pattern', default=None, help='Filter layers by pattern (e.g., "layers.0" for first layer)')
    args = parser.parse_args()
    
    # Model definitions
    base_model = "meta-llama/Llama-3.1-8B"
    sft_model = "allenai/Llama-3.1-Tulu-3-8B-SFT" 
    rlvr_model = "allenai/Llama-3.1-Tulu-3-8B"
    
    print("🔍 Weight Difference Analysis: SFT vs RLVR")
    print("=" * 50)
    
    # Load all models
    try:
        base_weights = load_model_weights(base_model, args.device)
        sft_weights = load_model_weights(sft_model, args.device)
        rlvr_weights = load_model_weights(rlvr_model, args.device)
        
        print("\n📊 Computing weight differences...")
        
        # Compute differences
        sft_diffs = compute_weight_differences(base_weights, sft_weights, normalize=True)
        rlvr_from_sft_diffs = compute_weight_differences(sft_weights, rlvr_weights, normalize=True)
        
        print(f"Analyzed {len(sft_diffs)} layers for SFT changes")
        print(f"Analyzed {len(rlvr_from_sft_diffs)} layers for RLVR changes")
        
        # Compute statistics
        sft_stats = compute_weight_statistics(sft_diffs, args.layer_pattern)
        rlvr_stats = compute_weight_statistics(rlvr_from_sft_diffs, args.layer_pattern)
        
        # Create plots and analysis
        print("\n📈 Generating visualizations...")
        sft_layer_stats, rlvr_layer_stats = create_comparison_plots(
            sft_diffs, rlvr_from_sft_diffs, args.output_dir
        )
        
        # Generate report
        print("\n📝 Generating analysis report...")
        generate_analysis_report(
            sft_stats, rlvr_stats, sft_layer_stats, rlvr_layer_stats, args.output_dir
        )
        
        print(f"\n✅ Analysis complete! Check {args.output_dir} for results.")
        
        # Quick summary
        print("\n🔍 Quick Summary:")
        avg_sft_change = np.mean([torch.norm(diff).item() for diff in sft_diffs.values()])
        avg_rlvr_change = np.mean([torch.norm(diff).item() for diff in rlvr_from_sft_diffs.values()])
        
        print(f"Average SFT weight change magnitude: {avg_sft_change:.6f}")
        print(f"Average RLVR weight change magnitude: {avg_rlvr_change:.6f}")
        print(f"RLVR changes are {avg_rlvr_change/avg_sft_change:.2f}x the magnitude of SFT changes")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Note: This requires significant RAM. Consider using a smaller model or GPU.")

if __name__ == "__main__":
    main() 