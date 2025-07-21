#!/usr/bin/env python3
"""
Memory-Efficient Weight Analysis: SFT vs RLVR
Loads models one at a time to avoid OOM on limited GPU memory
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM
import json
from collections import defaultdict
import pandas as pd
from pathlib import Path
import gc

class MemoryEfficientAnalyzer:
    def __init__(self, base_model_path, sft_model_path, rlvr_model_path, device="cuda"):
        self.device = device
        self.base_model_path = base_model_path
        self.sft_model_path = sft_model_path
        self.rlvr_model_path = rlvr_model_path
        
        # We'll load models one at a time and extract their state dicts
        print("Loading models sequentially to save memory...")
        
    def extract_state_dict(self, model_path, name):
        """Load model, extract state dict, then free memory"""
        print(f"Loading {name} model from {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use fp16 to save memory
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Extract state dict
        state_dict = {k: v.cpu().float() for k, v in model.state_dict().items()}
        
        # Free memory immediately
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✅ {name} model loaded and cached")
        return state_dict
    
    def compute_weight_differences(self):
        """Compute weight differences with memory efficiency"""
        print("Computing weight differences (memory efficient)...")
        
        # Load models one by one
        print("\n1/3: Loading base model...")
        base_state = self.extract_state_dict(self.base_model_path, "Base")
        
        print("\n2/3: Loading SFT model...")
        sft_state = self.extract_state_dict(self.sft_model_path, "SFT")
        
        print("\n3/3: Loading RLVR model...")
        rlvr_state = self.extract_state_dict(self.rlvr_model_path, "RLVR")
        
        print("\nAnalyzing weight differences...")
        
        results = {
            'layer_analysis': {},
            'component_analysis': {},
            'key_findings': []
        }
        
        param_count = 0
        
        # Analyze each parameter
        for param_name in base_state.keys():
            if param_name not in sft_state or param_name not in rlvr_state:
                continue
                
            base_param = base_state[param_name]
            sft_param = sft_state[param_name]
            rlvr_param = rlvr_state[param_name]
            
            # Skip if shapes don't match
            if base_param.shape != sft_param.shape or base_param.shape != rlvr_param.shape:
                continue
            
            # Compute differences
            sft_diff = sft_param - base_param
            rlvr_diff = rlvr_param - base_param
            
            # Analyze this parameter
            self._analyze_parameter(param_name, base_param, sft_diff, rlvr_diff, results)
            
            param_count += 1
            if param_count % 50 == 0:
                print(f"Processed {param_count} parameters...")
        
        print(f"✅ Analyzed {param_count} parameters total")
        return results
    
    def _analyze_parameter(self, param_name, base_param, sft_diff, rlvr_diff, results):
        """Analyze a specific parameter's changes (same as before but more memory efficient)"""
        
        # Extract layer and component info
        layer_info = self._parse_parameter_name(param_name)
        
        if layer_info['layer_num'] is not None:
            layer_key = layer_info['layer_num']
            
            if layer_key not in results['layer_analysis']:
                results['layer_analysis'][layer_key] = {
                    'sft_total_change': 0,
                    'rlvr_total_change': 0,
                    'sft_max_change': 0,
                    'rlvr_max_change': 0,
                    'components': []
                }
            
            # Magnitude analysis (compute on CPU to save GPU memory)
            sft_magnitude = torch.norm(sft_diff).item()
            rlvr_magnitude = torch.norm(rlvr_diff).item()
            base_magnitude = torch.norm(base_param).item()
            
            # Relative changes
            sft_relative = sft_magnitude / (base_magnitude + 1e-8)
            rlvr_relative = rlvr_magnitude / (base_magnitude + 1e-8)
            
            # Direction analysis (cosine similarity)
            sft_flat = sft_diff.flatten()
            rlvr_flat = rlvr_diff.flatten()
            
            # Compute cosine similarity efficiently
            dot_product = torch.dot(sft_flat, rlvr_flat).item()
            sft_norm = torch.norm(sft_flat).item()
            rlvr_norm = torch.norm(rlvr_flat).item()
            
            direction_similarity = dot_product / (sft_norm * rlvr_norm + 1e-8)
            
            # Update results
            results['layer_analysis'][layer_key]['sft_total_change'] += sft_relative
            results['layer_analysis'][layer_key]['rlvr_total_change'] += rlvr_relative
            results['layer_analysis'][layer_key]['sft_max_change'] = max(
                results['layer_analysis'][layer_key]['sft_max_change'], 
                sft_relative
            )
            results['layer_analysis'][layer_key]['rlvr_max_change'] = max(
                results['layer_analysis'][layer_key]['rlvr_max_change'], 
                rlvr_relative
            )
            results['layer_analysis'][layer_key]['components'].append({
                'name': param_name,
                'component_type': layer_info['component'],
                'sft_change': sft_relative,
                'rlvr_change': rlvr_relative,
                'direction_similarity': direction_similarity
            })
    
    def _parse_parameter_name(self, param_name):
        """Parse parameter name to extract layer and component information"""
        parts = param_name.split('.')
        
        layer_num = None
        component = "other"
        
        # Extract layer number
        for part in parts:
            if part.isdigit():
                layer_num = int(part)
                break
        
        # Determine component type
        if 'attn' in param_name or 'attention' in param_name:
            if 'q_proj' in param_name or 'query' in param_name:
                component = "attention_query"
            elif 'k_proj' in param_name or 'key' in param_name:
                component = "attention_key"
            elif 'v_proj' in param_name or 'value' in param_name:
                component = "attention_value"
            elif 'o_proj' in param_name or 'output' in param_name:
                component = "attention_output"
            else:
                component = "attention_other"
        elif 'mlp' in param_name or 'feed_forward' in param_name:
            if 'gate' in param_name:
                component = "mlp_gate"
            elif 'up' in param_name:
                component = "mlp_up"
            elif 'down' in param_name:
                component = "mlp_down"
            else:
                component = "mlp_other"
        elif 'embed' in param_name:
            component = "embedding"
        elif 'norm' in param_name or 'layer_norm' in param_name:
            component = "norm"
        elif 'lm_head' in param_name:
            component = "lm_head"
        
        return {
            'layer_num': layer_num,
            'component': component,
            'full_name': param_name
        }
    
    def generate_quick_insights(self, results):
        """Generate quick insights from the analysis"""
        print("\n" + "="*60)
        print("🔍 QUICK INSIGHTS: SFT vs RLVR Weight Analysis")
        print("="*60)
        
        # Find layers with biggest differences
        if results['layer_analysis']:
            layer_data = []
            for layer_num, data in results['layer_analysis'].items():
                layer_data.append({
                    'layer': layer_num,
                    'sft_change': data['sft_total_change'],
                    'rlvr_change': data['rlvr_total_change'],
                    'ratio': data['rlvr_total_change'] / (data['sft_total_change'] + 1e-8)
                })
            
            # Sort by total change
            layer_data.sort(key=lambda x: x['sft_change'] + x['rlvr_change'], reverse=True)
            
            print("\n📊 Top 5 Most Changed Layers:")
            for i, layer in enumerate(layer_data[:5]):
                print(f"{i+1}. Layer {layer['layer']}: SFT={layer['sft_change']:.4f}, RLVR={layer['rlvr_change']:.4f}")
            
            # Find component patterns
            component_stats = defaultdict(lambda: {'sft': 0, 'rlvr': 0, 'count': 0})
            
            for layer_data in results['layer_analysis'].values():
                for comp in layer_data['components']:
                    comp_type = comp['component_type']
                    component_stats[comp_type]['sft'] += comp['sft_change']
                    component_stats[comp_type]['rlvr'] += comp['rlvr_change']
                    component_stats[comp_type]['count'] += 1
            
            print("\n🧠 Component Analysis:")
            for comp_type, stats in component_stats.items():
                if stats['count'] > 0:
                    avg_sft = stats['sft'] / stats['count']
                    avg_rlvr = stats['rlvr'] / stats['count']
                    print(f"• {comp_type}: SFT={avg_sft:.4f}, RLVR={avg_rlvr:.4f}")
        
        # Save detailed results
        with open("weight_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Detailed results saved to 'weight_analysis_results.json'")
        
        return results

def main():
    """Main analysis function - memory efficient version"""
    print("🔬 Memory-Efficient Weight Analysis: SFT vs RLVR")
    print("="*60)
    
    # Model paths
    base_model = "meta-llama/Llama-3.1-8B"
    sft_model = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    rlvr_model = "allenai/Llama-3.1-Tulu-3-8B"
    
    analyzer = MemoryEfficientAnalyzer(base_model, sft_model, rlvr_model)
    
    print("\n🔄 Computing weight differences...")
    results = analyzer.compute_weight_differences()
    
    print("\n📊 Generating insights...")
    analyzer.generate_quick_insights(results)
    
    print(f"\n🎉 Analysis complete!")
    print(f"💡 This shows you exactly how SFT vs RLVR learning differs at the weight level!")

if __name__ == "__main__":
    main() 