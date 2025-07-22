#!/usr/bin/env python3
"""
Weight-Only Analysis: SFT vs RLVR
Focuses on weight differences, skips attention analysis to avoid SDPA issues
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

class WeightOnlyAnalyzer:
    def __init__(self, base_model_path, sft_model_path, rlvr_model_path, device="cuda"):
        self.device = device
        self.base_model_path = base_model_path
        self.sft_model_path = sft_model_path
        self.rlvr_model_path = rlvr_model_path
        
        print("Loading models for weight analysis...")
        self.base_model = self._load_model(base_model_path, "Base")
        self.sft_model = self._load_model(sft_model_path, "SFT")
        self.rlvr_model = self._load_model(rlvr_model_path, "RLVR")
        
    def _load_model(self, model_path, name):
        print(f"Loading {name} model from {model_path}")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use fp32 for precise diff calculations
            device_map="cpu",  # Load to CPU first for memory efficiency
            trust_remote_code=True
        )
    
    def compute_weight_differences(self):
        """Compute weight differences between base, SFT, and RLVR models"""
        print("Computing weight differences...")
        
        results = {
            'layer_analysis': {},
            'component_analysis': {},
            'magnitude_analysis': {},
            'direction_analysis': {}
        }
        
        base_state = self.base_model.state_dict()
        sft_state = self.sft_model.state_dict()
        rlvr_state = self.rlvr_model.state_dict()
        
        param_count = 0
        
        # Analyze each parameter
        for param_name in base_state.keys():
            if param_name not in sft_state or param_name not in rlvr_state:
                continue
                
            base_param = base_state[param_name].float()
            sft_param = sft_state[param_name].float()
            rlvr_param = rlvr_state[param_name].float()
            
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
        """Analyze a specific parameter's changes"""
        
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
            
            # Magnitude analysis
            sft_magnitude = torch.norm(sft_diff).item()
            rlvr_magnitude = torch.norm(rlvr_diff).item()
            base_magnitude = torch.norm(base_param).item()
            
            # Relative changes
            sft_relative = sft_magnitude / (base_magnitude + 1e-8)
            rlvr_relative = rlvr_magnitude / (base_magnitude + 1e-8)
            
            # Direction analysis (cosine similarity)
            sft_flat = sft_diff.flatten()
            rlvr_flat = rlvr_diff.flatten()
            
            direction_similarity = torch.cosine_similarity(
                sft_flat.unsqueeze(0), 
                rlvr_flat.unsqueeze(0)
            ).item()
            
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
    
    def generate_comprehensive_report(self, results):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE SFT vs RLVR WEIGHT ANALYSIS REPORT")
        print("="*80)
        
        # 1. Overall Summary
        total_layers = len(results['layer_analysis'])
        print(f"\n📊 OVERVIEW")
        print(f"• Analyzed {total_layers} transformer layers")
        print(f"• Compared Base → SFT vs Base → RLVR weight changes")
        
        # 2. Layer-wise Analysis
        print(f"\n🧠 LAYER-WISE ANALYSIS")
        layer_data = []
        for layer_num, data in results['layer_analysis'].items():
            layer_data.append({
                'layer': layer_num,
                'sft_change': data['sft_total_change'],
                'rlvr_change': data['rlvr_total_change'],
                'ratio': data['rlvr_total_change'] / (data['sft_total_change'] + 1e-8)
            })
        
        # Sort by total change magnitude
        layer_data.sort(key=lambda x: x['sft_change'] + x['rlvr_change'], reverse=True)
        
        print(f"\nTop 10 Most Changed Layers:")
        for i, layer in enumerate(layer_data[:10]):
            ratio_str = f"RLVR={layer['ratio']:.2f}x SFT" if layer['ratio'] > 1 else f"SFT={1/layer['ratio']:.2f}x RLVR"
            print(f"{i+1:2d}. Layer {layer['layer']:2d}: SFT={layer['sft_change']:.4f}, RLVR={layer['rlvr_change']:.4f} ({ratio_str})")
        
        # 3. Component Analysis
        print(f"\n⚙️  COMPONENT ANALYSIS")
        component_stats = defaultdict(lambda: {'sft': 0, 'rlvr': 0, 'count': 0, 'similarities': []})
        
        for layer_data in results['layer_analysis'].values():
            for comp in layer_data['components']:
                comp_type = comp['component_type']
                component_stats[comp_type]['sft'] += comp['sft_change']
                component_stats[comp_type]['rlvr'] += comp['rlvr_change']
                component_stats[comp_type]['count'] += 1
                component_stats[comp_type]['similarities'].append(comp['direction_similarity'])
        
        print(f"\nAverage changes by component type:")
        component_summary = []
        for comp_type, stats in component_stats.items():
            if stats['count'] > 0:
                avg_sft = stats['sft'] / stats['count']
                avg_rlvr = stats['rlvr'] / stats['count']
                avg_similarity = sum(stats['similarities']) / len(stats['similarities'])
                component_summary.append({
                    'component': comp_type,
                    'sft': avg_sft,
                    'rlvr': avg_rlvr,
                    'similarity': avg_similarity,
                    'count': stats['count']
                })
        
        # Sort by total change
        component_summary.sort(key=lambda x: x['sft'] + x['rlvr'], reverse=True)
        
        for comp in component_summary:
            print(f"• {comp['component']:<20}: SFT={comp['sft']:.4f}, RLVR={comp['rlvr']:.4f}, "
                  f"Similarity={comp['similarity']:+.3f}, Count={comp['count']}")
        
        # 4. Key Insights
        print(f"\n💡 KEY INSIGHTS")
        
        # Find patterns
        insights = []
        
        # Check which approach changes weights more
        total_sft_change = sum(layer['sft_change'] for layer in layer_data)
        total_rlvr_change = sum(layer['rlvr_change'] for layer in layer_data)
        
        if total_rlvr_change > total_sft_change * 1.1:
            insights.append(f"RLVR makes {total_rlvr_change/total_sft_change:.2f}x larger weight changes than SFT")
        elif total_sft_change > total_rlvr_change * 1.1:
            insights.append(f"SFT makes {total_sft_change/total_rlvr_change:.2f}x larger weight changes than RLVR")
        else:
            insights.append("SFT and RLVR make similar magnitude weight changes")
        
        # Check component preferences
        for comp in component_summary[:3]:  # Top 3 most changed
            if comp['rlvr'] > comp['sft'] * 1.2:
                insights.append(f"RLVR particularly affects {comp['component']} components")
            elif comp['sft'] > comp['rlvr'] * 1.2:
                insights.append(f"SFT particularly affects {comp['component']} components")
        
        # Check direction similarity
        avg_similarities = [comp['similarity'] for comp in component_summary]
        overall_similarity = sum(avg_similarities) / len(avg_similarities)
        
        if overall_similarity > 0.5:
            insights.append(f"SFT and RLVR change weights in similar directions (similarity={overall_similarity:.3f})")
        elif overall_similarity < -0.1:
            insights.append(f"SFT and RLVR change weights in opposite directions (similarity={overall_similarity:.3f})")
        else:
            insights.append(f"SFT and RLVR change weights in different directions (similarity={overall_similarity:.3f})")
        
        # Check layer patterns
        early_layers = [l for l in layer_data if l['layer'] < total_layers // 3]
        late_layers = [l for l in layer_data if l['layer'] > 2 * total_layers // 3]
        
        early_total = sum(l['sft_change'] + l['rlvr_change'] for l in early_layers)
        late_total = sum(l['sft_change'] + l['rlvr_change'] for l in late_layers)
        
        if late_total > early_total * 1.5:
            insights.append("Later layers (near output) change more than early layers")
        elif early_total > late_total * 1.5:
            insights.append("Early layers (near input) change more than late layers")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # 5. Save detailed results
        detailed_results = {
            'summary': {
                'total_layers': total_layers,
                'total_sft_change': total_sft_change,
                'total_rlvr_change': total_rlvr_change,
                'overall_similarity': overall_similarity
            },
            'layer_analysis': results['layer_analysis'],
            'component_summary': component_summary,
            'insights': insights
        }
        
        with open("detailed_weight_analysis.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n✅ Detailed results saved to 'detailed_weight_analysis.json'")
        print(f"🎉 Analysis complete! This shows exactly how SFT vs RLVR affects model weights!")
        
        return detailed_results

def main():
    """Main analysis function"""
    print("🔬 Weight-Only Analysis: SFT vs RLVR")
    print("="*60)
    
    # Model paths
    base_model = "meta-llama/Llama-3.1-8B"
    sft_model = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    rlvr_model = "allenai/Llama-3.1-Tulu-3-8B"
    
    analyzer = WeightOnlyAnalyzer(base_model, sft_model, rlvr_model)
    
    print("\n1. Computing weight differences...")
    results = analyzer.compute_weight_differences()
    
    print("\n2. Generating comprehensive report...")
    report = analyzer.generate_comprehensive_report(results)
    
    print(f"\n🎯 SUCCESS! You now have detailed insights into how SFT vs RLVR affects model weights!")

if __name__ == "__main__":
    main() 