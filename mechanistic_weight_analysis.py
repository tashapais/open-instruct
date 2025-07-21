#!/usr/bin/env python3
"""
Mechanistic Interpretability: SFT vs RLVR Weight Analysis
Understanding what changes in the weights after different fine-tuning approaches
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

class WeightAnalyzer:
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
    
    def analyze_attention_patterns(self, sample_texts):
        """Analyze how attention patterns differ between models"""
        print("Analyzing attention patterns...")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        attention_results = {}
        
        for i, text in enumerate(sample_texts):
            print(f"Processing text {i+1}/{len(sample_texts)}")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Get attention weights from each model
            with torch.no_grad():
                sft_outputs = self.sft_model(**inputs, output_attentions=True)
                rlvr_outputs = self.rlvr_model(**inputs, output_attentions=True)
            
            # Compare attention patterns
            sft_attentions = torch.stack(sft_outputs.attentions)  # [layers, batch, heads, seq, seq]
            rlvr_attentions = torch.stack(rlvr_outputs.attentions)
            
            # Compute differences
            attention_diff = torch.abs(sft_attentions - rlvr_attentions).mean(dim=[1, 3, 4])  # Average over batch and sequence
            
            attention_results[f"text_{i}"] = {
                'text': text,
                'attention_differences': attention_diff.tolist(),  # [layers, heads]
                'max_diff_layer': torch.argmax(attention_diff.mean(dim=1)).item(),
                'max_diff_head': torch.argmax(attention_diff.flatten()).item()
            }
        
        return attention_results
    
    def visualize_results(self, results, save_dir="analysis_results"):
        """Create visualizations of the analysis results"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # 1. Layer-wise change magnitude
        self._plot_layer_changes(results['layer_analysis'], save_dir)
        
        # 2. Component-wise analysis
        self._plot_component_analysis(results, save_dir)
        
        # 3. Direction similarity heatmap
        self._plot_direction_similarity(results, save_dir)
        
        print(f"Visualizations saved to {save_dir}/")
    
    def _plot_layer_changes(self, layer_analysis, save_dir):
        """Plot layer-wise weight changes"""
        layers = sorted(layer_analysis.keys())
        sft_changes = [layer_analysis[l]['sft_total_change'] for l in layers]
        rlvr_changes = [layer_analysis[l]['rlvr_total_change'] for l in layers]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(layers))
        width = 0.35
        
        plt.bar(x - width/2, sft_changes, width, label='SFT', alpha=0.8)
        plt.bar(x + width/2, rlvr_changes, width, label='RLVR', alpha=0.8)
        
        plt.xlabel('Layer')
        plt.ylabel('Total Weight Change (Relative)')
        plt.title('Weight Changes by Layer: SFT vs RLVR')
        plt.xticks(x, layers)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/layer_changes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_analysis(self, results, save_dir):
        """Plot component-wise analysis"""
        # Aggregate by component type
        component_data = defaultdict(lambda: {'sft': [], 'rlvr': []})
        
        for layer_data in results['layer_analysis'].values():
            for comp in layer_data['components']:
                component_data[comp['component_type']]['sft'].append(comp['sft_change'])
                component_data[comp['component_type']]['rlvr'].append(comp['rlvr_change'])
        
        # Create DataFrame for plotting
        plot_data = []
        for comp_type, data in component_data.items():
            for change in data['sft']:
                plot_data.append({'Component': comp_type, 'Change': change, 'Model': 'SFT'})
            for change in data['rlvr']:
                plot_data.append({'Component': comp_type, 'Change': change, 'Model': 'RLVR'})
        
        df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df, x='Component', y='Change', hue='Model')
        plt.xticks(rotation=45)
        plt.title('Weight Changes by Component Type')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/component_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_direction_similarity(self, results, save_dir):
        """Plot direction similarity heatmap"""
        # Create similarity matrix
        similarity_data = []
        
        for layer_data in results['layer_analysis'].values():
            for comp in layer_data['components']:
                similarity_data.append({
                    'component': comp['component_type'],
                    'similarity': comp['direction_similarity']
                })
        
        if similarity_data:
            df = pd.DataFrame(similarity_data)
            pivot_df = df.groupby('component')['similarity'].mean().reset_index()
            
            plt.figure(figsize=(10, 6))
            plt.bar(pivot_df['component'], pivot_df['similarity'])
            plt.xlabel('Component Type')
            plt.ylabel('Average Direction Similarity (SFT vs RLVR)')
            plt.title('How Similar are SFT and RLVR Weight Change Directions?')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No similarity')
            plt.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Perfect similarity')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/direction_similarity.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, results, attention_results=None):
        """Generate a comprehensive analysis report"""
        report = {
            "summary": self._generate_summary(results),
            "detailed_findings": self._generate_detailed_findings(results),
            "attention_analysis": attention_results,
            "recommendations": self._generate_recommendations(results)
        }
        
        # Save report
        with open("mechanistic_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print key findings
        print("\n" + "="*60)
        print("🔍 MECHANISTIC ANALYSIS REPORT")
        print("="*60)
        print(report["summary"])
        print("\n📋 Key Findings:")
        for finding in report["detailed_findings"]:
            print(f"• {finding}")
        
        return report
    
    def _generate_summary(self, results):
        """Generate summary of findings"""
        total_layers = len(results['layer_analysis'])
        
        # Find layers with biggest differences
        sft_changes = [(k, v['sft_total_change']) for k, v in results['layer_analysis'].items()]
        rlvr_changes = [(k, v['rlvr_total_change']) for k, v in results['layer_analysis'].items()]
        
        max_sft_layer = max(sft_changes, key=lambda x: x[1])
        max_rlvr_layer = max(rlvr_changes, key=lambda x: x[1])
        
        return f"""
        Analyzed {total_layers} layers across SFT and RLVR fine-tuned models.
        
        Largest SFT changes: Layer {max_sft_layer[0]} (change: {max_sft_layer[1]:.4f})
        Largest RLVR changes: Layer {max_rlvr_layer[0]} (change: {max_rlvr_layer[1]:.4f})
        
        This suggests different fine-tuning approaches affect different parts of the network.
        """
    
    def _generate_detailed_findings(self, results):
        """Generate detailed findings"""
        findings = []
        
        # Analyze which components change most
        component_changes = defaultdict(lambda: {'sft': 0, 'rlvr': 0, 'count': 0})
        
        for layer_data in results['layer_analysis'].values():
            for comp in layer_data['components']:
                comp_type = comp['component_type']
                component_changes[comp_type]['sft'] += comp['sft_change']
                component_changes[comp_type]['rlvr'] += comp['rlvr_change']
                component_changes[comp_type]['count'] += 1
        
        # Find most affected components
        for comp_type, data in component_changes.items():
            if data['count'] > 0:
                avg_sft = data['sft'] / data['count']
                avg_rlvr = data['rlvr'] / data['count']
                
                if avg_sft > avg_rlvr * 1.2:
                    findings.append(f"SFT affects {comp_type} components more than RLVR ({avg_sft:.4f} vs {avg_rlvr:.4f})")
                elif avg_rlvr > avg_sft * 1.2:
                    findings.append(f"RLVR affects {comp_type} components more than SFT ({avg_rlvr:.4f} vs {avg_sft:.4f})")
        
        return findings
    
    def _generate_recommendations(self, results):
        """Generate recommendations for further analysis"""
        return [
            "Examine specific attention heads that show large differences",
            "Analyze activation patterns on reasoning vs non-reasoning tasks",
            "Investigate whether RLVR changes relate to step-by-step thinking",
            "Study how these changes affect model confidence and uncertainty",
            "Probe for emergence of new reasoning circuits in RLVR model"
        ]

def main():
    """Main analysis function"""
    print("🔬 Mechanistic Interpretability: SFT vs RLVR Analysis")
    print("="*60)
    
    # Model paths (adjust these to your actual model paths)
    base_model = "meta-llama/Llama-3.1-8B"
    sft_model = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    rlvr_model = "allenai/Llama-3.1-Tulu-3-8B"
    
    analyzer = WeightAnalyzer(base_model, sft_model, rlvr_model)
    
    print("\n1. Computing weight differences...")
    results = analyzer.compute_weight_differences()
    
    print("\n2. Analyzing attention patterns...")
    sample_texts = [
        "Solve this step by step: Janet has 3 apples. She gives away 1 apple and buys 2 more. How many apples does she have?",
        "A train travels 60 miles in 2 hours. What is its speed?",
        "Sarah has $15. She buys a book for $8 and a pen for $3. How much money does she have left?"
    ]
    attention_results = analyzer.analyze_attention_patterns(sample_texts)
    
    print("\n3. Creating visualizations...")
    analyzer.visualize_results(results)
    
    print("\n4. Generating report...")
    report = analyzer.generate_report(results, attention_results)
    
    print(f"\n✅ Analysis complete! Check 'mechanistic_analysis_report.json' for detailed results.")

if __name__ == "__main__":
    main()