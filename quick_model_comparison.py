#!/usr/bin/env python3
"""
Quick Model Comparison: SFT vs RLVR Behavioral Analysis
Tests the same problems on both models to see reasoning differences.

This lightweight script helps you understand the behavioral differences 
between SFT and RLVR models before investing in full training.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Test problems (GSM8K style)
TEST_PROBLEMS = [
    {
        "problem": "Janet has 3 apples. She gives away 1 apple and buys 2 more. How many apples does she have?",
        "expected": "4"
    },
    {
        "problem": "A train travels 60 miles in 2 hours. What is its speed in miles per hour?",
        "expected": "30"
    },
    {
        "problem": "Sarah has $15. She buys a book for $8 and a pen for $3. How much money does she have left?",
        "expected": "$4"
    }
]

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer"""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate response from model"""
    # Format as chat if possible
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "user", "content": f"Solve this step by step:\n{prompt}"}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = f"Solve this step by step:\n{prompt}\n\nAnswer:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()

def compare_models(sft_model_name, rlvr_model_name):
    """Compare responses between SFT and RLVR models"""
    
    print("🔍 Quick Model Comparison: SFT vs RLVR")
    print("=" * 50)
    
    # Load models
    sft_model, sft_tokenizer = load_model_and_tokenizer(sft_model_name)
    rlvr_model, rlvr_tokenizer = load_model_and_tokenizer(rlvr_model_name)
    
    results = []
    
    for i, test_case in enumerate(TEST_PROBLEMS):
        print(f"\n📝 Problem {i+1}: {test_case['problem']}")
        print(f"Expected: {test_case['expected']}")
        print("-" * 40)
        
        # Get responses from both models
        sft_response = generate_response(sft_model, sft_tokenizer, test_case['problem'])
        rlvr_response = generate_response(rlvr_model, rlvr_tokenizer, test_case['problem'])
        
        print(f"🔵 SFT Response:\n{sft_response}\n")
        print(f"🔴 RLVR Response:\n{rlvr_response}\n")
        
        # Simple analysis
        sft_has_expected = test_case['expected'].lower() in sft_response.lower()
        rlvr_has_expected = test_case['expected'].lower() in rlvr_response.lower()
        
        print(f"✅ SFT Correct: {sft_has_expected}")
        print(f"✅ RLVR Correct: {rlvr_has_expected}")
        
        # Compare reasoning length
        sft_length = len(sft_response.split())
        rlvr_length = len(rlvr_response.split())
        
        print(f"📏 SFT Response Length: {sft_length} words")
        print(f"📏 RLVR Response Length: {rlvr_length} words")
        
        results.append({
            'problem': test_case['problem'],
            'expected': test_case['expected'],
            'sft_response': sft_response,
            'rlvr_response': rlvr_response,
            'sft_correct': sft_has_expected,
            'rlvr_correct': rlvr_has_expected,
            'sft_length': sft_length,
            'rlvr_length': rlvr_length
        })
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 30)
    
    sft_correct_count = sum(r['sft_correct'] for r in results)
    rlvr_correct_count = sum(r['rlvr_correct'] for r in results)
    avg_sft_length = sum(r['sft_length'] for r in results) / len(results)
    avg_rlvr_length = sum(r['rlvr_length'] for r in results) / len(results)
    
    print(f"SFT Accuracy: {sft_correct_count}/{len(results)} ({sft_correct_count/len(results)*100:.1f}%)")
    print(f"RLVR Accuracy: {rlvr_correct_count}/{len(results)} ({rlvr_correct_count/len(results)*100:.1f}%)")
    print(f"Average SFT Response Length: {avg_sft_length:.1f} words")
    print(f"Average RLVR Response Length: {avg_rlvr_length:.1f} words")
    
    # Save detailed results
    with open('model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: model_comparison_results.json")
    
    return results

def analyze_weight_specific_layers():
    """Quick analysis of specific layer differences (lightweight version)"""
    print("\n🔍 Quick Weight Analysis Preview")
    print("=" * 40)
    
    base_model = "meta-llama/Llama-3.1-8B"
    sft_model = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    rlvr_model = "allenai/Llama-3.1-Tulu-3-8B"
    
    print("Available models for weight analysis:")
    print(f"📦 Base: {base_model}")
    print(f"📦 SFT: {sft_model}")
    print(f"📦 RLVR: {rlvr_model}")
    
    print("\n💡 To run full weight analysis:")
    print("python analyze_weight_differences.py --device cuda")
    print("\n⚠️  Note: Requires ~50GB RAM or GPU memory")
    print("Consider running on a cloud instance with sufficient resources")

if __name__ == "__main__":
    # Model names for comparison
    sft_model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
    rlvr_model_name = "allenai/Llama-3.1-Tulu-3-8B"
    
    print("🚀 Quick Model Comparison")
    print("This script tests SFT vs RLVR behavior on simple math problems")
    print("Choose an option:")
    print("1. Compare model responses (requires GPU)")
    print("2. Weight analysis info only")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        try:
            results = compare_models(sft_model_name, rlvr_model_name)
            print("\n🎯 Key Insights to Look For:")
            print("- Does RLVR show more detailed reasoning?")
            print("- Are there different solution approaches?") 
            print("- Which model is more confident in answers?")
            print("- Does RLVR explore alternative methods?")
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try running on a machine with GPU and sufficient memory")
    
    elif choice == "2":
        analyze_weight_specific_layers()
    
    else:
        print("Invalid choice. Please run again with 1 or 2.") 