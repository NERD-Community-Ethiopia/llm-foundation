#!/usr/bin/env python3
"""
Attention Pipeline Runner
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 Running Attention Mechanism Pipeline\n")
    
    try:
        # Import and run attention examples
        from attention.examples.attention_examples import (
            basic_attention_example,
            self_attention_example, 
            multi_head_attention_example,
            visualize_attention_weights
        )
        
        print("=" * 60)
        print("🧠 ATTENTION MECHANISM PIPELINE")
        print("=" * 60)
        
        # Step 1: Basic Attention
        print("\n📊 STEP 1: Basic Attention Mechanism")
        print("-" * 40)
        basic_attn, basic_weights = basic_attention_example()
        visualize_attention_weights(basic_weights, "Basic Attention")
        
        # Step 2: Self-Attention
        print("\n📊 STEP 2: Self-Attention Mechanism")
        print("-" * 40)
        self_attn, self_weights = self_attention_example()
        visualize_attention_weights(self_weights, "Self Attention")
        
        # Step 3: Multi-Head Attention
        print("\n📊 STEP 3: Multi-Head Attention Mechanism")
        print("-" * 40)
        mha, mha_weights = multi_head_attention_example()
        visualize_attention_weights(mha_weights[0, 0], "Multi-Head Attention (Head 0)")
        
        # Step 4: Analysis
        print("\n📊 STEP 4: Attention Analysis")
        print("-" * 40)
        print("✅ Basic Attention: Query-Key-Value mechanism working")
        print("✅ Self-Attention: Words attending to all other words")
        print("✅ Multi-Head Attention: Multiple attention heads in parallel")
        print("✅ Causal Masking: Future positions masked for decoder")
        
        print("\n🎯 ATTENTION INSIGHTS:")
        print("-" * 40)
        print("• Basic attention allows flexible query-key matching")
        print("• Self-attention enables each word to understand all other words")
        print("• Multi-head attention captures different types of relationships")
        print("• Attention weights show what the model is focusing on")
        
        print("\n📈 VISUALIZATIONS CREATED:")
        print("-" * 40)
        print("• attention_weights_basic_attention.png")
        print("• attention_weights_self_attention.png") 
        print("• attention_weights_multi-head_attention_(head_0).png")
        
        print("\n✅ ATTENTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all attention modules are properly implemented")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 