#!/usr/bin/env python3
"""
Transformer Pipeline Runner
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 Running Transformer Pipeline")
    print("=" * 60)
    
    try:
        # Import and run transformer examples
        from transformer.examples.transformer_examples import (
            simple_translation_example,
            visualize_attention_weights,
            demonstrate_positional_encoding
        )
        
        print("🧠 TRANSFORMER IMPLEMENTATION")
        print("=" * 60)
        
        # Run examples
        transformer, vocab = simple_translation_example()
        visualize_attention_weights()
        demonstrate_positional_encoding()
        
        print("\n✅ Transformer pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
