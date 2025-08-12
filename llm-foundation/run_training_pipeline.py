#!/usr/bin/env python3
"""
Transformer Training Pipeline Runner
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚀 Running Transformer Training Pipeline")
    print("=" * 60)
    
    try:
        # Import and run training examples
        from transformer.examples.training_example import (
            train_transformer_example,
            demonstrate_attention_learning
        )
        
        print("🔥 TRANSFORMER TRAINING PIPELINE")
        print("=" * 60)
        
        # Run complete training example
        transformer, trainer, vocab = train_transformer_example()
        
        # Run attention demonstration
        transformer2, trainer2 = demonstrate_attention_learning()
        
        print("\n✅ Training pipeline completed successfully!")
        print("\n🎯 What we accomplished:")
        print("   - ✅ Complete training pipeline with Adam optimizer")
        print("   - ✅ Cross-entropy loss with gradient clipping")
        print("   - ✅ Real English to French translation training")
        print("   - ✅ Training history visualization")
        print("   - ✅ Attention weight tracking")
        print("   - ✅ Model evaluation and testing")
        print("   - ✅ Autoregressive text generation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 