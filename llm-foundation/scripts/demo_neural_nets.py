#!/usr/bin/env python3
"""
Demo script for Neural Networks
Run this to see the neural networks in action
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from neural_nets.examples.simple_examples import (
    xor_example,
    linear_regression_example,
    classification_example
)

def main():
    """Run all neural network examples"""
    print("🧠 Neural Networks Demo")
    print("=" * 50)
    
    try:
        # Run XOR example
        print("\n1. XOR Problem")
        xor_network, xor_trainer = xor_example()
        
        # Run linear regression example
        print("\n2. Linear Regression")
        lr_network, lr_trainer = linear_regression_example()
        
        # Run classification example
        print("\n3. Binary Classification")
        clf_network, clf_trainer = classification_example()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running demos: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
