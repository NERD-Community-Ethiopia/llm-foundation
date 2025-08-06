#!/usr/bin/env python3
"""
Script to save neural network plots to files
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
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
    """Run examples and save plots"""
    print("🧠 Running Neural Networks and Saving Plots")
    print("=" * 50)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        # XOR example
        print("\n1. XOR Problem")
        xor_network, xor_trainer = xor_example()
        
        # Save XOR training history
        plt.figure(figsize=(8, 6))
        plt.plot(xor_trainer.training_history['loss'])
        plt.title('XOR Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('plots/xor_training.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Linear regression example
        print("\n2. Linear Regression")
        lr_network, lr_trainer = linear_regression_example()
        
        # Save linear regression results
        plt.figure(figsize=(10, 4))
        
        # Generate test data for visualization
        X_test = np.array([[0, 2, 4, 6, 8, 10]])
        predictions = lr_network.predict(X_test)
        
        plt.subplot(1, 2, 1)
        plt.scatter([0, 2, 4, 6, 8, 10], [1, 5, 9, 13, 17, 21], 
                   c='blue', alpha=0.6, label='Expected')
        plt.plot(X_test[0], predictions[0], 'r-', linewidth=2, label='Predictions')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(lr_trainer.training_history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/linear_regression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification example
        print("\n3. Binary Classification")
        clf_network, clf_trainer = classification_example()
        
        # Save classification results
        plt.figure(figsize=(10, 4))
        
        # Generate classification data for visualization
        np.random.seed(42)
        n_samples = 200
        class0_x = np.random.normal(0, 1, n_samples // 2)
        class0_y = np.random.normal(0, 1, n_samples // 2)
        class1_x = np.random.normal(3, 1, n_samples // 2)
        class1_y = np.random.normal(3, 1, n_samples // 2)
        
        X = np.vstack([np.hstack([class0_x, class1_x]),
                       np.hstack([class0_y, class1_y])])
        
        plt.subplot(1, 2, 1)
        plt.scatter(X[0, :n_samples//2], X[1, :n_samples//2], 
                   c='blue', alpha=0.6, label='Class 0')
        plt.scatter(X[0, n_samples//2:], X[1, n_samples//2:], 
                   c='red', alpha=0.6, label='Class 1')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Classification Data')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(clf_trainer.training_history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/classification.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n✅ All plots saved to 'plots/' directory!")
        print("📁 Generated files:")
        print("  - plots/xor_training.png")
        print("  - plots/linear_regression.png") 
        print("  - plots/classification.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 