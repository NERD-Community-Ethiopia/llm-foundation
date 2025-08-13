#!/bin/bash

# Build Course Structure for Neural Networks (Checkpoint 1)
# This script creates the folder structure and initial files for the neural networks section

echo "🚀 Building Neural Networks Course Structure..."

# Create main source directory
mkdir -p src

# Create neural networks specific directory
mkdir -p src/neural_nets

# Create supporting directories that will be needed for neural nets
mkdir -p src/utils
mkdir -p src/data
mkdir -p tests
mkdir -p data
mkdir -p models

# Create neural networks specific subdirectories
mkdir -p src/neural_nets/basic
mkdir -p src/neural_nets/backpropagation
mkdir -p src/neural_nets/training
mkdir -p src/neural_nets/examples

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/neural_nets/__init__.py
touch src/neural_nets/basic/__init__.py
touch src/neural_nets/backpropagation/__init__.py
touch src/neural_nets/training/__init__.py
touch src/neural_nets/examples/__init__.py
touch src/utils/__init__.py
touch src/data/__init__.py

# Create main neural network implementation files
echo "📝 Creating neural network implementation files..."

# Basic feedforward neural network
cat > src/neural_nets/basic/feedforward.py << 'EOF'
"""
Basic Feedforward Neural Network Implementation
"""
import numpy as np
from typing import List, Tuple, Optional

class FeedforwardNeuralNetwork:
    """
    A simple feedforward neural network implementation
    """
    
    def __init__(self, layer_sizes: List[int], activation='sigmoid'):
        """
        Initialize the neural network
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            activation: Activation function to use ('sigmoid', 'relu', 'tanh')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization for better training
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((layer_sizes[i + 1], 1))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function"""
        return np.where(z > 0, 1, 0)
    
    def tanh(self, z: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(z)
    
    def tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of tanh function"""
        return 1 - np.tanh(z) ** 2
    
    def activate(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def activate_derivative(self, z: np.ndarray) -> np.ndarray:
        """Apply derivative of activation function"""
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation == 'relu':
            return self.relu_derivative(z)
        elif self.activation == 'tanh':
            return self.tanh_derivative(z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network
        
        Args:
            X: Input data of shape (input_size, batch_size)
            
        Returns:
            Tuple of (activations, z_values) for each layer
        """
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            
            # Don't apply activation to the last layer (output layer)
            if i == self.num_layers - 2:
                activations.append(z)  # Linear output for regression
            else:
                activations.append(self.activate(z))
        
        return activations, z_values
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network
        
        Args:
            X: Input data of shape (input_size, batch_size)
            
        Returns:
            Predictions
        """
        activations, _ = self.forward(X)
        return activations[-1]
EOF

# Backpropagation implementation
cat > src/neural_nets/backpropagation/backprop.py << 'EOF'
"""
Backpropagation Implementation
"""
import numpy as np
from typing import List, Tuple

class Backpropagation:
    """
    Backpropagation algorithm implementation
    """
    
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = 'mse') -> float:
        """
        Compute loss between true and predicted values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            loss_type: Type of loss function ('mse', 'cross_entropy')
            
        Returns:
            Loss value
        """
        if loss_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif loss_type == 'cross_entropy':
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def compute_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = 'mse') -> np.ndarray:
        """
        Compute derivative of loss function
        
        Args:
            y_true: True values
            y_pred: Predicted values
            loss_type: Type of loss function ('mse', 'cross_entropy')
            
        Returns:
            Loss derivative
        """
        if loss_type == 'mse':
            return 2 * (y_pred - y_true) / y_true.size
        elif loss_type == 'cross_entropy':
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return (y_pred - y_true) / (y_pred * (1 - y_pred))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def backward_pass(
        activations: List[np.ndarray],
        z_values: List[np.ndarray],
        weights: List[np.ndarray],
        y_true: np.ndarray,
        network: 'FeedforwardNeuralNetwork',
        loss_type: str = 'mse'
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward pass to compute gradients
        
        Args:
            activations: List of activations from forward pass
            z_values: List of z values from forward pass
            weights: List of weight matrices
            y_true: True target values
            network: Neural network instance
            loss_type: Type of loss function
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = y_true.shape[1]  # batch size
        num_layers = len(activations)
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in weights]
        bias_gradients = [np.zeros_like(b) for b in network.biases]
        
        # Compute initial error (output layer)
        delta = Backpropagation.compute_loss_derivative(y_true, activations[-1], loss_type)
        
        # Backpropagate error through layers
        for layer in range(num_layers - 2, -1, -1):
            # Compute gradients for current layer
            weight_gradients[layer] = np.dot(delta, activations[layer].T) / m
            bias_gradients[layer] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Compute error for next layer back (if not at input layer)
            if layer > 0:
                delta = np.dot(weights[layer].T, delta) * network.activate_derivative(z_values[layer - 1])
        
        return weight_gradients, bias_gradients
EOF

# Training module
cat > src/neural_nets/training/trainer.py << 'EOF'
"""
Neural Network Training Module
"""
import numpy as np
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
    """
    Trainer class for neural networks
    """
    
    def __init__(self, network, learning_rate: float = 0.01):
        """
        Initialize trainer
        
        Args:
            network: Neural network instance
            learning_rate: Learning rate for gradient descent
        """
        self.network = network
        self.learning_rate = learning_rate
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        loss_type: str = 'mse',
        verbose: bool = True,
        early_stopping: bool = False,
        patience: int = 10
    ) -> dict:
        """
        Train the neural network
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            loss_type: Type of loss function
            verbose: Whether to print training progress
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Mini-batch training
            if batch_size and batch_size < X_train.shape[1]:
                indices = np.random.permutation(X_train.shape[1])
                for i in range(0, X_train.shape[1], batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch = X_train[:, batch_indices]
                    y_batch = y_train[:, batch_indices]
                    
                    self._train_step(X_batch, y_batch, loss_type)
            else:
                self._train_step(X_train, y_train, loss_type)
            
            # Compute training loss
            train_loss = self._compute_loss(X_train, y_train, loss_type)
            self.training_history['loss'].append(train_loss)
            
            # Compute validation loss if validation data provided
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(X_val, y_val, loss_type)
                
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            
            # Print progress
            if verbose and epoch % 100 == 0:
                val_info = f", Val Loss: {val_loss:.6f}" if X_val is not None else ""
                print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}{val_info}")
        
        return self.training_history
    
    def _train_step(self, X: np.ndarray, y: np.ndarray, loss_type: str):
        """Single training step"""
        # Forward pass
        activations, z_values = self.network.forward(X)
        
        # Backward pass
        weight_gradients, bias_gradients = self.network.backward_pass(
            activations, z_values, self.network.weights, y, self.network, loss_type
        )
        
        # Update weights and biases
        for i in range(len(self.network.weights)):
            self.network.weights[i] -= self.learning_rate * weight_gradients[i]
            self.network.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray, loss_type: str) -> float:
        """Compute loss for given data"""
        y_pred = self.network.predict(X)
        return self.network.compute_loss(y, y_pred, loss_type)
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if self.training_history['accuracy']:
            plt.subplot(1, 2, 2)
            plt.plot(self.training_history['accuracy'])
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
EOF

# Simple training examples
cat > src/neural_nets/examples/simple_examples.py << 'EOF'
"""
Simple Training Examples for Neural Networks
"""
import numpy as np
import matplotlib.pyplot as plt
from src.neural_nets.basic.feedforward import FeedforwardNeuralNetwork
from src.neural_nets.training.trainer import NeuralNetworkTrainer

def xor_example():
    """
    Train a neural network to learn XOR function
    """
    print("🔍 Training Neural Network on XOR Problem")
    
    # XOR data
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Create network: 2 input -> 4 hidden -> 1 output
    network = FeedforwardNeuralNetwork([2, 4, 1], activation='sigmoid')
    
    # Create trainer
    trainer = NeuralNetworkTrainer(network, learning_rate=0.1)
    
    # Train the network
    history = trainer.train(X, y, epochs=5000, verbose=True)
    
    # Test predictions
    predictions = network.predict(X)
    print("\n📊 XOR Results:")
    print("Input\t\tPredicted\tExpected")
    print("-" * 40)
    for i in range(X.shape[1]):
        print(f"({X[0, i]}, {X[1, i]})\t\t{predictions[0, i]:.4f}\t\t{y[0, i]}")
    
    # Plot training history
    trainer.plot_training_history()
    
    return network, trainer

def linear_regression_example():
    """
    Train a neural network for linear regression
    """
    print("\n🔍 Training Neural Network for Linear Regression")
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X = np.random.rand(1, 100) * 10
    y = 2 * X + 1 + np.random.normal(0, 0.1, (1, 100))
    
    # Create network: 1 input -> 1 output (linear)
    network = FeedforwardNeuralNetwork([1, 1], activation='linear')
    
    # Create trainer
    trainer = NeuralNetworkTrainer(network, learning_rate=0.01)
    
    # Train the network
    history = trainer.train(X, y, epochs=1000, verbose=True)
    
    # Test predictions
    X_test = np.array([[0, 2, 4, 6, 8, 10]])
    predictions = network.predict(X_test)
    
    print("\n📊 Linear Regression Results:")
    print("Input\tPredicted\tExpected (2x + 1)")
    print("-" * 40)
    for i in range(X_test.shape[1]):
        expected = 2 * X_test[0, i] + 1
        print(f"{X_test[0, i]:.1f}\t{predictions[0, i]:.4f}\t\t{expected:.1f}")
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[0], y[0], alpha=0.6, label='Training Data')
    plt.plot(X_test[0], predictions[0], 'r-', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return network, trainer

def classification_example():
    """
    Train a neural network for binary classification
    """
    print("\n🔍 Training Neural Network for Binary Classification")
    
    # Generate synthetic data for classification
    np.random.seed(42)
    n_samples = 200
    
    # Class 0: centered at (0, 0)
    class0_x = np.random.normal(0, 1, n_samples // 2)
    class0_y = np.random.normal(0, 1, n_samples // 2)
    
    # Class 1: centered at (3, 3)
    class1_x = np.random.normal(3, 1, n_samples // 2)
    class1_y = np.random.normal(3, 1, n_samples // 2)
    
    # Combine data
    X = np.vstack([np.hstack([class0_x, class1_x]),
                   np.hstack([class0_y, class1_y])])
    y = np.vstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Create network: 2 input -> 8 hidden -> 1 output
    network = FeedforwardNeuralNetwork([2, 8, 1], activation='sigmoid')
    
    # Create trainer
    trainer = NeuralNetworkTrainer(network, learning_rate=0.1)
    
    # Train the network
    history = trainer.train(X, y, epochs=2000, verbose=True)
    
    # Test predictions
    predictions = network.predict(X)
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y)
    
    print(f"\n📊 Classification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
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
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return network, trainer

if __name__ == "__main__":
    # Run all examples
    print("🚀 Running Neural Network Examples\n")
    
    # XOR example
    xor_network, xor_trainer = xor_example()
    
    # Linear regression example
    lr_network, lr_trainer = linear_regression_example()
    
    # Classification example
    clf_network, clf_trainer = classification_example()
    
    print("\n✅ All examples completed!")
EOF

# Create test files
cat > tests/test_neural_nets.py << 'EOF'
"""
Tests for Neural Networks Implementation
"""
import numpy as np
import pytest
from src.neural_nets.basic.feedforward import FeedforwardNeuralNetwork
from src.neural_nets.backpropagation.backprop import Backpropagation
from src.neural_nets.training.trainer import NeuralNetworkTrainer

class TestFeedforwardNeuralNetwork:
    """Test cases for FeedforwardNeuralNetwork"""
    
    def test_initialization(self):
        """Test network initialization"""
        network = FeedforwardNeuralNetwork([2, 3, 1])
        
        assert len(network.weights) == 2
        assert len(network.biases) == 2
        assert network.weights[0].shape == (3, 2)
        assert network.weights[1].shape == (1, 3)
        assert network.biases[0].shape == (3, 1)
        assert network.biases[1].shape == (1, 1)
    
    def test_forward_pass(self):
        """Test forward pass"""
        network = FeedforwardNeuralNetwork([2, 3, 1])
        X = np.array([[1, 2], [3, 4]])
        
        activations, z_values = network.forward(X)
        
        assert len(activations) == 3
        assert len(z_values) == 2
        assert activations[0].shape == (2, 2)
        assert activations[1].shape == (3, 2)
        assert activations[2].shape == (1, 2)
    
    def test_activation_functions(self):
        """Test activation functions"""
        network = FeedforwardNeuralNetwork([1, 1], activation='sigmoid')
        
        # Test sigmoid
        z = np.array([[0]])
        assert network.sigmoid(z)[0, 0] == pytest.approx(0.5, rel=1e-5)
        
        # Test ReLU
        network.activation = 'relu'
        assert network.relu(np.array([[-1, 0, 1]]))[0, 0] == 0
        assert network.relu(np.array([[-1, 0, 1]]))[0, 1] == 0
        assert network.relu(np.array([[-1, 0, 1]]))[0, 2] == 1

class TestBackpropagation:
    """Test cases for Backpropagation"""
    
    def test_mse_loss(self):
        """Test MSE loss computation"""
        y_true = np.array([[1, 2, 3]])
        y_pred = np.array([[1, 2, 3]])
        
        loss = Backpropagation.compute_loss(y_true, y_pred, 'mse')
        assert loss == pytest.approx(0.0, rel=1e-10)
    
    def test_mse_loss_derivative(self):
        """Test MSE loss derivative"""
        y_true = np.array([[1, 2, 3]])
        y_pred = np.array([[1, 2, 3]])
        
        derivative = Backpropagation.compute_loss_derivative(y_true, y_pred, 'mse')
        assert np.allclose(derivative, np.zeros_like(derivative))

class TestNeuralNetworkTrainer:
    """Test cases for NeuralNetworkTrainer"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        network = FeedforwardNeuralNetwork([2, 3, 1])
        trainer = NeuralNetworkTrainer(network, learning_rate=0.01)
        
        assert trainer.network == network
        assert trainer.learning_rate == 0.01
        assert 'loss' in trainer.training_history
        assert 'accuracy' in trainer.training_history

if __name__ == "__main__":
    pytest.main([__file__])
EOF

# Create main neural networks module
cat > src/neural_nets/neural_networks.py << 'EOF'
"""
Main Neural Networks Module
Combines all neural network components
"""
from .basic.feedforward import FeedforwardNeuralNetwork
from .backpropagation.backprop import Backpropagation
from .training.trainer import NeuralNetworkTrainer

__all__ = [
    'FeedforwardNeuralNetwork',
    'Backpropagation', 
    'NeuralNetworkTrainer'
]
EOF

# Create a simple demo script
cat > demo_neural_nets.py << 'EOF'
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
EOF

# Create requirements for neural networks
cat > requirements_neural_nets.txt << 'EOF'
numpy>=1.21.0
matplotlib>=3.5.0
pytest>=6.0.0
EOF

# Make the script executable
chmod +x build_neural_nets_structure.sh

echo "✅ Neural Networks course structure created successfully!"
echo ""
echo "📁 Created directories:"
echo "  - src/neural_nets/ (main neural networks module)"
echo "  - src/neural_nets/basic/ (feedforward networks)"
echo "  - src/neural_nets/backpropagation/ (backpropagation algorithm)"
echo "  - src/neural_nets/training/ (training utilities)"
echo "  - src/neural_nets/examples/ (simple training examples)"
echo "  - src/utils/ (utility functions)"
echo "  - src/data/ (data handling)"
echo "  - tests/ (test files)"
echo "  - data/ (dataset storage)"
echo "  - models/ (saved models)"
echo ""
echo "📝 Created files:"
echo "  - src/neural_nets/basic/feedforward.py (basic neural network)"
echo "  - src/neural_nets/backpropagation/backprop.py (backpropagation)"
echo "  - src/neural_nets/training/trainer.py (training module)"
echo "  - src/neural_nets/examples/simple_examples.py (training examples)"
echo "  - tests/test_neural_nets.py (unit tests)"
echo "  - demo_neural_nets.py (demo script)"
echo "  - requirements_neural_nets.txt (dependencies)"
echo ""
echo "🚀 Next steps:"
echo "  1. Install dependencies: pip install -r requirements_neural_nets.txt"
echo "  2. Run tests: python -m pytest tests/test_neural_nets.py"
echo "  3. Run demo: python demo_neural_nets.py"
echo "  4. Start implementing your neural network solutions!" 