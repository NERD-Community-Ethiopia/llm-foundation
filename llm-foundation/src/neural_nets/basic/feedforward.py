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
    
    def backward_pass(self, activations, z_values, weights, y_true, network, loss_type='mse'):
        """Backward pass using the Backpropagation class"""
        from ..backpropagation.backprop import Backpropagation
        return Backpropagation.backward_pass(activations, z_values, weights, y_true, network, loss_type)
    
    def compute_loss(self, y_true, y_pred, loss_type='mse'):
        """Compute loss using the Backpropagation class"""
        from ..backpropagation.backprop import Backpropagation
        return Backpropagation.compute_loss(y_true, y_pred, loss_type)
