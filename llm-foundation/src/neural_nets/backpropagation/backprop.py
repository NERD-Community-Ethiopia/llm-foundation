"""
Backpropagation Implementation
"""
import numpy as np
from typing import List, Tuple
from ..basic.feedforward import FeedforwardNeuralNetwork

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
