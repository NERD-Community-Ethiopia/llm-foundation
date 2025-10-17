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
