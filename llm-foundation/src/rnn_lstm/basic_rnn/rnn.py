"""
Basic Recurrent Neural Network (RNN) Implementation
"""
import numpy as np
from typing import List, Tuple, Optional

class SimpleRNN:
    """
    A simple Recurrent Neural Network implementation
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 activation='tanh', output_activation='softmax'):
        """
        Initialize the RNN
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
            activation: Hidden layer activation function
            output_activation: Output layer activation function
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.output_activation = output_activation
        
        # Initialize weights
        # W_xh: input to hidden weights
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        # W_hh: hidden to hidden weights (recurrent)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        # W_hy: hidden to output weights
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Initialize gradients
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function"""
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of softmax function"""
        s = self.softmax(x)
        return s * (1 - s)
    
    def forward(self, inputs: List[np.ndarray], h0: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the RNN
        
        Args:
            inputs: List of input sequences, each of shape (input_size, batch_size)
            h0: Initial hidden state, if None will be initialized to zeros
            
        Returns:
            Tuple of (outputs, hidden_states)
        """
        seq_length = len(inputs)
        batch_size = inputs[0].shape[1]
        
        # Initialize hidden states and outputs
        hidden_states = []
        outputs = []
        
        # Initialize hidden state
        if h0 is None:
            h = np.zeros((self.hidden_size, batch_size))
        else:
            h = h0
        
        # Process each time step
        for t in range(seq_length):
            # Hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h = self.tanh(np.dot(self.W_xh, inputs[t]) + 
                         np.dot(self.W_hh, h) + self.b_h)
            hidden_states.append(h)
            
            # Output: y_t = softmax(W_hy * h_t + b_y)
            y = self.softmax(np.dot(self.W_hy, h) + self.b_y)
            outputs.append(y)
        
        return outputs, hidden_states
    
    def backward(self, inputs: List[np.ndarray], hidden_states: List[np.ndarray], 
                outputs: List[np.ndarray], targets: List[np.ndarray]) -> None:
        """
        Backward pass through the RNN (Backpropagation Through Time - BPTT)
        
        Args:
            inputs: List of input sequences
            hidden_states: List of hidden states from forward pass
            outputs: List of outputs from forward pass
            targets: List of target sequences
        """
        seq_length = len(inputs)
        batch_size = inputs[0].shape[1]
        
        # Initialize gradients
        dh_next = np.zeros((self.hidden_size, batch_size))
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            # Gradient of loss with respect to output
            dy = outputs[t] - targets[t]
            
            # Gradient of loss with respect to W_hy and b_y
            self.dW_hy += np.dot(dy, hidden_states[t].T)
            self.db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient of loss with respect to hidden state
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Gradient of loss with respect to tanh input
            dh_raw = self.tanh_derivative(hidden_states[t]) * dh
            
            # Gradient of loss with respect to biases
            self.db_h += np.sum(dh_raw, axis=1, keepdims=True)
            
            # Gradient of loss with respect to weights
            self.dW_xh += np.dot(dh_raw, inputs[t].T)
            if t > 0:
                self.dW_hh += np.dot(dh_raw, hidden_states[t-1].T)
            
            
            # Gradient of loss with respect to next hidden state
            dh_next = np.dot(self.W_hh.T, dh_raw)
    
    def update_weights(self, learning_rate: float) -> None:
        """Update weights using computed gradients"""
        self.W_xh -= learning_rate * self.dW_xh
        self.W_hh -= learning_rate * self.dW_hh
        self.W_hy -= learning_rate * self.dW_hy
        self.b_h -= learning_rate * self.db_h
        self.b_y -= learning_rate * self.db_y
        
        # Reset gradients
        self.dW_xh.fill(0)
        self.dW_hh.fill(0)
        self.dW_hy.fill(0)
        self.db_h.fill(0)
        self.db_y.fill(0)
    
    def predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Make predictions using the trained RNN"""
        outputs, _ = self.forward(inputs)
        return outputs
