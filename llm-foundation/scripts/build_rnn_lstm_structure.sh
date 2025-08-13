#!/bin/bash

# Build Course Structure for RNN & LSTM (Checkpoint 2)
# This script creates the folder structure and initial files for the RNN & LSTM section

echo "🚀 Building RNN & LSTM Course Structure..."

# Create RNN & LSTM specific directory
mkdir -p src/rnn_lstm

# Create RNN & LSTM specific subdirectories
mkdir -p src/rnn_lstm/basic_rnn
mkdir -p src/rnn_lstm/lstm
mkdir -p src/rnn_lstm/sequence_models
mkdir -p src/rnn_lstm/examples

# Create __init__.py files for Python packages
touch src/rnn_lstm/__init__.py
touch src/rnn_lstm/basic_rnn/__init__.py
touch src/rnn_lstm/lstm/__init__.py
touch src/rnn_lstm/sequence_models/__init__.py
touch src/rnn_lstm/examples/__init__.py

# Create main RNN & LSTM implementation files
echo "📝 Creating RNN & LSTM implementation files..."

# Basic RNN implementation
cat > src/rnn_lstm/basic_rnn/rnn.py << 'EOF'
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
            self.dW_hh += np.dot(dh_raw, hidden_states[t-1].T if t > 0 else np.zeros((self.hidden_size, batch_size)))
            
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
EOF

# LSTM implementation
cat > src/rnn_lstm/lstm/lstm.py << 'EOF'
"""
Long Short-Term Memory (LSTM) Implementation
"""
import numpy as np
from typing import List, Tuple, Optional

class LSTM:
    """
    Long Short-Term Memory (LSTM) implementation
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the LSTM
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            output_size: Size of output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights for all gates
        # Input gate
        self.W_xi = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.W_xf = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        # Output gate
        self.W_xo = np.random.randn(hidden_size, input_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        
        # Cell state
        self.W_xc = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output layer
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
        # Initialize gradients
        self._init_gradients()
    
    def _init_gradients(self):
        """Initialize gradient storage"""
        self.dW_xi = np.zeros_like(self.W_xi)
        self.dW_hi = np.zeros_like(self.W_hi)
        self.db_i = np.zeros_like(self.b_i)
        
        self.dW_xf = np.zeros_like(self.W_xf)
        self.dW_hf = np.zeros_like(self.W_hf)
        self.db_f = np.zeros_like(self.b_f)
        
        self.dW_xo = np.zeros_like(self.W_xo)
        self.dW_ho = np.zeros_like(self.W_ho)
        self.db_o = np.zeros_like(self.b_o)
        
        self.dW_xc = np.zeros_like(self.W_xc)
        self.dW_hc = np.zeros_like(self.W_hc)
        self.db_c = np.zeros_like(self.b_c)
        
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
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
    
    def forward(self, inputs: List[np.ndarray], h0: Optional[np.ndarray] = None, 
                c0: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the LSTM
        
        Args:
            inputs: List of input sequences
            h0: Initial hidden state
            c0: Initial cell state
            
        Returns:
            Tuple of (outputs, hidden_states, cell_states)
        """
        seq_length = len(inputs)
        batch_size = inputs[0].shape[1]
        
        # Initialize states
        if h0 is None:
            h = np.zeros((self.hidden_size, batch_size))
        else:
            h = h0
            
        if c0 is None:
            c = np.zeros((self.hidden_size, batch_size))
        else:
            c = c0
        
        # Storage for states
        hidden_states = []
        cell_states = []
        outputs = []
        
        # Process each time step
        for t in range(seq_length):
            x_t = inputs[t]
            
            # Input gate: i_t = σ(W_xi * x_t + W_hi * h_{t-1} + b_i)
            i_t = self.sigmoid(np.dot(self.W_xi, x_t) + np.dot(self.W_hi, h) + self.b_i)
            
            # Forget gate: f_t = σ(W_xf * x_t + W_hf * h_{t-1} + b_f)
            f_t = self.sigmoid(np.dot(self.W_xf, x_t) + np.dot(self.W_hf, h) + self.b_f)
            
            # Output gate: o_t = σ(W_xo * x_t + W_ho * h_{t-1} + b_o)
            o_t = self.sigmoid(np.dot(self.W_xo, x_t) + np.dot(self.W_ho, h) + self.b_o)
            
            # Cell candidate: c̃_t = tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)
            c_tilde = self.tanh(np.dot(self.W_xc, x_t) + np.dot(self.W_hc, h) + self.b_c)
            
            # Cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
            c = f_t * c + i_t * c_tilde
            
            # Hidden state: h_t = o_t * tanh(c_t)
            h = o_t * self.tanh(c)
            
            # Output: y_t = softmax(W_hy * h_t + b_y)
            y_t = self.softmax(np.dot(self.W_hy, h) + self.b_y)
            
            # Store states
            hidden_states.append(h)
            cell_states.append(c)
            outputs.append(y_t)
        
        return outputs, hidden_states, cell_states
    
    def backward(self, inputs: List[np.ndarray], hidden_states: List[np.ndarray], 
                cell_states: List[np.ndarray], outputs: List[np.ndarray], 
                targets: List[np.ndarray]) -> None:
        """
        Backward pass through the LSTM (BPTT)
        
        Args:
            inputs: List of input sequences
            hidden_states: List of hidden states from forward pass
            cell_states: List of cell states from forward pass
            outputs: List of outputs from forward pass
            targets: List of target sequences
        """
        seq_length = len(inputs)
        batch_size = inputs[0].shape[1]
        
        # Initialize gradients
        dh_next = np.zeros((self.hidden_size, batch_size))
        dc_next = np.zeros((self.hidden_size, batch_size))
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            # Gradient of loss with respect to output
            dy = outputs[t] - targets[t]
            
            # Gradient of loss with respect to W_hy and b_y
            self.dW_hy += np.dot(dy, hidden_states[t].T)
            self.db_y += np.sum(dy, axis=1, keepdims=True)
            
            # Gradient of loss with respect to hidden state
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Gradient of loss with respect to cell state
            dc = dc_next + dh * self.tanh(cell_states[t])
            
            # Gate gradients
            do = dh * self.tanh(cell_states[t])
            dc_tilde = dh * self.sigmoid(np.dot(self.W_xo, inputs[t]) + np.dot(self.W_ho, hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size))) + self.b_o)
            di = dc * self.tanh(np.dot(self.W_xc, inputs[t]) + np.dot(self.W_hc, hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size))) + self.b_c)
            df = dc * (cell_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size)))
            
            # Update weight gradients
            self._update_gradients(inputs[t], hidden_states[t-1] if t > 0 else np.zeros((self.hidden_size, batch_size)), 
                                 di, df, do, dc_tilde)
            
            # Prepare for next iteration
            dh_next = (np.dot(self.W_hi.T, di) + np.dot(self.W_hf.T, df) + 
                      np.dot(self.W_ho.T, do) + np.dot(self.W_hc.T, dc_tilde))
            dc_next = df
    
    def _update_gradients(self, x_t, h_prev, di, df, do, dc_tilde):
        """Update gradients for all gates"""
        # Input gate
        self.dW_xi += np.dot(di, x_t.T)
        self.dW_hi += np.dot(di, h_prev.T)
        self.db_i += np.sum(di, axis=1, keepdims=True)
        
        # Forget gate
        self.dW_xf += np.dot(df, x_t.T)
        self.dW_hf += np.dot(df, h_prev.T)
        self.db_f += np.sum(df, axis=1, keepdims=True)
        
        # Output gate
        self.dW_xo += np.dot(do, x_t.T)
        self.dW_ho += np.dot(do, h_prev.T)
        self.db_o += np.sum(do, axis=1, keepdims=True)
        
        # Cell state
        self.dW_xc += np.dot(dc_tilde, x_t.T)
        self.dW_hc += np.dot(dc_tilde, h_prev.T)
        self.db_c += np.sum(dc_tilde, axis=1, keepdims=True)
    
    def update_weights(self, learning_rate: float) -> None:
        """Update weights using computed gradients"""
        self.W_xi -= learning_rate * self.dW_xi
        self.W_hi -= learning_rate * self.dW_hi
        self.b_i -= learning_rate * self.db_i
        
        self.W_xf -= learning_rate * self.dW_xf
        self.W_hf -= learning_rate * self.dW_hf
        self.b_f -= learning_rate * self.db_f
        
        self.W_xo -= learning_rate * self.dW_xo
        self.W_ho -= learning_rate * self.dW_ho
        self.b_o -= learning_rate * self.db_o
        
        self.W_xc -= learning_rate * self.dW_xc
        self.W_hc -= learning_rate * self.dW_hc
        self.b_c -= learning_rate * self.db_c
        
        self.W_hy -= learning_rate * self.dW_hy
        self.b_y -= learning_rate * self.db_y
        
        # Reset gradients
        self._init_gradients()
    
    def predict(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Make predictions using the trained LSTM"""
        outputs, _, _ = self.forward(inputs)
        return outputs
EOF

# Sequence models utilities
cat > src/rnn_lstm/sequence_models/sequence_utils.py << 'EOF'
"""
Utilities for sequence modeling
"""
import numpy as np
from typing import List, Tuple, Dict
import random

def create_sequences(data: List[int], seq_length: int) -> Tuple[List[List[int]], List[int]]:
    """
    Create sequences for training
    
    Args:
        data: List of integers (e.g., text as character indices)
        seq_length: Length of each sequence
        
    Returns:
        Tuple of (input_sequences, target_sequences)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets

def one_hot_encode(sequences: List[List[int]], vocab_size: int) -> List[np.ndarray]:
    """
    Convert sequences to one-hot encoded format
    
    Args:
        sequences: List of sequences
        vocab_size: Size of vocabulary
        
    Returns:
        List of one-hot encoded sequences
    """
    encoded_sequences = []
    
    for seq in sequences:
        encoded_seq = []
        for token in seq:
            one_hot = np.zeros((vocab_size, 1))
            one_hot[token] = 1
            encoded_seq.append(one_hot)
        encoded_sequences.append(encoded_seq)
    
    return encoded_sequences

def create_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create vocabulary from text
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (char_to_idx, idx_to_char) dictionaries
    """
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return char_to_idx, idx_to_char

def text_to_sequences(text: str, char_to_idx: Dict[str, int]) -> List[int]:
    """
    Convert text to sequence of indices
    
    Args:
        text: Input text
        char_to_idx: Character to index mapping
        
    Returns:
        List of indices
    """
    return [char_to_idx[ch] for ch in text]

def sequences_to_text(sequences: List[int], idx_to_char: Dict[int, str]) -> str:
    """
    Convert sequence of indices back to text
    
    Args:
        sequences: List of indices
        idx_to_char: Index to character mapping
        
    Returns:
        Text string
    """
    return ''.join([idx_to_char[idx] for idx in sequences])

def sample_from_model(model, seed_text: str, char_to_idx: Dict[str, int], 
                     idx_to_char: Dict[int, str], length: int = 100, 
                     temperature: float = 1.0) -> str:
    """
    Generate text using trained model
    
    Args:
        model: Trained RNN/LSTM model
        seed_text: Starting text
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        length: Length of text to generate
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated text
    """
    vocab_size = len(char_to_idx)
    generated_text = seed_text
    
    # Convert seed text to sequence
    seed_sequence = text_to_sequences(seed_text, char_to_idx)
    
    for _ in range(length):
        # Prepare input
        x = np.array(seed_sequence[-1]).reshape(-1, 1)
        x_one_hot = np.zeros((vocab_size, 1))
        x_one_hot[x] = 1
        
        # Get prediction
        output = model.predict([x_one_hot])[0]
        
        # Apply temperature
        output = np.log(output) / temperature
        output = np.exp(output)
        output = output / np.sum(output)
        
        # Sample next character
        next_char_idx = np.random.choice(vocab_size, p=output.flatten())
        next_char = idx_to_char[next_char_idx]
        
        generated_text += next_char
        seed_sequence.append(next_char_idx)
    
    return generated_text
EOF

# Training examples
cat > src/rnn_lstm/examples/rnn_examples.py << 'EOF'
"""
RNN Training Examples
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from rnn_lstm.basic_rnn.rnn import SimpleRNN
from rnn_lstm.lstm.lstm import LSTM
from rnn_lstm.sequence_models.sequence_utils import (
    create_sequences, one_hot_encode, create_vocab,
    text_to_sequences, sample_from_model
)

def simple_sequence_example():
    """
    Train RNN on a simple sequence prediction task
    """
    print("🔍 Training RNN on Simple Sequence Prediction")
    
    # Create simple sequence: [1, 2, 3, 1, 2, 3, ...]
    data = [1, 2, 3] * 100
    seq_length = 3
    vocab_size = 4  # 0, 1, 2, 3
    
    # Create sequences
    sequences, targets = create_sequences(data, seq_length)
    
    # Convert to one-hot encoding
    X = []
    y = []
    
    for seq, target in zip(sequences, targets):
        # Convert sequence to one-hot
        seq_one_hot = []
        for token in seq:
            one_hot = np.zeros((vocab_size, 1))
            one_hot[token] = 1
            seq_one_hot.append(one_hot)
        X.append(seq_one_hot)
        
        # Convert target to one-hot
        target_one_hot = np.zeros((vocab_size, 1))
        target_one_hot[target] = 1
        y.append(target_one_hot)
    
    # Create RNN
    rnn = SimpleRNN(input_size=vocab_size, hidden_size=10, output_size=vocab_size)
    
    # Training
    learning_rate = 0.1
    epochs = 100
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(X)):
            # Forward pass
            outputs, _ = rnn.forward(X[i])
            
            # Compute loss (cross-entropy)
            loss = -np.sum(y[i] * np.log(outputs[-1] + 1e-8))
            total_loss += loss
            
            # Backward pass
            rnn.backward(X[i], [np.zeros((10, 1))] * len(X[i]), outputs, [y[i]])
            rnn.update_weights(learning_rate)
        
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('RNN Training Loss - Simple Sequence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('plots/rnn_simple_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Final Loss: {losses[-1]:.4f}")
    return rnn, losses

def text_generation_example():
    """
    Train LSTM on text generation task
    """
    print("\n🔍 Training LSTM on Text Generation")
    
    # Simple text for demonstration
    text = """
    hello world this is a simple text generation example
    we will train an lstm to learn to generate similar text
    the model should learn patterns in the text and be able
    to generate new text that follows similar patterns
    """ * 10  # Repeat to have more data
    
    # Create vocabulary
    char_to_idx, idx_to_char = create_vocab(text)
    vocab_size = len(char_to_idx)
    
    # Convert text to sequences
    data = text_to_sequences(text, char_to_idx)
    seq_length = 20
    
    # Create sequences
    sequences, targets = create_sequences(data, seq_length)
    
    # Convert to one-hot encoding
    X = []
    y = []
    
    for seq, target in zip(sequences, targets):
        # Convert sequence to one-hot
        seq_one_hot = []
        for token in seq:
            one_hot = np.zeros((vocab_size, 1))
            one_hot[token] = 1
            seq_one_hot.append(one_hot)
        X.append(seq_one_hot)
        
        # Convert target to one-hot
        target_one_hot = np.zeros((vocab_size, 1))
        target_one_hot[target] = 1
        y.append(target_one_hot)
    
    # Create LSTM
    lstm = LSTM(input_size=vocab_size, hidden_size=50, output_size=vocab_size)
    
    # Training
    learning_rate = 0.01
    epochs = 50
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(X)):
            # Forward pass
            outputs, _, _ = lstm.forward(X[i])
            
            # Compute loss
            loss = -np.sum(y[i] * np.log(outputs[-1] + 1e-8))
            total_loss += loss
            
            # Backward pass
            lstm.backward(X[i], [np.zeros((50, 1))] * len(X[i]), 
                         [np.zeros((50, 1))] * len(X[i]), outputs, [y[i]])
            lstm.update_weights(learning_rate)
        
        avg_loss = total_loss / len(X)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('LSTM Training Loss - Text Generation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('plots/lstm_text_generation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate some text
    seed_text = "hello world"
    generated_text = sample_from_model(lstm, seed_text, char_to_idx, idx_to_char, 
                                     length=50, temperature=0.8)
    
    print(f"\nGenerated text: {generated_text}")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    return lstm, losses, generated_text

if __name__ == "__main__":
    print("🚀 Running RNN & LSTM Examples\n")
    
    # Simple sequence example
    rnn, rnn_losses = simple_sequence_example()
    
    # Text generation example
    lstm, lstm_losses, generated_text = text_generation_example()
    
    print("\n✅ All RNN & LSTM examples completed!") 