"""
Self-Attention Mechanism Implementation
"""
import numpy as np
from typing import Tuple, List

class SelfAttention:
    """
    Self-attention mechanism where queries, keys, and values come from the same input
    """
    
    def __init__(self, input_dim: int, num_heads: int = 1):
        """
        Initialize self-attention
        
        Args:
            input_dim: Dimension of input vectors
            num_heads: Number of attention heads
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Initialize weight matrices for Q, K, V
        self.W_q = np.random.randn(input_dim, input_dim) * 0.01
        self.W_k = np.random.randn(input_dim, input_dim) * 0.01
        self.W_v = np.random.randn(input_dim, input_dim) * 0.01
        self.W_o = np.random.randn(input_dim, input_dim) * 0.01  # Output projection
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of self-attention
        
        Args:
            x: Input sequence (seq_len, input_dim)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        seq_len, input_dim = x.shape
        
        # Step 1: Linear transformations
        Q = np.dot(x, self.W_q)  # (seq_len, input_dim)
        K = np.dot(x, self.W_k)  # (seq_len, input_dim)
        V = np.dot(x, self.W_v)  # (seq_len, input_dim)
        
        # Step 2: Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim)
        
        # Step 3: Transpose for easier computation
        Q = Q.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        K = K.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        V = V.transpose(1, 0, 2)  # (num_heads, seq_len, head_dim)
        
        # Step 4: Compute attention for each head
        attention_outputs = []
        attention_weights = []
        
        for head in range(self.num_heads):
            # Compute attention scores
            scores = np.dot(Q[head], K[head].T) / np.sqrt(self.head_dim)
            
            # Apply softmax
            weights = self.softmax(scores)
            
            # Apply attention
            head_output = np.dot(weights, V[head])
            
            attention_outputs.append(head_output)
            attention_weights.append(weights)
        
        # Step 5: Concatenate heads
        attention_output = np.concatenate(attention_outputs, axis=1)  # (seq_len, input_dim)
        
        # Step 6: Final linear transformation
        output = np.dot(attention_output, self.W_o)
        
        # Average attention weights across heads
        avg_attention_weights = np.mean(attention_weights, axis=0)
        
        return output, avg_attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
