"""
Basic Attention Mechanism Implementation
"""
import numpy as np
from typing import Tuple, List

class BasicAttention:
    """
    Basic attention mechanism implementation
    """
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int):
        """
        Initialize attention mechanism
        
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors  
            value_dim: Dimension of value vectors
        """
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Initialize weight matrices
        self.W_q = np.random.randn(query_dim, query_dim) * 0.01
        self.W_k = np.random.randn(key_dim, key_dim) * 0.01
        self.W_v = np.random.randn(value_dim, value_dim) * 0.01
        
    def forward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of attention mechanism
        
        Args:
            queries: Query vectors (seq_len_q, query_dim)
            keys: Key vectors (seq_len_k, key_dim)
            values: Value vectors (seq_len_k, value_dim)
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Step 1: Transform queries, keys, and values
        Q = np.dot(queries, self.W_q)  # (seq_len_q, query_dim)
        K = np.dot(keys, self.W_k)     # (seq_len_k, key_dim)
        V = np.dot(values, self.W_v)   # (seq_len_k, value_dim)
        
        # Step 2: Compute attention scores
        # scores = Q * K^T / sqrt(d_k)
        scores = np.dot(Q, K.T) / np.sqrt(self.key_dim)  # (seq_len_q, seq_len_k)
        
        # Step 3: Apply softmax to get attention weights
        attention_weights = self.softmax(scores)  # (seq_len_q, seq_len_k)
        
        # Step 4: Apply attention weights to values
        attention_output = np.dot(attention_weights, V)  # (seq_len_q, value_dim)
        
        return attention_output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray, 
                attention_weights: np.ndarray, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of attention mechanism
        
        Args:
            queries: Query vectors from forward pass
            keys: Key vectors from forward pass
            values: Value vectors from forward pass
            attention_weights: Attention weights from forward pass
            grad_output: Gradient of output
            
        Returns:
            Tuple of gradients for queries, keys, and values
        """
        # Transform inputs
        Q = np.dot(queries, self.W_q)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)
        
        # Gradient with respect to attention weights
        grad_attention_weights = np.dot(grad_output, V.T)
        
        # Gradient with respect to values
        grad_V = np.dot(attention_weights.T, grad_output)
        
        # Gradient with respect to scores
        grad_scores = self.softmax_gradient(attention_weights, grad_attention_weights)
        
        # Gradient with respect to Q and K
        grad_Q = np.dot(grad_scores, K) / np.sqrt(self.key_dim)
        grad_K = np.dot(grad_scores.T, Q) / np.sqrt(self.key_dim)
        
        # Gradient with respect to original inputs
        grad_queries = np.dot(grad_Q, self.W_q.T)
        grad_keys = np.dot(grad_K, self.W_k.T)
        grad_values = np.dot(grad_V, self.W_v.T)
        
        return grad_queries, grad_keys, grad_values
    
    def softmax_gradient(self, softmax_output: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradient of softmax function"""
        # This is a simplified version - in practice, you'd need the full Jacobian
        return grad_output * softmax_output * (1 - softmax_output)
