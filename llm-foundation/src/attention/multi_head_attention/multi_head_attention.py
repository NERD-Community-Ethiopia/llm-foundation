"""
Multi-Head Attention Implementation
"""
import numpy as np
from typing import Tuple, List, Optional

class MultiHeadAttention:
    """
    Multi-head attention mechanism with scaled dot-product attention
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear transformations for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
        
        # Initialize gradients
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)
        
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention
        
        Args:
            Q: Query matrix (batch_size, num_heads, seq_len_q, d_k)
            K: Key matrix (batch_size, num_heads, seq_len_k, d_k)
            V: Value matrix (batch_size, num_heads, seq_len_k, d_k)
            mask: Optional mask for padding or causal attention
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply dropout (simplified - just multiply by (1 - dropout))
        if self.dropout > 0:
            attention_weights = attention_weights * (1 - self.dropout)
        
        # Apply attention weights to values
        attention_output = np.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention
        
        Args:
            queries: Query matrix (batch_size, seq_len_q, d_model)
            keys: Key matrix (batch_size, seq_len_k, d_model)
            values: Value matrix (batch_size, seq_len_k, d_model)
            mask: Optional mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q, _ = queries.shape
        _, seq_len_k, _ = keys.shape
        
        # Linear transformations
        Q = np.dot(queries, self.W_q)  # (batch_size, seq_len_q, d_model)
        K = np.dot(keys, self.W_k)     # (batch_size, seq_len_k, d_model)
        V = np.dot(values, self.W_v)   # (batch_size, seq_len_k, d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        
        # Final linear transformation
        output = np.dot(attention_output, self.W_o)
        
        return output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask for decoder self-attention
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask (seq_len, seq_len)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9  # Large negative value
        return mask
