"""
Multi-Head Attention Implementation
"""
import numpy as np
from typing import Tuple, Optional

class MultiHeadAttention:
    """
    Multi-head attention mechanism
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize multi-head attention
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        # Weight matrices for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        # Gradients
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, 
                                   V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention
        
        Args:
            Q: Query matrix of shape (batch_size, seq_length, d_k)
            K: Key matrix of shape (batch_size, seq_length, d_k)
            V: Value matrix of shape (batch_size, seq_length, d_k)
            mask: Optional mask for causal attention
            
        Returns:
            Attention output and attention weights
        """
        # Compute attention scores: (batch_size, q_len, k_len)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Support mask shapes: (batch, 1, k_len) or (1, q_len, k_len) or (batch, q_len, k_len)
            if mask.ndim == 3:
                scores = scores + mask
            else:
                # If provided as (k_len, k_len) causal, expand to (1, q_len, k_len)
                scores = scores + mask[None, :, :]
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Apply to values: (batch_size, seq_length, d_k)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def create_causal_mask(self, seq_length: int) -> np.ndarray:
        """
        Create causal mask for decoder self-attention
        
        Args:
            seq_length: Length of the sequence
            
        Returns:
            Causal mask matrix
        """
        mask = np.triu(np.ones((seq_length, seq_length)), k=1)
        mask = mask * -1e9  # Large negative number
        return mask
    
    def forward(self, queries: np.ndarray, keys: np.ndarray, values: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through multi-head attention
        
        Args:
            queries: Query matrix
            keys: Key matrix
            values: Value matrix
            mask: Optional mask
            
        Returns:
            Attention output and attention weights
        """
        # Expect (batch, seq_len, d_model)
        batch_size, seq_length, d_model = queries.shape
        
        # Linear transformations
        Q = np.dot(queries, self.W_q)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)
        
        # Reshape for multi-head attention
        # Handle different sequence lengths for cross-attention
        q_seq_length = Q.shape[1]
        k_seq_length = K.shape[1]
        v_seq_length = V.shape[1]
        
        Q = Q.reshape(batch_size, q_seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, k_seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, v_seq_length, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply attention for each head
        attention_outputs = []
        attention_weights = []
        
        for head in range(self.num_heads):
            head_output, head_weights = self.scaled_dot_product_attention(
                Q[:, head], K[:, head], V[:, head], mask
            )
            attention_outputs.append(head_output)
            attention_weights.append(head_weights)
        
        # Concatenate heads along the last dimension
        attention_output = np.concatenate(attention_outputs, axis=-1)
        
        # Final linear transformation
        output = np.dot(attention_output, self.W_o)
        
        return output, attention_weights
    
    def backward(self, d_output: np.ndarray, queries: np.ndarray, keys: np.ndarray, 
                values: np.ndarray, attention_weights: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through multi-head attention
        
        Args:
            d_output: Gradient of output
            queries: Query matrix
            keys: Key matrix
            values: Value matrix
            attention_weights: Attention weights from forward pass
            
        Returns:
            Gradients for queries, keys, and values
        """
        # This is a simplified backward pass
        # In practice, you'd need to implement the full backpropagation
        
        # Compute gradients for weight matrices
        self.dW_o += np.dot(attention_weights[0].T, d_output)
        
        return np.zeros_like(queries), np.zeros_like(keys), np.zeros_like(values)
