"""
Encoder Layer Implementation
"""
import numpy as np
from typing import Tuple, Optional
from ..attention_layers.multi_head_attention import MultiHeadAttention

class EncoderLayer:
    """
    Single encoder layer
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize encoder layer
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ff_network = FeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
    
    def forward(self, x: np.ndarray, src_key_padding_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through encoder layer
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Prepare mask shape broadcastable to (batch, q_len, k_len): use (batch, 1, k_len)
        mask = None
        if src_key_padding_mask is not None:
            if src_key_padding_mask.ndim == 2:
                pad = src_key_padding_mask.astype(float) * -1e9  # (batch, k_len)
                mask = pad[:, None, :]  # (batch, 1, k_len)
            else:
                mask = src_key_padding_mask

        # Self-attention with residual connection
        attn_output, _ = self.self_attention.forward(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff_network.forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class FeedForward:
    """Feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(np.maximum(0, np.dot(x, self.W1) + self.b1), self.W2) + self.b2

class LayerNormalization:
    """Layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Make the class callable"""
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta
