"""
Encoder Block Implementation
"""
import numpy as np
from typing import Optional
from ..attention_layers.multi_head_attention import MultiHeadAttention

class EncoderBlock:
    """
    Single encoder block with self-attention and feed-forward
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize encoder block
        
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
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ff_network = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        
        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through encoder block
        
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            mask: Optional mask for padding
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection and normalization
        attn_output, _ = self.self_attention.forward(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.ff_network.forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2.forward(x + ff_output)
        
        return x

class FeedForward:
    """Feed-forward network with dropout"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # First linear transformation with ReLU
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        hidden = self.dropout(hidden)
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        return output

class LayerNormalization:
    """Layer normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Dropout:
    """Dropout layer"""
    
    def __init__(self, rate: float = 0.1):
        self.rate = rate
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.rate > 0:
            mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * mask
        return x
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
