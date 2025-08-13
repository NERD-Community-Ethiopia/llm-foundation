"""
Decoder Block Implementation
"""
import numpy as np
from typing import Optional
from ..attention_layers.multi_head_attention import MultiHeadAttention

class DecoderBlock:
    """
    Single decoder block with self-attention, cross-attention, and feed-forward
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Multi-head self-attention (causal) - no dropout arg
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Multi-head cross-attention - no dropout arg
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ff_network = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        
        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray, 
                tgt_mask: Optional[np.ndarray] = None, 
                src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Self-attention with causal mask (residual + norm)
        attn_output, _ = self.self_attention.forward(x, x, x, tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1.forward(x + attn_output)
        
        # Cross-attention with encoder output (residual + norm)
        cross_attn_output, _ = self.cross_attention.forward(x, encoder_output, encoder_output, src_mask)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = self.norm2.forward(x + cross_attn_output)
        
        # Feed-forward (residual + norm)
        ff_output = self.ff_network.forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.norm3.forward(x + ff_output)
        
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
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        hidden = self.dropout(hidden)
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
