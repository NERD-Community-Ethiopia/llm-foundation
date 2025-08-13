"""
Decoder Layer Implementation
"""
import numpy as np
from typing import Tuple, Optional
from ..attention_layers.multi_head_attention import MultiHeadAttention

class DecoderLayer:
    """
    Single decoder layer
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize decoder layer
        
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
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.ff_network = FeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
    
    def forward(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        tgt_padding_mask: Optional[np.ndarray] = None,
        src_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through decoder layer
        
        Args:
            x: Input tensor
            encoder_output: Output from encoder
            
        Returns:
            Output tensor
        """
        # Create causal mask for self-attention and combine with target padding mask
        seq_length = x.shape[1]
        causal_mask = self.self_attention.create_causal_mask(seq_length)  # (1,1,T,T)
        attn_mask = causal_mask
        if tgt_padding_mask is not None:
            # (batch, 1, 1, T)
            pad_mask = self.self_attention.create_padding_mask(tgt_padding_mask)
            # Broadcast add: keep causal and pad constraints
            attn_mask = attn_mask + pad_mask
        
        # Masked self-attention with residual connection
        attn_output, _ = self.self_attention.forward(x, x, x, attn_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention with residual connection
        # Cross-attention uses source padding mask to avoid attending to PAD in encoder outputs
        cross_mask = None
        if src_padding_mask is not None:
            cross_mask = self.cross_attention.create_padding_mask(src_padding_mask)
        cross_output, _ = self.cross_attention.forward(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + cross_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff_network.forward(x)
        x = self.norm3(x + ff_output)
        
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
