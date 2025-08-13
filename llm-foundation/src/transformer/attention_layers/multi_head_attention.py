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

        # Gradients (placeholders for completeness)
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

    @staticmethod
    def create_padding_mask(sequence: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
        """
        Create a padding mask for attention scores.

        Args:
            sequence: (batch_size, seq_len)
            pad_token_id: token id used for padding

        Returns:
            mask of shape (batch_size, 1, 1, seq_len) with 0 for keep and -1e9 for mask
        """
        mask = (sequence == pad_token_id).astype(np.float32)  # 1 where pad
        mask = mask[:, np.newaxis, np.newaxis, :]  # (batch, 1, 1, seq_len)
        return mask * -1e9

    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        mask = mask * -1e9
        return mask[np.newaxis, np.newaxis, :, :]

    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention

        Args:
            Q: (batch_size, num_heads, seq_len_q, d_k)
            K: (batch_size, num_heads, seq_len_k, d_k)
            V: (batch_size, num_heads, seq_len_k, d_k)
            mask: Optional mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            (attention_output, attention_weights)
        """
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        if mask is not None:
            # Broadcast add
            scores = scores + mask

        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)

    def forward(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through multi-head attention

        Args:
            queries: (batch_size, seq_len_q, d_model)
            keys: (batch_size, seq_len_k, d_model)
            values: (batch_size, seq_len_k, d_model)
            mask: Optional mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len_q, _ = queries.shape
        _, seq_len_k, _ = keys.shape

        # Linear projections
        Q = np.dot(queries, self.W_q)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)

        # Reshape to heads
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)

        # Final linear
        output = np.dot(attn_output, self.W_o)
        return output, attn_weights

    def backward(
        self,
        d_output: np.ndarray,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        attention_weights: list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Placeholder backward pass; full implementation omitted in this checkpoint.
        """
        return np.zeros_like(queries), np.zeros_like(keys), np.zeros_like(values)
