"""
Complete Transformer Implementation
"""
import numpy as np
from typing import List, Tuple, Optional
from ..positional_encoding.positional_encoding import PositionalEncoding
from ..encoder_decoder.encoder_layer import EncoderLayer
from ..encoder_decoder.decoder_layer import DecoderLayer

class Transformer:
    """
    Complete Transformer model
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_seq_length: int = 5000,
                 dropout: float = 0.1):
        """
        Initialize Transformer
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Dimension of feed-forward networks
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        # Embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder layers
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) 
                              for _ in range(num_layers)]
        
        # Decoder layers
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) 
                              for _ in range(num_layers)]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
        
        # Layer normalization
        self.norm = LayerNormalization(d_model)
    
    def _embed_with_pos(self, token_ids: np.ndarray) -> np.ndarray:
        """Lookup embeddings and add positional encoding. token_ids: (batch, seq_len)"""
        embedded = self.embedding[token_ids]  # (batch, seq_len, d_model)
        embedded = self.pos_encoding(embedded)
        return embedded

    def encode(self, x: np.ndarray, src_padding_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode input sequence
        
        Args:
            x: Input sequence (batch_size, seq_length)
            
        Returns:
            Encoded sequence
        """
        batch_size, seq_length = x.shape

        # Embedding + positional encoding
        embedded = self._embed_with_pos(x)
        
        # Pass through encoder layers
        encoded = embedded
        for layer in self.encoder_layers:
            encoded = layer.forward(encoded, padding_mask=src_padding_mask)
        
        return encoded
    
    def decode(
        self,
        x: np.ndarray,
        encoder_output: np.ndarray,
        tgt_padding_mask: Optional[np.ndarray] = None,
        src_padding_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decode sequence
        
        Args:
            x: Target sequence (batch_size, seq_length)
            encoder_output: Output from encoder
            
        Returns:
            Decoded sequence
        """
        batch_size, seq_length = x.shape

        # Embedding + positional encoding
        embedded = self._embed_with_pos(x)
        
        # Pass through decoder layers
        decoded = embedded
        for layer in self.decoder_layers:
            decoded = layer.forward(
                decoded,
                encoder_output,
                tgt_padding_mask=tgt_padding_mask,
                src_padding_mask=src_padding_mask,
            )
        
        # Final layer normalization
        decoded = self.norm.forward(decoded)
        
        return decoded
    
    def forward(self, src: np.ndarray, tgt: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
        """
        Forward pass through Transformer
        
        Args:
            src: Source sequence (batch_size, src_seq_length)
            tgt: Target sequence (batch_size, tgt_seq_length)
            
        Returns:
            Output logits
        """
        # For masking, pass token id sequences; padding id assumed pad_token_id
        src_tokens_for_mask = src
        tgt_tokens_for_mask = tgt

        # Encode source sequence
        encoder_output = self.encode(src, src_tokens_for_mask)
        
        # Shift target right for teacher forcing: input is all but last token
        # Keep same shape by slicing; logits will be computed for positions 1..T
        decoder_input = tgt
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            tgt_padding_mask=tgt_tokens_for_mask,
            src_padding_mask=src_tokens_for_mask,
        )
        
        # Project to vocabulary
        logits = np.dot(decoder_output, self.output_projection)
        
        return logits
    
    def generate(self, src: np.ndarray, max_length: int = 50, 
                start_token: int = 0, end_token: int = 1, pad_token_id: int = 0) -> List[int]:
        """
        Generate sequence autoregressively
        
        Args:
            src: Source sequence
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            Generated sequence
        """
        # Encode source sequence
        src_pad_mask = (src == pad_token_id).astype(np.int32)
        encoder_output = self.encode(src, src_pad_mask)
        
        # Initialize target sequence
        batch_size = src.shape[0]
        tgt = np.array([[start_token]] * batch_size)
        
        generated = [start_token]
        
        for _ in range(max_length):
            # Decode current sequence
            decoder_pad_mask = (tgt == pad_token_id).astype(np.int32)
            decoder_output = self.decode(
                tgt,
                encoder_output,
                tgt_padding_mask=decoder_pad_mask,
                src_padding_mask=src_pad_mask,
            )
            
            # Get next token
            logits = np.dot(decoder_output[:, -1], self.output_projection)
            next_token = np.argmax(logits)
            
            generated.append(next_token)
            
            # Stop if end token
            if next_token == end_token:
                break
            
            # Add to target sequence
            tgt = np.column_stack([tgt, [[next_token]] * batch_size])
        
        return generated

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
