"""
Complete Transformer Implementation with Encoder-Decoder Architecture
"""
import numpy as np
from typing import List, Tuple, Optional
from ..positional_encoding.positional_encoding import PositionalEncoding
from ..blocks.encoder_block import EncoderBlock
from ..blocks.decoder_block import DecoderBlock
from ..masking.masks import create_padding_mask, create_causal_mask

class CompleteTransformer:
    """
    Complete Transformer with Encoder-Decoder Architecture
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 d_ff: int = 2048, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize Complete Transformer
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of feed-forward networks
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        # Embeddings
        self.src_embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.tgt_embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Multiple encoder blocks
        self.encoder_blocks = [EncoderBlock(d_model, num_heads, d_ff, dropout) 
                              for _ in range(num_encoder_layers)]
        
        # Multiple decoder blocks
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff, dropout) 
                              for _ in range(num_decoder_layers)]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.1
        
        # Final layer normalization
        self.final_norm = LayerNormalization(d_model)
    
    def encode(self, src: np.ndarray, src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode source sequence through multiple encoder blocks
        
        Args:
            src: Source sequence (batch_size, src_seq_length)
            src_mask: Padding mask for source sequence
            
        Returns:
            Encoded sequence
        """
        batch_size, src_seq_length = src.shape
        
        # Embedding
        embedded = self.src_embedding[src]
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Pass through multiple encoder blocks
        encoded = embedded
        for i, encoder_block in enumerate(self.encoder_blocks):
            encoded = encoder_block.forward(encoded, src_mask)
            print(f"Encoder block {i+1} output shape: {encoded.shape}")
        
        return encoded
    
    def decode(self, tgt: np.ndarray, encoder_output: np.ndarray, 
               tgt_mask: Optional[np.ndarray] = None, 
               src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode target sequence through multiple decoder blocks
        
        Args:
            tgt: Target sequence (batch_size, tgt_seq_length)
            encoder_output: Output from encoder
            tgt_mask: Causal mask for target sequence
            src_mask: Padding mask for source sequence
            
        Returns:
            Decoded sequence
        """
        batch_size, tgt_seq_length = tgt.shape
        
        # Embedding
        embedded = self.tgt_embedding[tgt]
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Pass through multiple decoder blocks
        decoded = embedded
        for i, decoder_block in enumerate(self.decoder_blocks):
            decoded = decoder_block.forward(decoded, encoder_output, tgt_mask, src_mask)
            print(f"Decoder block {i+1} output shape: {decoded.shape}")
        
        # Final layer normalization
        decoded = self.final_norm.forward(decoded)
        
        return decoded
    
    def forward(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """
        Forward pass through Complete Transformer
        
        Args:
            src: Source sequence (batch_size, src_seq_length)
            tgt: Target sequence (batch_size, tgt_seq_length)
            
        Returns:
            Output logits
        """
        
        # Create masks
        # Padding mask for encoder (batch, 1, src_len) to broadcast over query length
        src_mask_4d = create_padding_mask(src)  # (batch, 1, 1, src_len)
        src_mask = np.squeeze(np.squeeze(src_mask_4d, axis=1), axis=1)  # (batch, src_len)
        src_mask = src_mask[:, None, :]  # (batch, 1, src_len)

        # Decoder target mask: causal + target padding mask
        tgt_len = tgt.shape[1]
        causal = np.triu(np.ones((tgt_len, tgt_len)), k=1) * -1e9  # (tgt_len, tgt_len)
        causal = causal[None, :, :]  # (1, tgt_len, tgt_len)
        tgt_pad_4d = create_padding_mask(tgt)  # (batch, 1, 1, tgt_len)
        tgt_pad = np.squeeze(np.squeeze(tgt_pad_4d, axis=1), axis=1)  # (batch, tgt_len)
        tgt_pad = tgt_pad[:, None, :]  # (batch, 1, tgt_len)
        tgt_mask = causal + tgt_pad  # broadcast to (batch, tgt_len, tgt_len)

        # Encode source sequence
        print("Encoding source sequence...")
        encoder_output = self.encode(src, src_mask=src_mask)

        # Decode target sequence
        print("Decoding target sequence...")
        decoder_output = self.decode(tgt, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
        
        # Project to vocabulary logits
        logits = np.dot(decoder_output, self.output_projection)
        
        return logits
    
    def generate(self, src: np.ndarray, max_length: int = 50, 
                start_token: int = 0, end_token: int = 1) -> List[int]:
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
        # Encode source sequence with padding mask
        src_mask_4d = create_padding_mask(src)
        src_mask = np.squeeze(np.squeeze(src_mask_4d, axis=1), axis=1)  # (batch, src_len)
        src_mask = src_mask[:, None, :]
        encoder_output = self.encode(src, src_mask=src_mask)
        
        # Initialize target sequence
        batch_size = src.shape[0]
        tgt = np.array([[start_token]] * batch_size)
        
        generated = [start_token]
        
        for step in range(max_length):
            # Create causal mask for current sequence (2D -> 3D for broadcast)
            L = tgt.shape[1]
            tgt_mask = np.triu(np.ones((L, L)), k=1) * -1e9
            tgt_mask = tgt_mask[None, :, :]  # (1, L, L)
            
            # Decode current sequence
            decoder_output = self.decode(tgt, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
            
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
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)