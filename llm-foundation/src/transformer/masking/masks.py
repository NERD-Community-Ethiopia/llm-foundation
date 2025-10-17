"""
Masking Implementation for Transformer
"""
import numpy as np
from typing import Optional

def create_padding_mask(sequences: np.ndarray, pad_token: int = 0) -> np.ndarray:
    """
    Create padding mask for sequences
    
    Args:
        sequences: Input sequences (batch_size, seq_length)
        pad_token: Padding token ID
        
    Returns:
        Padding mask (batch_size, 1, 1, seq_length)
    """
    batch_size, seq_length = sequences.shape
    
    # Create mask where pad_token positions are True (will be masked)
    mask = (sequences == pad_token).astype(np.float32)
    
    # Reshape for attention: (batch_size, 1, 1, seq_length)
    mask = mask.reshape(batch_size, 1, 1, seq_length)
    
    # Convert to large negative values for softmax
    mask = mask * -1e9
    
    return mask

def create_causal_mask(seq_length: int) -> np.ndarray:
    """
    Create causal mask for decoder self-attention
    
    Args:
        seq_length: Length of the sequence
        
    Returns:
        Causal mask (1, 1, seq_length, seq_length)
    """
    # Create upper triangular matrix (including diagonal)
    mask = np.triu(np.ones((seq_length, seq_length)), k=1)
    
    # Convert to large negative values for softmax
    mask = mask * -1e9
    
    # Reshape for attention: (1, 1, seq_length, seq_length)
    mask = mask.reshape(1, 1, seq_length, seq_length)
    
    return mask

def create_combined_mask(tgt_sequences: np.ndarray, src_sequences: np.ndarray, 
                        pad_token: int = 0) -> tuple:
    """
    Create both padding and causal masks for decoder
    
    Args:
        tgt_sequences: Target sequences (batch_size, tgt_seq_length)
        src_sequences: Source sequences (batch_size, src_seq_length)
        pad_token: Padding token ID
        
    Returns:
        Tuple of (tgt_mask, src_mask)
    """
    # Padding mask for target sequences
    tgt_padding_mask = create_padding_mask(tgt_sequences, pad_token)
    
    # Causal mask for target sequences
    tgt_causal_mask = create_causal_mask(tgt_sequences.shape[1])
    
    # Combine padding and causal masks
    tgt_mask = tgt_padding_mask + tgt_causal_mask
    
    # Padding mask for source sequences
    src_mask = create_padding_mask(src_sequences, pad_token)
    
    return tgt_mask, src_mask

def apply_mask_to_attention_scores(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to attention scores
    
    Args:
        scores: Attention scores (batch_size, num_heads, seq_length, seq_length)
        mask: Mask to apply
        
    Returns:
        Masked attention scores
    """
    # Add mask to scores (mask contains large negative values)
    masked_scores = scores + mask
    
    return masked_scores

def visualize_masks(sequences: np.ndarray, mask: np.ndarray, title: str = "Mask Visualization"):
    """
    Visualize masks for debugging
    
    Args:
        sequences: Input sequences
        mask: Mask to visualize
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Plot sequences
    plt.subplot(1, 2, 1)
    plt.imshow(sequences, cmap='viridis', aspect='auto')
    plt.title('Input Sequences')
    plt.xlabel('Sequence Position')
    plt.ylabel('Batch')
    plt.colorbar()
    
    # Plot mask
    plt.subplot(1, 2, 2)
    # Remove the extra dimensions for visualization
    mask_vis = mask.squeeze()
    plt.imshow(mask_vis, cmap='Reds', aspect='auto')
    plt.title('Mask')
    plt.xlabel('Sequence Position')
    plt.ylabel('Batch')
    plt.colorbar()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
