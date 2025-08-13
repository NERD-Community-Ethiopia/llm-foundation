"""
Collation functions for batching translation data
"""
import numpy as np
from typing import List, Tuple


def collate_batch(batch_data: List[Tuple[List[int], List[int]]], 
                  source_pad_idx: int = 0, target_pad_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collate a batch of translation data with padding
    
    Args:
        batch_data: List of (source_sequence, target_sequence) tuples
        source_pad_idx: Padding index for source sequences
        target_pad_idx: Padding index for target sequences
        
    Returns:
        Tuple of (source_batch, target_batch) as numpy arrays
    """
    # Separate source and target sequences
    source_sequences = [item[0] for item in batch_data]
    target_sequences = [item[1] for item in batch_data]
    
    # Find maximum lengths
    max_source_len = max(len(seq) for seq in source_sequences)
    max_target_len = max(len(seq) for seq in target_sequences)
    
    # Pad sequences
    source_batch = []
    target_batch = []
    
    for src_seq, tgt_seq in zip(source_sequences, target_sequences):
        # Pad source sequence
        src_padded = src_seq + [source_pad_idx] * (max_source_len - len(src_seq))
        source_batch.append(src_padded)
        
        # Pad target sequence
        tgt_padded = tgt_seq + [target_pad_idx] * (max_target_len - len(tgt_seq))
        target_batch.append(tgt_padded)
    
    # Convert to numpy arrays
    source_batch = np.array(source_batch, dtype=np.int64)
    target_batch = np.array(target_batch, dtype=np.int64)
    
    return source_batch, target_batch


def create_padding_mask(sequences: np.ndarray, pad_idx: int = 0) -> np.ndarray:
    """
    Create padding mask for sequences
    
    Args:
        sequences: Batch of sequences (batch_size, seq_len)
        pad_idx: Padding token index
        
    Returns:
        Boolean mask where True indicates padding tokens
    """
    return sequences == pad_idx


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask for autoregressive generation
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Lower triangular mask (seq_len, seq_len)
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask.astype(bool)


def create_attention_masks(source_batch: np.ndarray, target_batch: np.ndarray,
                          source_pad_idx: int = 0, target_pad_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create attention masks for transformer training
    
    Args:
        source_batch: Source sequences (batch_size, src_len)
        target_batch: Target sequences (batch_size, tgt_len)
        source_pad_idx: Source padding index
        target_pad_idx: Target padding index
        
    Returns:
        Tuple of (source_padding_mask, target_padding_mask, causal_mask)
    """
    batch_size, src_len = source_batch.shape
    _, tgt_len = target_batch.shape
    
    # Source padding mask (batch_size, src_len)
    source_padding_mask = create_padding_mask(source_batch, source_pad_idx)
    
    # Target padding mask (batch_size, tgt_len)
    target_padding_mask = create_padding_mask(target_batch, target_pad_idx)
    
    # Causal mask (tgt_len, tgt_len)
    causal_mask = create_causal_mask(tgt_len)
    
    return source_padding_mask, target_padding_mask, causal_mask


if __name__ == "__main__":
    # Test collation
    print("🔧 Testing Collation Functions")
    print("=" * 40)
    
    # Create sample batch data
    batch_data = [
        ([1, 2, 3], [4, 5, 6, 7]),
        ([1, 2], [4, 5]),
        ([1, 2, 3, 4], [4, 5, 6, 7, 8])
    ]
    
    # Test collate_batch
    source_batch, target_batch = collate_batch(batch_data, source_pad_idx=0, target_pad_idx=0)
    
    print(f"Source batch shape: {source_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")
    print(f"Source batch:\n{source_batch}")
    print(f"Target batch:\n{target_batch}")
    
    # Test padding masks
    source_padding_mask = create_padding_mask(source_batch, pad_idx=0)
    target_padding_mask = create_padding_mask(target_batch, pad_idx=0)
    
    print(f"\nSource padding mask:\n{source_padding_mask}")
    print(f"Target padding mask:\n{target_padding_mask}")
    
    # Test causal mask
    causal_mask = create_causal_mask(5)
    print(f"\nCausal mask (5x5):\n{causal_mask}")
    
    # Test attention masks
    src_pad_mask, tgt_pad_mask, causal_mask = create_attention_masks(
        source_batch, target_batch
    )
    
    print(f"\nAttention masks created successfully!")
    print(f"Source padding mask shape: {src_pad_mask.shape}")
    print(f"Target padding mask shape: {tgt_pad_mask.shape}")
    print(f"Causal mask shape: {causal_mask.shape}")
    
    print("\n✅ Collation test completed!")
