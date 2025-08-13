"""
Dataset and DataLoader for Translation Tasks
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from .tokenization import Tokenizer, build_vocabulary, create_sample_data
from .collate import collate_batch


class TranslationDataset:
    """
    Dataset for translation tasks
    """
    
    def __init__(self, source_texts: List[str], target_texts: List[str], 
                 source_tokenizer: Tokenizer, target_tokenizer: Tokenizer,
                 max_source_length: Optional[int] = None,
                 max_target_length: Optional[int] = None):
        """
        Initialize translation dataset
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Tokenize all texts
        self.source_sequences = []
        self.target_sequences = []
        
        for src_text, tgt_text in zip(source_texts, target_texts):
            # Tokenize source (no special tokens for input)
            src_tokens = self.source_tokenizer.encode(src_text, add_special_tokens=False)
            
            # Tokenize target (with special tokens)
            tgt_tokens = self.target_tokenizer.encode(tgt_text, add_special_tokens=True)
            
            # Apply length limits
            if max_source_length:
                src_tokens = src_tokens[:max_source_length]
            if max_target_length:
                tgt_tokens = tgt_tokens[:max_target_length]
            
            self.source_sequences.append(src_tokens)
            self.target_sequences.append(tgt_tokens)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.source_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (source_sequence, target_sequence)
        """
        return self.source_sequences[idx], self.target_sequences[idx]
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """
        Get vocabulary sizes
        
        Returns:
            Tuple of (source_vocab_size, target_vocab_size)
        """
        return len(self.source_tokenizer), len(self.target_tokenizer)


def create_data_loader(dataset: TranslationDataset, batch_size: int = 32,
                      shuffle: bool = True, collate_fn=None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create data loader for translation dataset
    
    Args:
        dataset: Translation dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        collate_fn: Collation function (if None, uses default)
        
    Returns:
        List of batches as (source_batch, target_batch) tuples
    """
    if collate_fn is None:
        collate_fn = collate_batch
    
    # Create indices
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    # Create batches
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = [dataset[idx] for idx in batch_indices]
        
        # Collate batch
        source_batch, target_batch = collate_fn(batch_data, 
                                               source_pad_idx=dataset.source_tokenizer.pad_idx,
                                               target_pad_idx=dataset.target_tokenizer.pad_idx)
        
        batches.append((source_batch, target_batch))
    
    return batches


def split_dataset(dataset: TranslationDataset, train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
    """
    Split dataset into train/validation/test sets
    
    Args:
        dataset: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Split source and target texts
    train_source = dataset.source_texts[:train_size]
    train_target = dataset.target_texts[:train_size]
    
    val_source = dataset.source_texts[train_size:train_size + val_size]
    val_target = dataset.target_texts[train_size:train_size + val_size]
    
    test_source = dataset.source_texts[train_size + val_size:]
    test_target = dataset.target_texts[train_size + val_size:]
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_source, train_target,
        dataset.source_tokenizer, dataset.target_tokenizer,
        dataset.max_source_length, dataset.max_target_length
    )
    
    val_dataset = TranslationDataset(
        val_source, val_target,
        dataset.source_tokenizer, dataset.target_tokenizer,
        dataset.max_source_length, dataset.max_target_length
    )
    
    test_dataset = TranslationDataset(
        test_source, test_target,
        dataset.source_tokenizer, dataset.target_tokenizer,
        dataset.max_source_length, dataset.max_target_length
    )
    
    return train_dataset, val_dataset, test_dataset


def create_sample_datasets() -> Tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
    """
    Create sample train/validation/test datasets for testing
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create sample data
    source_texts, target_texts = create_sample_data()
    
    # Build vocabularies
    source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=30)
    target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=30)
    
    # Create tokenizers
    source_tokenizer = Tokenizer(source_vocab)
    target_tokenizer = Tokenizer(target_vocab)
    
    # Create full dataset
    full_dataset = TranslationDataset(
        source_texts, target_texts,
        source_tokenizer, target_tokenizer,
        max_source_length=10,
        max_target_length=12
    )
    
    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test dataset creation
    print("📊 Testing Dataset Creation")
    print("=" * 40)
    
    # Create sample datasets
    train_dataset, val_dataset, test_dataset = create_sample_datasets()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test a sample
    src_seq, tgt_seq = train_dataset[0]
    print(f"\nSample source sequence: {src_seq}")
    print(f"Sample target sequence: {tgt_seq}")
    
    # Test data loader
    train_loader = create_data_loader(train_dataset, batch_size=2, shuffle=False)
    print(f"\nNumber of batches: {len(train_loader)}")
    
    for i, (src_batch, tgt_batch) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Source batch shape: {src_batch.shape}")
        print(f"  Target batch shape: {tgt_batch.shape}")
        break
    
    print("\n✅ Dataset test completed!")
