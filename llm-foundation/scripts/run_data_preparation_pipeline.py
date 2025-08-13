#!/usr/bin/env python3
"""
Data Preparation Pipeline Demo (Checkpoint 6)
"""

import sys
import os

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

for p in (PROJECT_ROOT, SRC_PATH):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
from src.data_preparation.tokenization import Tokenizer, build_vocabulary, create_sample_data
from src.data_preparation.dataset import TranslationDataset, create_data_loader, split_dataset
from src.data_preparation.collate import collate_batch, create_padding_mask, create_causal_mask


def main():
    """Run data preparation demo"""
    print("🚀 Data Preparation Pipeline Demo (Checkpoint 6)")
    print("=" * 60)
    
    # Create sample data
    source_texts, target_texts = create_sample_data()
    print(f"Sample data: {len(source_texts)} translation pairs")
    
    # Build vocabularies
    source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=30)
    target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=30)
    
    print(f"Source vocab size: {len(source_vocab)}")
    print(f"Target vocab size: {len(target_vocab)}")
    
    # Create tokenizers
    source_tokenizer = Tokenizer(source_vocab)
    target_tokenizer = Tokenizer(target_vocab)
    
    # Test tokenization
    test_text = "hello world"
    encoded = source_tokenizer.encode(test_text)
    decoded = source_tokenizer.decode(encoded)
    print(f"\nTokenization test: '{test_text}' -> {encoded} -> '{decoded}'")
    
    # Create dataset
    dataset = TranslationDataset(
        source_texts, target_texts,
        source_tokenizer, target_tokenizer,
        max_source_length=10,
        max_target_length=12
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loader
    train_loader = create_data_loader(train_dataset, batch_size=2)
    print(f"Number of batches: {len(train_loader)}")
    
    # Test a batch
    src_batch, tgt_batch = train_loader[0]
    print(f"Batch shapes: {src_batch.shape}, {tgt_batch.shape}")
    
    # Create masks
    src_pad_mask = create_padding_mask(src_batch, pad_idx=source_tokenizer.pad_idx)
    tgt_pad_mask = create_padding_mask(tgt_batch, pad_idx=target_tokenizer.pad_idx)
    causal_mask = create_causal_mask(tgt_batch.shape[1])
    
    print(f"Mask shapes: {src_pad_mask.shape}, {tgt_pad_mask.shape}, {causal_mask.shape}")
    
    print("\n✅ Data preparation demo completed!")


if __name__ == "__main__":
    main()
