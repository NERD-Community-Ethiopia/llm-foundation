#!/usr/bin/env python3
"""
Checkpoint 5 & 6 Integration Demo
Demonstrates how the Complete Transformer (Checkpoint 5) works with Data Preparation (Checkpoint 6)
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
from src.complete_transformer import Transformer
from src.data_preparation import (
    Tokenizer, build_vocabulary, create_sample_data,
    TranslationDataset, create_data_loader, split_dataset,
    collate_batch, create_padding_mask, create_causal_mask
)


def demo_integration():
    """Demonstrate integration between Checkpoint 5 and 6"""
    print("🚀 Checkpoint 5 & 6 Integration Demo")
    print("=" * 60)
    print("Complete Transformer + Data Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Data Preparation (Checkpoint 6)
    print("\n📊 STEP 1: Data Preparation (Checkpoint 6)")
    print("-" * 50)
    
    # Create sample translation data
    source_texts, target_texts = create_sample_data()
    print(f"Sample data: {len(source_texts)} translation pairs")
    print(f"Example: '{source_texts[0]}' -> '{target_texts[0]}'")
    
    # Build vocabularies
    source_vocab = build_vocabulary(source_texts, min_freq=1, max_vocab_size=50)
    target_vocab = build_vocabulary(target_texts, min_freq=1, max_vocab_size=50)
    
    print(f"Source vocabulary size: {len(source_vocab)}")
    print(f"Target vocabulary size: {len(target_vocab)}")
    
    # Create tokenizers
    source_tokenizer = Tokenizer(source_vocab)
    target_tokenizer = Tokenizer(target_vocab)
    
    # Test tokenization
    test_text = "hello world"
    encoded = source_tokenizer.encode(test_text)
    decoded = source_tokenizer.decode(encoded)
    print(f"Tokenization test: '{test_text}' -> {encoded} -> '{decoded}'")
    
    # Create dataset
    dataset = TranslationDataset(
        source_texts, target_texts,
        source_tokenizer, target_tokenizer,
        max_source_length=10,
        max_target_length=12
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create data loader
    train_loader = create_data_loader(train_dataset, batch_size=2)
    print(f"Number of batches: {len(train_loader)}")
    
    # Get a sample batch
    src_batch, tgt_batch = train_loader[0]
    print(f"Sample batch shapes: {src_batch.shape}, {tgt_batch.shape}")
    
    # Create attention masks
    src_pad_mask = create_padding_mask(src_batch, pad_idx=source_tokenizer.pad_idx)
    tgt_pad_mask = create_padding_mask(tgt_batch, pad_idx=target_tokenizer.pad_idx)
    causal_mask = create_causal_mask(tgt_batch.shape[1])
    
    print(f"Mask shapes: {src_pad_mask.shape}, {tgt_pad_mask.shape}, {causal_mask.shape}")
    
    # Step 2: Complete Transformer (Checkpoint 5)
    print("\n🧠 STEP 2: Complete Transformer (Checkpoint 5)")
    print("-" * 50)
    
    # Create transformer model
    transformer = Transformer(
        vocab_size=len(target_vocab),
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_length=20
    )
    
    print(f"Transformer created with:")
    print(f"  - Vocabulary size: {len(target_vocab)}")
    print(f"  - Model dimension: 64")
    print(f"  - Number of heads: 4")
    print(f"  - Number of layers: 2")
    print(f"  - Feed-forward dimension: 128")
    
    # Step 3: Integration - Forward Pass
    print("\n🔗 STEP 3: Integration - Forward Pass")
    print("-" * 50)
    
    # Forward pass with real data
    logits = transformer.forward(src_batch, tgt_batch, src_key_padding_mask=src_pad_mask)
    print(f"Forward pass successful!")
    print(f"Input shapes: {src_batch.shape}, {tgt_batch.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Step 4: Integration - Generation
    print("\n🎯 STEP 4: Integration - Text Generation")
    print("-" * 50)
    
    # Generate translation for a sample input
    sample_src = source_texts[0]
    sample_src_tokens = source_tokenizer.encode(sample_src, add_special_tokens=False)
    sample_src_batch = np.array([sample_src_tokens])
    
    print(f"Generating translation for: '{sample_src}'")
    print(f"Source tokens: {sample_src_tokens}")
    
    # Generate translation
    generated_tokens = transformer.generate(
        sample_src_batch,
        max_length=15,
        start_token=target_tokenizer.start_idx,
        end_token=target_tokenizer.end_idx
    )
    
    # Decode generated tokens
    generated_text = target_tokenizer.decode(generated_tokens[0], remove_special_tokens=True)
    print(f"Generated tokens: {generated_tokens[0]}")
    print(f"Generated text: '{generated_text}'")
    
    # Step 5: Integration - Training Preparation
    print("\n🏋️ STEP 5: Integration - Training Preparation")
    print("-" * 50)
    
    # Prepare training data with proper masks
    for i, (src_batch, tgt_batch) in enumerate(train_loader):
        if i >= 2:  # Just show first 2 batches
            break
            
        # Create masks for this batch
        src_pad_mask = create_padding_mask(src_batch, pad_idx=source_tokenizer.pad_idx)
        tgt_pad_mask = create_padding_mask(tgt_batch, pad_idx=target_tokenizer.pad_idx)
        
        # Forward pass with masks
        logits = transformer.forward(src_batch, tgt_batch, src_key_padding_mask=src_pad_mask)
        
        print(f"Batch {i+1}:")
        print(f"  Source shape: {src_batch.shape}")
        print(f"  Target shape: {tgt_batch.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Source padding mask: {src_pad_mask.sum()} padding tokens")
        print(f"  Target padding mask: {tgt_pad_mask.sum()} padding tokens")
    
    # Step 6: Summary
    print("\n✅ INTEGRATION SUMMARY")
    print("=" * 60)
    print("✅ Data Preparation (Checkpoint 6):")
    print("   • Word-level tokenization with special tokens")
    print("   • Frequency-based vocabulary building")
    print("   • Dataset creation with train/val/test splitting")
    print("   • Dynamic batching with proper padding")
    print("   • Attention mask generation (padding + causal)")
    
    print("\n✅ Complete Transformer (Checkpoint 5):")
    print("   • Full encoder-decoder architecture")
    print("   • Multi-head attention with masking support")
    print("   • Positional encoding and layer normalization")
    print("   • Autoregressive text generation")
    print("   • Training infrastructure ready")
    
    print("\n✅ Integration Working:")
    print("   • Data flows seamlessly from preparation to transformer")
    print("   • Proper masking ensures attention works correctly")
    print("   • Tokenization and generation work end-to-end")
    print("   • Ready for actual training!")
    
    print("\n🎯 Next Steps:")
    print("   • Implement training loop (Checkpoint 7)")
    print("   • Add loss computation and optimization")
    print("   • Add model checkpointing and saving")
    print("   • Add training monitoring and logging")
    
    print("\n🚀 Integration demo completed successfully!")


if __name__ == "__main__":
    demo_integration()
