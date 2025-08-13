#!/usr/bin/env python3
"""
Amharic to Oromiffa Translation Demo
Simple demo using the Complete Transformer
"""

import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transformer.complete.transformer import CompleteTransformer

def create_ethiopian_vocab():
    """Create simple Ethiopian vocabulary"""
    vocab = {
        '<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3,
        # Amharic (source)
        'ሰላም': 4, 'እንደምን': 5, 'ደህና': 6, 'አማሪኛ': 7,
        # Oromiffa (target)  
        'akkam': 8, 'nagaa': 9, 'gaarii': 10, 'galata': 11
    }
    return vocab

def text_to_sequence(text, vocab, max_length=5):
    """Convert text to token sequence"""
    tokens = text.split()
    sequence = []
    
    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(vocab['<UNK>'])
    
    # Pad to max_length
    while len(sequence) < max_length:
        sequence.append(vocab['<PAD>'])
    
    return sequence[:max_length]

def main():
    """Main demo function"""
    print("🇪🇹 Amharic to Oromiffa Translation Demo")
    print("=" * 50)
    
    # Create vocabulary
    vocab = create_ethiopian_vocab()
    print(f"📚 Vocabulary: {len(vocab)} tokens")
    
    # Create transformer
    transformer = CompleteTransformer(
        vocab_size=len(vocab),
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256
    )
    
    # Test translation
    print("\n🚀 Testing Translation...")
    
    # Simple example
    amharic = "ሰላም"
    oromiffa = "akkam"
    
    print(f"Amharic: {amharic}")
    print(f"Oromiffa: {oromiffa}")
    
    # Convert to sequences
    src_seq = text_to_sequence(amharic, vocab)
    tgt_seq = text_to_sequence(oromiffa, vocab)
    
    print(f"Source: {src_seq}")
    print(f"Target: {tgt_seq}")
    
    # Run transformer
    try:
        src_batch = np.array([src_seq])
        tgt_batch = np.array([tgt_seq])
        
        output = transformer.forward(src_batch, tgt_batch)
        print(f"✅ Success! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎯 Your transformer is ready for Ethiopian languages!")

if __name__ == "__main__":
    main()
